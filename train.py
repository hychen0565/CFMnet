import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from model.mfanet import mfanet
from util.dataset_fss import SemData as dataset
from util import transform, config
from util.util import *
import base64
import io
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

global best_iou
global best_epoch



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='MFANet')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/fold0_resnet50.yaml', help='config file')
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def get_model(args):
    model = mfanet(args)
    optimizer = model.get_optim(args, args.base_lr)
    freeze_modules(model)
    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:
                # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(resume_path))
    return model, optimizer


def main_process():
    return True


def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    logger.info(" -------------------- creating model-------------------")
    model, optimizer = get_model(args)
    logger.info(model)

    # ---------------------- DATASET ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # Train
    train_transform = [
        transform.Resize(args.train_h, args.train_w),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset(split=args.split, shot=args.shot, data_root=args.data_root, \
                         data_list=args.train_list, transform=train_transform, mode='train', data_set=args.data_set)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(h=args.val_size, w=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        val_data = dataset(split=args.split, shot=args.shot, data_root=args.data_root, data_list=args.val_list,
                           transform=val_transform, mode='val', data_set=args.data_set)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # ---------------------- TRAINVAL ----------------------
    global best_miou, best_FBiou, best_piou, best_epoch, keep_epoch, val_num
    global best_miou_m
    best_miou = 0.
    best_FBiou = 0.
    best_piou = 0.
    best_epoch = 0
    val_num = 0
    max_iou = 0.
    max_fbiou = 0
    best_epoch = 0
    filename = 'SD-AANet.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        epoch_log = epoch + 1
        loss_train, mIoU_train, mAcc_train, allAcc_train, mani_loss_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
            writer.add_scalar('mani_loss_train', mani_loss_train, epoch_log)

        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou, class_iou_class = validate(val_loader, model)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

                if class_miou > max_iou:
                    max_iou = class_miou
                    best_epoch = epoch
                    if os.path.exists(filename):
                        os.remove(filename)
                    filename = args.save_path + '/train_epoch_' + str(epoch) + '_' + str(max_iou) + '.pth'
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, filename)
                if mIoU_val > max_fbiou:
                    max_fbiou = mIoU_val
                logger.info(
                    'Best Epoch {:.1f}, Best IOU {:.4f} Best FB-IoU {:4F}'.format(best_epoch, max_iou, max_fbiou))

                log_file_path = os.path.join(args.save_path, 'val_results_log.txt')
                prefix = "[{}/{}] ".format(epoch, args.epochs - 1)
                log_line = "meanIoU---Val result: mIoU {:.4f}. ".format(class_miou)
                for i in range(len(class_iou_class)):
                    log_line += "Class_{} Result: iou {:.4f}. ".format(i + 1, class_iou_class[i])
                log_line += "FBIoU---Val result: mIoU {:.4f}".format(mIoU_val)
                with open(log_file_path, 'a') as f:
                    f.write(prefix + log_line + '\n')


def train(train_loader, model, optimizer, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Train <<<<<<<<<<<<<<<<<<')
    batch_time = AverageMeter()
    data_time = AverageMeter()

    manifold_loss_meter = AverageMeter()

    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    print('Warmup: {}'.format(args.warmup))

    for i, (input, target, s_input, s_mask, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        index_split = -1
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                               index_split=index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output, total_loss, seg_loss, m_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)

        loss = total_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        manifold_loss_meter.update(m_loss.item(), n)
        main_loss_meter.update(seg_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'SegLoss {main_loss_meter.val:.4f} '
                        'ManiLoss {manifold_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch, args.epochs - 1, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          manifold_loss_meter=manifold_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>end train<<<<<<<<<<<<<<<<<<<<<<<<<<')
    return main_loss_meter.avg, mIoU, mAcc, allAcc, manifold_loss_meter.avg

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def img_to_svg_base64(img_np):
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def create_svg_with_images(imgs_data, titles, save_path):
    num_imgs = len(imgs_data)
    if num_imgs == 0:
        return

    h, w = imgs_data[0].shape[:2]

    svg_width = num_imgs * (w + 20) + 20
    svg_height = h + 60

    svg_root = Element('svg')
    svg_root.set('xmlns', 'http://www.w3.org/2000/svg')
    svg_root.set('width', str(svg_width))
    svg_root.set('height', str(svg_height))

    for i, img in enumerate(imgs_data):
        x_offset = 10 + i * (w + 20)

        text_elem = SubElement(svg_root, 'text')
        text_elem.set('x', str(x_offset + w / 2))
        text_elem.set('y', str(h + 40))
        text_elem.set('text-anchor', 'middle')
        text_elem.set('font-size', '16')
        text_elem.set('font-family', 'Arial')
        text_elem.text = titles[i]

        image_elem = SubElement(svg_root, 'image')
        image_elem.set('x', str(x_offset))
        image_elem.set('y', '10')
        image_elem.set('width', str(w))
        image_elem.set('height', str(h))
        image_elem.set('href', 'data:image/png;base64,' + img_to_svg_base64(img))

    xml_str = minidom.parseString(tostring(svg_root)).toprettyxml(indent="   ")
    with open(save_path, 'w') as f:
        f.write(xml_str)

def validate(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    split_gap = len(val_loader.dataset.val_class)
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model.eval()
    end = time.time()
    val_start = end
    test_num = len(val_loader)
    iter_num = 0
    total_time = 0

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array([item * value_scale for item in mean])
    std = np.array([item * value_scale for item in std])

    palette = get_palette(args.classes)

    if main_process():
        vis_dir = os.path.join(args.result_path, 'vis_svg')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    for e in range(1):
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if (iter_num - 1) * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)

            start_time = time.time()
            output = model(s_x=s_input, s_y=s_mask, x=input, y=target)
            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda() * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()
                output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)

            loss = criterion(output, target)
            n = input.size(0)
            loss = torch.mean(loss)
            output = output.max(1)[1]

            if main_process():
                for j in range(output.size(0)):
                    query_np = input[j].cpu().numpy().transpose(1, 2, 0)
                    query_np = (query_np * std) + mean
                    query_np = query_np.astype(np.uint8)
                    gt_np = target[j].cpu().numpy()
                    gt_color = np.zeros((gt_np.shape[0], gt_np.shape[1], 3), dtype=np.uint8)
                    for c in range(args.classes):
                        gt_color[gt_np == c] = palette[c * 3:c * 3 + 3]
                    pred_np = output[j].cpu().numpy()
                    pred_color = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
                    for c in range(args.classes):
                        pred_color[pred_np == c] = palette[c * 3:c * 3 + 3]
                    s_imgs_np = []
                    s_titles = []
                    shot_num = s_input.size(1)
                    for k in range(shot_num):
                        s_img = s_input[j, k].cpu().numpy().transpose(1, 2, 0)
                        s_img = (s_img * std) + mean
                        s_img = s_img.astype(np.uint8)
                        s_imgs_np.append(s_img)
                        s_titles.append(f'Supp_{k}')
                        s_m = s_mask[j, k].cpu().numpy()
                        s_m_color = np.zeros((s_m.shape[0], s_m.shape[1], 3), dtype=np.uint8)

                        for c in range(args.classes):
                            s_m_color[s_m == c] = palette[c * 3:c * 3 + 3]
                        s_imgs_np.append(s_m_color)
                        s_titles.append(f'Supp_Mask_{k}')

                    all_imgs = [query_np] + s_imgs_np + [gt_color, pred_color]
                    all_titles = ['Query'] + s_titles + ['GT', 'Pred']

                    save_idx = iter_num * args.batch_size_val + j
                    svg_path = os.path.join(vis_dir, f'{save_idx:04d}.svg')
                    create_svg_with_images(all_imgs, all_titles, svg_path)

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            subcls = subcls[0].cpu().numpy()[0]
            if 0 <= subcls < split_gap:
                class_intersection_meter[subcls] += intersection[1]
                class_union_meter[subcls] += union[1]
            else:
                print(f"Warning: subcls {subcls} out of range [0, {split_gap - 1}]")

            accuracy = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (iter_num % test_num == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time, batch_time=batch_time,
                                                              loss_meter=loss_meter, accuracy=accuracy))

    val_time = time.time() - val_start
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)

    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))
    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou, class_iou_class

if __name__ == '__main__':
    main()
