import os
import numpy as np
import random
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def make_dataset(split=0, data_root=None, data_list=None, train_class=None):
    assert split in [0, 1, 2]

    if not os.path.isfile(data_list):
        raise RuntimeError("Image list file do not exist: " + data_list + "\n")

    img_gt_class_list = []
    list_read = open(data_list, encoding='UTF-8-sig').readlines()
    class_img_gt_dict = {}

    for sub_c in train_class:
        class_img_gt_dict[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        if not line:
            continue

        line_split = line.split()
        if len(line_split) < 2:
            print(f"Warning: skipping invalid line {l_idx}: {line}")
            continue

        image_name = os.path.join(data_root, line_split[0])
        temp = line_split[0].replace('Images/', 'GT/')
        gt_name = os.path.join(data_root, temp)
        gt_name = gt_name.replace('jpg', 'png').replace('JPG', 'png')
        image_class = int(line_split[1])

        item = (image_name, gt_name, image_class)

        if image_class in train_class:
            img_gt_class_list.append(item)
            class_img_gt_dict[image_class].append(item)

    print(f"Dataset loaded: {len(img_gt_class_list)} samples")
    print(f"Class distribution: { {k: len(v) for k, v in class_img_gt_dict.items()} }")
    return img_gt_class_list, class_img_gt_dict


class SemData(Dataset):
    def __init__(self, split=0, shot=1, data_root=None, data_list=None,
                 transform=None, mode='train', data_set='fssd12'):
        self.mode = mode
        self.split = split
        self.shot = shot
        self.data_root = data_root
        self.data_list = data_list

        assert data_set in ['fssd12', 'cgfds'], f"Unsupported data_set: {data_set}"

        if data_set == 'fssd12':
            self.class_list = list(range(1, 13))
            if self.split == 2:
                self.train_class = list(range(1, 9))
                self.val_class = list(range(9, 13))
            elif self.split == 1:
                self.train_class = list(range(1, 5)) + list(range(9, 13))
                self.val_class = list(range(5, 9))
            elif self.split == 0:
                self.train_class = list(range(5, 13))
                self.val_class = list(range(1, 5))

        elif data_set == 'cgfds':
            # CGFSDS-9
            self.class_list = list(range(1, 11))

            if self.split == 0:
                self.train_class = list(range(1, 4))
                self.val_class = list(range(5, 11))

            elif self.split == 1:
                self.train_class = [1, 2, 3, 5, 6, 7]
                self.val_class = [8, 9, 10]

            elif self.split == 2:
                self.train_class = [2, 3]
                self.val_class = [1]

        print(f"Mode: {mode}, DataSet: {data_set}, Split: {split}")
        print(f"Train classes: {self.train_class}")
        print(f"Val classes: {self.val_class}")

        if self.mode == 'train':
            self.img_gt_class_list, self.class_img_gt_dict = make_dataset(
                split, data_root, data_list, self.train_class)
        elif self.mode == 'val':
            self.img_gt_class_list, self.class_img_gt_dict = make_dataset(
                split, data_root, data_list, self.val_class)

        self.transform = transform

    def __len__(self):
        return len(self.img_gt_class_list)

    def __getitem__(self, index):
        query_img, query_gt, query_class = self.img_gt_class_list[index]

        query_rgb = cv2.imread(query_img, cv2.IMREAD_COLOR)
        if query_rgb is None:
            raise RuntimeError(f"Cannot read image: {query_img}")
        query_rgb = cv2.cvtColor(query_rgb, cv2.COLOR_BGR2RGB)
        query_rgb = np.float32(query_rgb)

        query_mask = cv2.imread(query_gt, cv2.IMREAD_GRAYSCALE)
        if query_mask is None:
            raise RuntimeError(f"Cannot read mask: {query_gt}")
        query_mask[query_mask != 255] = 0
        query_mask[query_mask == 255] = 1

        if query_rgb.shape[:2] != query_mask.shape:
            raise RuntimeError(f"Shape mismatch: {query_img} vs {query_gt}")

        class_chosen = query_class
        all_img_gt_list = self.class_img_gt_dict[int(class_chosen)]
        num_file = len(all_img_gt_list)

        support_image_path_list = []
        support_gt_path_list = []
        support_idx_list = []

        for k in range(self.shot):
            support_idx = random.randint(0, num_file - 1)
            support_image_path, support_label_path, _ = all_img_gt_list[support_idx]

            max_retry = 10
            retry = 0
            while support_image_path == query_img and retry < max_retry:
                support_idx = random.randint(0, num_file - 1)
                support_image_path, support_label_path, _ = all_img_gt_list[support_idx]
                retry += 1

            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_gt_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = []

        for k in range(self.shot):
            if self.mode == 'train':
                subcls_list.append(self.train_class.index(int(class_chosen)))
            else:
                subcls_list.append(self.val_class.index(int(class_chosen)))

            support_rgb = cv2.imread(support_image_path_list[k], cv2.IMREAD_COLOR)
            support_rgb = cv2.cvtColor(support_rgb, cv2.COLOR_BGR2RGB)
            support_rgb = np.float32(support_rgb)

            support_mask = cv2.imread(support_gt_path_list[k], cv2.IMREAD_GRAYSCALE)
            support_mask[support_mask != 255] = 0
            support_mask[support_mask == 255] = 1

            support_image_list.append(support_rgb)
            support_label_list.append(support_mask)

        raw_label = query_mask.copy()

        if self.transform is not None:
            query_rgb, query_mask = self.transform(query_rgb, query_mask)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(
                    support_image_list[k], support_label_list[k])

        s_x = torch.stack([s.unsqueeze(0) for s in support_image_list], dim=0).squeeze(1)
        s_y = torch.stack([s.unsqueeze(0) for s in support_label_list], dim=0).squeeze(1)

        if self.mode == 'train':
            return query_rgb, query_mask, s_x, s_y, subcls_list
        else:
            return query_rgb, query_mask, s_x, s_y, subcls_list, raw_label
