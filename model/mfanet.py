import torch
from torch import nn
import torch.nn.functional as F
import model.resnet as models
import model.vgg as vgg_models
import numpy as np


class GBM(nn.Module):
    def __init__(self, in_channels, out_channels=None, mid_ratio=8):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = max(in_channels // mid_ratio, 16)
        self.pw1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dw = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                            groups=mid_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_channels)
        self.pw2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.skip = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        g = self.gate(x)
        out = self.relu(self.bn1(self.pw1(x)))
        out = self.relu(self.bn_dw(self.dw(out)))
        out = self.bn2(self.pw2(out))
        return self.relu(residual + out * g)


class MCA(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        c_g = channels // self.groups
        assert c_g > 0

        self.softmax = nn.Softmax(-1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c_g, c_g)
        self.conv1x1 = nn.Conv2d(c_g, c_g, 1, bias=False)

        self.dw3x3 = nn.Conv2d(c_g, c_g, 3, padding=1, groups=c_g, bias=False)
        self.pool_local = nn.AdaptiveAvgPool2d((3, 3))
        self.hf_gate = nn.Sequential(
            nn.Conv2d(c_g, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.dw3x3(group_x)
        x_local = self.pool_local(x2)
        x_local_ch = x_local.mean(dim=[2, 3], keepdim=True)
        x11 = self.softmax(x_local_ch.reshape(b * self.groups, -1, 1).permute(0, 2, 1))

        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(x_local_ch.reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        hf_mask = self.hf_gate(x2)
        final_weights = weights.sigmoid() * (0.5 + hf_mask)
        return (group_x * final_weights).reshape(b, c, h, w)


class GLCM(nn.Module):
    def __init__(self, d_model, reduction=16):
        super().__init__()
        self.local_attn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.global_attn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                                         nn.Linear(d_model, d_model))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
            pool = torch.mean(x_flat, dim=1, keepdim=True)
            attn = self.sigmoid(self.local_attn(x_flat) + self.global_attn(pool))
            return x * attn.transpose(1, 2).reshape(b, c, h, w)
        pool = torch.mean(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.local_attn(x) + self.global_attn(pool))


class PFE(nn.Module):
    def __init__(self, in_c, out_c, drop_rate=0.2):
        super().__init__()
        self.embed = nn.Sequential(GBM(in_c, out_c), nn.Dropout2d(drop_rate))

    def forward(self, x):
        return self.embed(x)


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -harr_wav_H[0, 0]

        def make_filter(w1, w2):
            f = np.transpose(w1) * w2
            return torch.from_numpy(f).unsqueeze(0).float()

        for name, filt in [('LL', make_filter(harr_wav_L, harr_wav_L)),
                           ('LH', make_filter(harr_wav_L, harr_wav_H)),
                           ('HL', make_filter(harr_wav_H, harr_wav_L)),
                           ('HH', make_filter(harr_wav_H, harr_wav_H))]:
            conv = nn.Conv2d(in_channels, in_channels, kernel_size=2,
                             stride=2, padding=0, bias=False, groups=in_channels)
            conv.weight.data = filt.unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
            conv.weight.requires_grad_(False)
            setattr(self, name, conv)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class FDPG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.inter_c = channels // 2
        self.channels = channels

        self.gbm = GBM(channels, self.inter_c)
        self.wave_pool = WavePool(self.inter_c)

        self.mca_h = MCA(self.inter_c)
        self.mca_v = MCA(self.inter_c)
        self.mca_d = MCA(self.inter_c)
        self.glcm_low = GLCM(self.inter_c)

        self.dir_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.inter_c * 3, 3),
            nn.Softmax(dim=1)
        )

        self.freq_fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.theta = GBM(channels, self.inter_c)
        self.phi = GBM(channels, self.inter_c)
        self.W = nn.Sequential(GBM(self.inter_c, channels), nn.BatchNorm2d(channels))
        self.glcm_global = GLCM(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.size(0)
        gbm_feat = self.gbm(x)

        LL, LH, HL, HH = self.wave_pool(gbm_feat)

        hl_enh = self.mca_h(HL)
        lh_enh = self.mca_v(LH)
        hh_enh = self.mca_d(HH)

        target_size = LL.shape[2:]
        hl_up = F.interpolate(hl_enh, size=target_size, mode='bilinear', align_corners=False)
        lh_up = F.interpolate(lh_enh, size=target_size, mode='bilinear', align_corners=False)
        hh_up = F.interpolate(hh_enh, size=target_size, mode='bilinear', align_corners=False)

        dir_cat = torch.cat([hl_up, lh_up, hh_up], dim=1)
        w_dir = self.dir_weight(dir_cat)
        w_h, w_v, w_d = w_dir[:, 0:1, None, None], w_dir[:, 1:2, None, None], w_dir[:, 2:3, None, None]
        high_freq_feat = w_h * hl_up + w_v * lh_up + w_d * hh_up

        low_freq_feat = self.glcm_low(LL)

        w_high, w_low = self.freq_fusion_weight.softmax(dim=0)
        mca_feat = w_high * high_freq_feat + w_low * low_freq_feat

        original_size = x.shape[2:]
        mca_feat = F.interpolate(mca_feat, size=original_size, mode='bilinear', align_corners=False)
        theta_x = self.theta(x)
        phi_x = self.phi(x)
        g_flat = mca_feat.view(b, self.inter_c, -1).permute(0, 2, 1)
        theta_flat = theta_x.view(b, self.inter_c, -1).permute(0, 2, 1)
        phi_flat = phi_x.view(b, self.inter_c, -1)
        attn = F.softmax(torch.bmm(theta_flat, phi_flat), dim=-1)
        y = torch.bmm(attn, g_flat).permute(0, 2, 1).contiguous().view(b, self.inter_c, *x.shape[2:])
        W_y = self.W(y)
        z = self.gamma * W_y + x
        return self.glcm_global(z)


class PMA(nn.Module):
    def __init__(self, in_channels, mid_channels=None, sinkhorn_iter=3, epsilon=0.05, patch_size=4):
        super().__init__()
        mid_channels = mid_channels or in_channels // 4
        self.sinkhorn_iter = sinkhorn_iter
        self.epsilon = epsilon
        self.patch_size = patch_size

        self.transform_q = nn.Sequential(GBM(in_channels, mid_channels), nn.BatchNorm2d(mid_channels))
        self.transform_k = nn.Sequential(GBM(in_channels, mid_channels), nn.BatchNorm2d(mid_channels))
        self.adapter = nn.Sequential(nn.Conv2d(1, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels),
                                     nn.Sigmoid())

    def forward(self, base_feat, guidance_feat):
        b, c, h, w = base_feat.shape
        if guidance_feat.shape[-2:] != (h, w):
            guidance_feat = F.interpolate(guidance_feat, (h, w), mode='bilinear', align_corners=False)

        q = self.transform_q(guidance_feat)
        k = self.transform_k(base_feat)

        ps = self.patch_size
        q_patch = F.avg_pool2d(q, kernel_size=ps, stride=ps)
        k_patch = F.avg_pool2d(k, kernel_size=ps, stride=ps)

        b_p, c_p, h_p, w_p = q_patch.shape
        n_patch = h_p * w_p

        q_flat = q_patch.flatten(2).permute(0, 2, 1)
        k_flat = k_patch.flatten(2).permute(0, 2, 1)

        cost_matrix = torch.cdist(q_flat, k_flat, p=2)
        cost_matrix = cost_matrix / (cost_matrix.max() + 1e-8)

        K = torch.exp(-cost_matrix / self.epsilon)
        u = torch.ones(b_p, n_patch, 1, device=cost_matrix.device) / n_patch
        v = torch.ones(b_p, n_patch, 1, device=cost_matrix.device) / n_patch
        for _ in range(self.sinkhorn_iter):
            v = v / (K.transpose(1, 2) @ u + 1e-8)
            u = u / (K @ v + 1e-8)
        transport_plan = u * K * v.transpose(1, 2)
        align_weight, _ = transport_plan.max(dim=2)
        align_weight = align_weight.reshape(b, 1, h_p, w_p)
        align_weight = F.interpolate(align_weight, size=(h, w), mode='bilinear', align_corners=False)
        align_weight = self.adapter(align_weight)
        return (1 - align_weight) * base_feat + align_weight * guidance_feat

class CMAD(nn.Module):
    """Cross-Granularity Optimal-Transport-guided Alignment Decoder"""
    def __init__(self, dim=256, drop_rate=0.3):
        super().__init__()
        self.mca = MCA(dim)
        self.pma_fusion = PMA(dim)
        self.dropout = nn.Dropout2d(drop_rate)
        self.up1 = nn.Sequential(GBM(dim, dim), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.up2 = nn.Sequential(GBM(dim, dim), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.pma_low = PMA(dim)
        self.pma_high = PMA(dim)
        self.ms_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim // 4, 3, padding=2 ** i, dilation=2 ** i, bias=False),
                          nn.BatchNorm2d(dim // 4), nn.ReLU(inplace=True))
            for i in range(3)
        ])
        self.fusion = GBM(dim + 3 * (dim // 4), dim)
        self.detail_branch = nn.Sequential(GBM(dim, dim), MCA(dim), GBM(dim, dim))
        self.semantic_branch = nn.Sequential(GBM(dim, dim), GLCM(dim))
        self.head = nn.Sequential(GBM(dim, dim // 2), nn.Dropout2d(0.2), nn.Conv2d(dim // 2, 2, 1, bias=False))

    def forward(self, query_feat, support_feat, merge_feat, h, w):
        support_enhanced = self.mca(support_feat)
        fused_feat = self.pma_fusion(query_feat, support_enhanced)
        fused_feat = self.dropout(fused_feat)

        x2 = self.up1(merge_feat)
        x4 = self.up2(x2)
        x_fused = self.pma_low(x4, F.interpolate(x2, size=x4.shape[2:], mode='bilinear'))
        x_fused = self.pma_high(x_fused, F.interpolate(fused_feat, size=x4.shape[2:], mode='bilinear'))

        multi_feats = [x_fused] + [conv(x_fused) for conv in self.ms_conv]
        decode_feat = self.fusion(torch.cat(multi_feats, dim=1))
        out = self.detail_branch(decode_feat) + self.semantic_branch(decode_feat)

        seg_out = self.head(out)
        if seg_out.shape[2:] != (h, w):
            seg_out = F.interpolate(seg_out, size=(h, w), mode='bilinear', align_corners=True)
        return seg_out


class GBM_ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(GBM(in_c, out_c), GBM(out_c, out_c))
        self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class CFMANet(nn.Module):
    def __init__(self, args):
        super().__init__()
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        self.classes = args.classes
        self.pretrained = True
        models.BatchNorm = BatchNorm
        self.layers = args.layers

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = self._get_vgg16_layer(vgg16)
        else:
            print(f'>>>>>>>>> Using ResNet {self.layers} <<<<<<<<<')
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        reduce_dim = 256
        fea_dim = 512 + 256 if self.vgg else 1024 + 512

        self.pfe_query = PFE(fea_dim, reduce_dim)
        self.pfe_support = PFE(fea_dim, reduce_dim)
        self.fdpg = FDPG(reduce_dim)
        self.cmad = CMAD(reduce_dim)
        self.pma_ref = PMA(reduce_dim)
        self.pma_att = PMA(reduce_dim)

        self.init_merge = nn.Sequential(GBM(reduce_dim * 3 + 1, reduce_dim), nn.BatchNorm2d(reduce_dim),
                                        nn.ReLU(inplace=True))
        self.feature_enhance = nn.Sequential(GBM_ResBlock(reduce_dim, reduce_dim), MCA(reduce_dim),
                                             GBM_ResBlock(reduce_dim, reduce_dim))

        simple_in = 512 if self.vgg else 2048
        self.simple_proj = nn.Sequential(nn.Conv2d(simple_in, reduce_dim, 1, bias=False), nn.ReLU(inplace=True))
        self.simple_enhance = nn.Sequential(GBM(reduce_dim, reduce_dim), GBM(reduce_dim, reduce_dim))
        self.simple_head = nn.Conv2d(reduce_dim, self.classes, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self._freeze_backbone()

    def _freeze_backbone(self):
        for m in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters(): p.requires_grad = False

    def _get_vgg16_layer(self, model):
        ranges = [range(0, 7), range(7, 14), range(14, 24), range(24, 34), range(34, 43)]
        return tuple(nn.Sequential(*[model.features[i] for i in r]) for r in ranges)

    def get_optim(self, args, LR):
        return torch.optim.AdamW (self.parameters(), lr=LR, weight_decay=args.weight_decay)


    def _compute_spm_prior(self, q4, supp_list, mask_list, target_size):
        corr_list, eps = [], 1e-7
        for i, supp_feat in enumerate(supp_list):
            size = supp_feat.size(2)
            mask = F.interpolate(mask_list[i], size=(size, size), mode='bilinear', align_corners=True)
            masked_supp = supp_feat * mask
            b, c, sp = q4.size(0), q4.size(1), size * size
            q = q4.view(b, c, -1)
            s = masked_supp.view(b, c, -1).permute(0, 2, 1)
            q_norm = torch.norm(q, 2, 1, True)
            s_norm = torch.norm(s, 2, 2, True)
            sim = torch.bmm(s, q) / (torch.bmm(s_norm, q_norm) + eps)
            sim = sim.max(1)[0].view(b, sp)
            sim = (sim - sim.min(1, keepdim=True)[0]) / (
                    sim.max(1, keepdim=True)[0] - sim.min(1, keepdim=True)[0] + eps)
            corr = sim.view(b, 1, size, size)
            corr = F.interpolate(corr, size=target_size, mode='bilinear', align_corners=True)
            corr_list.append(corr)
        return torch.stack(corr_list).mean(dim=0)

    def patch_level_loss(self, query_feat, prototype):
        b, c, h, w = query_feat.shape
        q_flat = query_feat.flatten(2).permute(0, 2, 1)
        p_flat = prototype.flatten(2).permute(0, 2, 1)

        cost_matrix = torch.cdist(q_flat, p_flat, p=2)

        ot_weights = F.softmax(-cost_matrix / 0.1, dim=1)

        w_loss = (ot_weights * cost_matrix).sum(dim=1).mean()
        return w_loss

    def forward(self, x, s_x=None, s_y=None, y=None):
        if s_x is None:
            s_x = torch.zeros(x.size(0), self.shot, 3, x.size(2), x.size(3), device=x.device)
            s_y = torch.zeros(x.size(0), self.shot, x.size(2), x.size(3), device=x.device)
        h, w = x.size()[2:]

        with torch.no_grad():
            q0 = self.layer0(x);
            q1 = self.layer1(q0);
            q2 = self.layer2(q1)
            q3 = self.layer3(q2);
            q4 = self.layer4(q3)
            if self.vgg: q2 = F.interpolate(q2, size=q3.shape[2:], mode='bilinear', align_corners=True)
            query_cat = torch.cat([q3, q2], dim=1)

        query_feat = self.pfe_query(query_cat)
        mask_list, proto_list, supp_high_list = [], [], []

        for i in range(self.shot):
            supp_gt = (s_y[:, i] == 1).float().unsqueeze(1)
            with torch.no_grad():
                s0 = self.layer0(s_x[:, i]);
                s1 = self.layer1(s0);
                s2 = self.layer2(s1)
                s3 = self.layer3(s2);
                s4 = self.layer4(s3)
                if self.vgg: s2 = F.interpolate(s2, size=s3.shape[2:], mode='bilinear', align_corners=True)
                mask = F.interpolate(supp_gt, size=s3.shape[2:], mode='bilinear', align_corners=True)
                supp_cat = torch.cat([s3, s2], dim=1)
                supp_high_list.append(s4);
                mask_list.append(supp_gt)

            supp_feat = self.pfe_support(supp_cat)
            prototype = self.fdpg(supp_feat * mask)
            proto_list.append(prototype)

        prototype = torch.stack(proto_list).mean(dim=0) if self.shot > 1 else proto_list[0]

        query_refined = self.pma_ref(query_feat, prototype)
        query_enhanced = self.pma_att(query_refined, prototype)

        corr_mask = self._compute_spm_prior(q4, supp_high_list, mask_list, query_feat.shape[2:])
        merge_feat = self.init_merge(torch.cat([query_enhanced, prototype, query_feat, corr_mask], dim=1))
        merge_feat = self.feature_enhance(merge_feat)
        seg_out = self.cmad(query_feat=query_enhanced, support_feat=prototype, merge_feat=merge_feat, h=h, w=w)

        if not self.training:
            seg_out = self.max_pool(seg_out)
            seg_out = -self.max_pool(-seg_out)
            return seg_out

        simple_feat = self.simple_enhance(self.simple_proj(q4))
        simple_out = F.interpolate(self.simple_head(simple_feat), size=(h, w), mode='bilinear', align_corners=True)

        main_loss = self.criterion(seg_out, y.long())
        aux_loss = self.criterion(simple_out, y.long())

        manifold_loss = self.patch_level_loss(query_feat, prototype)
        total_loss = main_loss + 0.2 * aux_loss + 0.1 * manifold_loss
        return seg_out.max(1)[1], total_loss, main_loss, manifold_loss


class mfanet(CFMANet): pass