"""
OusterLSSTransform - Ouster球面投影版Lift-Splat-Shoot视图变换
用于BevFusion†在PALIN数据集上的适配

数学推导（来自extract_labels_from_json.py的point_to_pix反推）：
  已知特征像素(w_f, h_f)，特征步长feat_stride，图像尺寸(W=1024, H=128)：

    u_img = w_f * feat_stride          # 原始图像列（水平）
    v_img = h_f * feat_stride          # 原始图像行（垂直）

    azimuth_deg = u_img / W * 360
    azimuth_deg = (azimuth_deg - shift_degrees + 360) % 360
    theta = -azimuth_deg * π/180       # 等于 atan2(y, x)

    phi = beam_altitude_angles[v_img]  # 仰角（度）

  在深度d处的3D坐标（与OpenPCDet坐标系一致）：
    x = d * cos(phi) * cos(theta)      # 前向
    y = d * cos(phi) * sin(theta)      # 左向
    z = d * sin(phi)                   # 向上（BEV不使用）

配套的YAML参数（voxel=[0.1,0.1,0.2], range=[-51.2,-60,-3,51.2,40.8,5]）：
  BEV尺寸 = range / (voxel_size * backbone_stride) = 102.4/(0.1*8) × 100.8/(0.1*8)
           = 128(X) × 126(Y)
  BEV_RES = 0.8m/cell

文件位置: third_party/fusion/ouster_view_transform.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class OusterLSSTransform(nn.Module):
    """
    Ouster球面投影Lift-Splat-Shoot，用于将透视图像特征提升到BEV空间。

    与原版BevFusion DepthLSSTransform的区别：
    ┌─────────────────┬──────────────────────────┬──────────────────────────┐
    │                 │ 原版 DepthLSS             │ OusterLSS（本模块）       │
    ├─────────────────┼──────────────────────────┼──────────────────────────┤
    │ 相机模型         │ 针孔相机（内参矩阵K）       │ 球面投影（beam角度表）      │
    │ 像素方向         │ K^{-1} [u,v,1]           │ (azimuth, elevation)已知  │
    │ 标定需求         │ 需要内外参标定              │ 无需标定，beam角固定已知    │
    │ 深度估计         │ softmax分布 × D个bin       │ 相同（softmax深度分布）    │
    └─────────────────┴──────────────────────────┴──────────────────────────┘
    """

    def __init__(self, model_cfg):
        super().__init__()

        # ── 基本尺寸配置 ──────────────────────────────────────────────────────
        img_size = model_cfg.IMG_SIZE          # [H, W] = [128, 1024]
        self.img_h, self.img_w = img_size[0], img_size[1]
        self.feat_stride = model_cfg.FEAT_STRIDE  # YOLOv8 P3的步长 = 8
        self.feat_h = self.img_h // self.feat_stride   # 128//8 = 16
        self.feat_w = self.img_w // self.feat_stride   # 1024//8 = 128
        self.shift_degrees = model_cfg.SHIFT_DEGREES   # Bus=180.0, Boston=90.0

        # ── BEV网格配置 ───────────────────────────────────────────────────────
        # 与VoxelResBackBone8x + HeightCompression的输出尺寸完全匹配
        # voxel=[0.1,0.1,0.2], range=[-51.2,-60,-3,51.2,40.8,5], stride=8
        bev_range = model_cfg.BEV_RANGE        # [x_min, y_min, x_max, y_max]
        self.bev_xmin = bev_range[0]           # -51.2
        self.bev_ymin = bev_range[1]           # -60.0
        self.bev_xmax = bev_range[2]           #  51.2
        self.bev_ymax = bev_range[3]           #  40.8
        self.bev_res  = model_cfg.BEV_RES      # 0.8m = voxel_size(0.1) × stride(8)

        # BEV网格尺寸：
        #   X方向: (51.2-(-51.2))/0.8 = 128 cells
        #   Y方向: (40.8-(-60.0))/0.8 = 126 cells
        self.bev_w = round((self.bev_xmax - self.bev_xmin) / self.bev_res)  # 128
        self.bev_h = round((self.bev_ymax - self.bev_ymin) / self.bev_res)  # 126

        # ── 深度bin配置 ───────────────────────────────────────────────────────
        d_cfg = model_cfg.DEPTH_RANGE          # [d_min, d_max, d_step]
        depth_bins = torch.arange(d_cfg[0], d_cfg[1], d_cfg[2])
        self.register_buffer('depth_bins', depth_bins)
        self.D = len(depth_bins)

        # ── 通道配置 ──────────────────────────────────────────────────────────
        self.in_channels  = model_cfg.IN_CHANNELS   # YOLOv8-S P3 = 128
        self.out_channels = model_cfg.OUT_CHANNELS  # 输出BEV通道 = 80

        # ── 加载beam_altitude_angles ──────────────────────────────────────────
        angles_path = model_cfg.BEAM_ALTITUDE_ANGLES_PATH
        self.beam_altitude_angles = self._load_beam_angles(angles_path)

        # ── 预计算frustum → BEV的静态映射查找表 ──────────────────────────────
        bev_x = torch.zeros(self.feat_h, self.feat_w, self.D, dtype=torch.long)
        bev_y = torch.zeros(self.feat_h, self.feat_w, self.D, dtype=torch.long)
        valid = torch.zeros(self.feat_h, self.feat_w, self.D, dtype=torch.bool)
        self._fill_frustum_table(bev_x, bev_y, valid)
        self.register_buffer('frustum_bev_x', bev_x)
        self.register_buffer('frustum_bev_y', bev_y)
        self.register_buffer('frustum_valid', valid)

        # ── 深度预测头（可学习，对应LSS的Lift阶段）───────────────────────────
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.D, kernel_size=1),
        )

        # ── 特征通道压缩（in_channels → out_channels）────────────────────────
        self.feat_compress = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

        n_valid = self.frustum_valid.sum().item()
        print(f"[OusterLSS] 图像: {self.img_h}×{self.img_w}  "
              f"特征图: {self.feat_h}×{self.feat_w} (stride={self.feat_stride})")
        print(f"[OusterLSS] BEV: {self.bev_h}(Y)×{self.bev_w}(X), "
              f"分辨率={self.bev_res}m/cell, shift={self.shift_degrees}°")
        print(f"[OusterLSS] 深度bins: D={self.D} "
              f"[{d_cfg[0]:.1f}m, {d_cfg[1]:.1f}m, step={d_cfg[2]:.1f}m]")
        print(f"[OusterLSS] 有效frustum点: {n_valid:,} / "
              f"{self.feat_h * self.feat_w * self.D:,}")

    def _load_beam_angles(self, path):
        if path and os.path.exists(path):
            angles = np.load(path)
            print(f"[OusterLSS] 加载beam_altitude_angles: {path}")
            return angles
        print("[OusterLSS] ⚠ 未找到beam_altitude_angles，使用OS1-128近似默认值")
        print("  请运行: python tools/save_beam_angles.py")
        return np.linspace(22.5, -22.5, 128)

    def _fill_frustum_table(self, bev_x, bev_y, valid):
        """
        预计算每个特征像素(h_f, w_f)在各深度bin处对应的BEV格子索引。
        逆推自 point_to_pix(x, y, z, shift_degrees)。
        """
        beam_angles = self.beam_altitude_angles

        for h_f in range(self.feat_h):
            v_img = min(int((h_f + 0.5) * self.feat_stride), self.img_h - 1)
            phi_deg = float(beam_angles[v_img])
            cos_phi = np.cos(np.radians(phi_deg))
            sin_phi = np.sin(np.radians(phi_deg))

            for w_f in range(self.feat_w):
                u_img = min(int((w_f + 0.5) * self.feat_stride), self.img_w - 1)

                # 逆推 point_to_pix 的水平方向映射
                azimuth_deg = u_img / self.img_w * 360.0
                azimuth_deg = (azimuth_deg - self.shift_degrees + 360.0) % 360.0
                theta = np.radians(-azimuth_deg)   # = atan2(y, x)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                for d_idx in range(self.D):
                    d = self.depth_bins[d_idx].item()

                    x_3d = d * cos_phi * cos_theta   # 前向
                    y_3d = d * cos_phi * sin_theta   # 左向

                    bx = int((x_3d - self.bev_xmin) / self.bev_res)
                    by = int((y_3d - self.bev_ymin) / self.bev_res)

                    if (0 <= bx < self.bev_w) and (0 <= by < self.bev_h):
                        bev_x[h_f, w_f, d_idx] = bx
                        bev_y[h_f, w_f, d_idx] = by
                        valid[h_f, w_f, d_idx] = True

    def forward(self, img_feat):
        """
        Args:
            img_feat: (B, 128, 16, 128)  YOLOv8-S P3特征，透视空间

        Returns:
            bev_feat: (B, 80, 126, 128)  BEV空间特征
        """
        B = img_feat.shape[0]
        device = img_feat.device
        out_C = self.out_channels

        # Lift: 预测深度分布 + 特征压缩
        depth_prob = self.depth_head(img_feat).softmax(dim=1)    # (B, D, H, W)
        feat_c     = self.feat_compress(img_feat)                 # (B, out_C, H, W)

        # 外积得到各深度bin的加权特征 (B, out_C, H, W, D)
        # feat_3d = feat_c.unsqueeze(-1) * depth_prob.unsqueeze(1)
        # feat_c.unsqueeze(-1):  (B, 80, 16, 128, 1)
        # depth_prob.unsqueeze(1): (B,  1, 52,  16, 128)  ← 维度完全对不上
        feat_3d = feat_c.unsqueeze(-1) * depth_prob.permute(0, 2, 3, 1).unsqueeze(1)
        # feat_c.unsqueeze(-1):                    (B, 80, 16, 128,  1)
        # depth_prob.permute(0,2,3,1).unsqueeze(1):(B,  1, 16, 128, 52)
        # 结果 feat_3d:                            (B, 80, 16, 128, 52) ✓

        # Splat: scatter到BEV
        bev_feat = torch.zeros(B, out_C, self.bev_h, self.bev_w,
                               device=device, dtype=feat_3d.dtype)
        count    = torch.zeros(B, 1, self.bev_h, self.bev_w,
                               device=device, dtype=feat_3d.dtype)

        h_idx, w_idx, d_idx = self.frustum_valid.nonzero(as_tuple=True)

        if len(h_idx) > 0:
            bx = self.frustum_bev_x[h_idx, w_idx, d_idx]   # [N]
            by = self.frustum_bev_y[h_idx, w_idx, d_idx]   # [N]
            flat_idx = by * self.bev_w + bx                 # [N]

            feat_pts  = feat_3d[:, :, h_idx, w_idx, d_idx] # (B, out_C, N)

            bev_flat   = bev_feat.view(B, out_C, -1)
            count_flat = count.view(B, 1, -1)

            bev_flat.scatter_add_(
                2, flat_idx[None, None].expand(B, out_C, -1), feat_pts)
            count_flat.scatter_add_(
                2, flat_idx[None, None].expand(B, 1, -1),
                torch.ones(B, 1, len(h_idx), device=device, dtype=feat_3d.dtype))

            bev_feat = bev_flat.view(B, out_C, self.bev_h, self.bev_w)
            count    = count_flat.view(B, 1, self.bev_h, self.bev_w).clamp(min=1.0)
            bev_feat = bev_feat / count

        return bev_feat   # (B, 80, 126, 128)

class OusterLSSTransformDual(nn.Module):
    """
    双子集版OusterLSSTransform，支持Bus（shift=180°）和Boston（shift=90°）
    混合batch时逐样本切换对应的LSS变换和beam角度。

    使用方式：在bevfusion_palin.py中用此类替换OusterLSSTransform。
    需要batch_dict中包含 'dataset_flags' (B,) 整数tensor，0=Bus，1=Boston。
    """

    def __init__(self, model_cfg):
        super().__init__()

        # ── 构建Bus版本（shift=180°）──────────────────────────────────────────
        import copy
        cfg_bus = copy.deepcopy(model_cfg)
        cfg_bus.SHIFT_DEGREES = 180.0
        cfg_bus.BEAM_ALTITUDE_ANGLES_PATH = model_cfg.BEAM_ALTITUDE_ANGLES_PATH_BUS
        self.transform_bus = OusterLSSTransform(cfg_bus)

        # ── 构建Boston版本（shift=90°）────────────────────────────────────────
        cfg_boston = copy.deepcopy(model_cfg)
        cfg_boston.SHIFT_DEGREES = 90.0
        cfg_boston.BEAM_ALTITUDE_ANGLES_PATH = model_cfg.BEAM_ALTITUDE_ANGLES_PATH_BOSTON
        self.transform_boston = OusterLSSTransform(cfg_boston)

        print("[OusterLSSTransformDual] 初始化完成：Bus(shift=180°) + Boston(shift=90°)")

    def forward(self, img_feat, dataset_flags=None):
        """
        Args:
            img_feat:      (B, 128, 16, 128)  YOLOv8-S P3特征
            dataset_flags: (B,) int tensor，0=Bus，1=Boston
                           若为None则全部按Bus处理

        Returns:
            bev_feat: (B, 80, 126, 128)
        """
        if dataset_flags is None:
            return self.transform_bus(img_feat)

        B = img_feat.shape[0]

        # 判断batch是否全为同一子集（快速路径）
        flags = dataset_flags.tolist()
        if all(f == 0 for f in flags):
            return self.transform_bus(img_feat)
        if all(f == 1 for f in flags):
            return self.transform_boston(img_feat)

        # 混合batch：逐样本处理后拼接
        results = []
        for i in range(B):
            feat_i = img_feat[i:i+1]  # (1, C, H, W)
            if flags[i] == 0:
                bev_i = self.transform_bus(feat_i)
            else:
                bev_i = self.transform_boston(feat_i)
            results.append(bev_i)

        return torch.cat(results, dim=0)  # (B, 80, 126, 128)