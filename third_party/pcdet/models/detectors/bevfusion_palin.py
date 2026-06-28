"""
BevFusionPALIN - BevFusion在PALIN数据集上的适配版（论文中记为 BevFusion†）

LiDAR分支与原版BevFusion完全相同：
  MeanVFE → VoxelResBackBone8x → HeightCompression → LiDAR BEV (256ch)

Camera分支用Ouster球面投影替换针孔相机LSS：
  YOLOv8-S → OusterLSSTransform → Camera BEV (80ch)

融合与原版BevFusion相同：
  ConvFuser([256+80] → 256ch) → BaseBEVBackbone → AnchorHeadSingle

放置位置: pcdet/models/detectors/bevfusion_palin.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .detector3d_template import Detector3DTemplate

sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
from fusion.wrappers.yolo_extractor import YOLOv8FeatureExtractor
from fusion.ouster_view_transform import OusterLSSTransformDual


class ConvFuserBEV(nn.Module):
    """
    BEV空间ConvFuser，与BevFusion原版ConvFuser完全一致。
    输入: LiDAR BEV (256ch) + Camera BEV (80ch) → 拼接后卷积 → 256ch
    """

    def __init__(self, lidar_ch, camera_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(lidar_ch + camera_ch, out_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_bev, camera_bev):
        # 尺寸对齐（camera BEV插值到LiDAR BEV尺寸）
        if camera_bev.shape[2:] != lidar_bev.shape[2:]:
            camera_bev = F.interpolate(
                camera_bev, size=lidar_bev.shape[2:],
                mode='bilinear', align_corners=False
            )
        return self.fuse(torch.cat([lidar_bev, camera_bev], dim=1))


class BevFusionPALIN(Detector3DTemplate):
    """
    PALIN-BevFusion检测器。

    build_networks()自动构建LiDAR分支标准模块：
      VFE(MeanVFE) → BACKBONE_3D(VoxelResBackBone8x) →
      MAP_TO_BEV(HeightCompression) → BACKBONE_2D(BaseBEVBackbone) →
      DENSE_HEAD(AnchorHeadSingle)

    Camera分支在__init__中手动构建，在forward中插入HeightCompression之后。
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        # 构建LiDAR标准流程（VFE / BACKBONE_3D / MAP_TO_BEV / BACKBONE_2D / DENSE_HEAD）
        self.module_list = self.build_networks()

        # 定位 HeightCompression 在 module_list 中的位置
        self.height_compression_idx = None
        for i, m in enumerate(self.module_list):
            if type(m).__name__ == 'HeightCompression':
                self.height_compression_idx = i
                break
        assert self.height_compression_idx is not None, \
            "未在module_list中找到HeightCompression，请检查YAML中MAP_TO_BEV配置"

        # 构建Camera分支
        self._init_camera_branch()

    def _init_camera_branch(self):
        """初始化Camera分支：YOLOv8 + OusterLSS + ConvFuserBEV"""

        if not self.model_cfg.get('ENABLE_MULTIMODAL_FUSION', False):
            self.use_multimodal_fusion = False
            print("[BevFusionPALIN] 单模态模式（仅LiDAR）")
            return

        print("[BevFusionPALIN] 初始化Camera分支...")

        # 1. YOLOv8-S特征提取器
        yolo_cfg = self.model_cfg.YOLO_CONFIG
        self.yolo_extractor = YOLOv8FeatureExtractor(
            model_path_or_config=yolo_cfg.MODEL_PATH,
            pretrained_weights=getattr(yolo_cfg, 'PRETRAINED_WEIGHTS', None),
            freeze_weights=getattr(yolo_cfg, 'FREEZE_WEIGHTS', False),
            device='cuda'
        )

        # 2. Ouster球面投影LSS变换（替换原版DepthLSSTransform）
        self.lss_transform = OusterLSSTransformDual(self.model_cfg.LSS_CONFIG)

        # 3. ConvFuserBEV（与原版BevFusion ConvFuser结构相同）
        fuser_cfg = self.model_cfg.FUSER_CONFIG
        self.conv_fuser = ConvFuserBEV(
            lidar_ch=fuser_cfg.LIDAR_CHANNELS,    # 256 (HeightCompression输出)
            camera_ch=fuser_cfg.CAMERA_CHANNELS,  # 80  (OusterLSS输出)
            out_ch=fuser_cfg.OUT_CHANNELS          # 256 (BaseBEVBackbone输入)
        )

        self.use_multimodal_fusion = True

        # 参数量统计
        n_yolo = sum(p.numel() for p in self.yolo_extractor.parameters())
        n_lss = sum(p.numel() for p in self.lss_transform.parameters())
        n_fuser = sum(p.numel() for p in self.conv_fuser.parameters())
        print(f"  YOLOv8-S:         {n_yolo:>12,} params")
        print(f"  OusterLSS:        {n_lss:>12,} params")
        print(f"  ConvFuserBEV:     {n_fuser:>12,} params")
        print("[BevFusionPALIN] 初始化完成")

    def forward(self, batch_dict):
        """
        执行顺序:
          VFE → VoxelResBackBone8x → HeightCompression
          → [Camera分支: YOLOv8 → OusterLSS → ConvFuserBEV 融合]
          → BaseBEVBackbone → AnchorHeadSingle
        """
        for i, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)

            # HeightCompression执行完后立即融合
            if i == self.height_compression_idx:
                if self.use_multimodal_fusion and 'images' in batch_dict:
                    batch_dict = self._apply_camera_fusion(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            return {'loss': loss}, tb_dict, disp_dict
        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss()
        #     return {'loss': loss}, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def _apply_camera_fusion(self, batch_dict):
        """
        Camera分支前向 + ConvFuserBEV融合。

        batch_dict['spatial_features']:  HeightCompression输出, (B, 256, H_bev, W_bev)
        batch_dict['images']:            Near-IR图像, (B, 3, 128, 1024)

        融合后更新 batch_dict['spatial_features'] 为 (B, 256, H_bev, W_bev)
        以便 BaseBEVBackbone 正常读取。
        """
        try:
            images = batch_dict['images']
            device = next(self.parameters()).device
            if images.device != device:
                images = images.to(device)

            # --- Step 1: YOLOv8提取P3/P4/P5特征 ---
            if self.training:
                self.yolo_extractor.train()
            else:
                self.yolo_extractor.eval()
            img_feats = self.yolo_extractor.extract_multiscale_features(images)
            # P3: (B, 128, 16, 128)  P4: (B, 256, 8, 64)  P5: (B, 512, 4, 32)

            # --- Step 2: OusterLSS — 用P3做视图变换到BEV ---
            p3_feat = img_feats[0]                     # (B, 128, 16, 128)
            # camera_bev = self.lss_transform(p3_feat)   # (B, 80, H_bev, W_bev)
            # 改为：
            dataset_flags = batch_dict.get('dataset_flags', None)
            camera_bev = self.lss_transform(p3_feat, dataset_flags)
            # --- Step 3: ConvFuserBEV ---
            # HeightCompression将结果存入 spatial_features
            lidar_bev = batch_dict['spatial_features']  # (B, 256, H_bev, W_bev)
            fused = self.conv_fuser(lidar_bev, camera_bev)  # (B, 256, H_bev, W_bev)

            # 更新spatial_features供BaseBEVBackbone使用
            batch_dict['spatial_features'] = fused

            print(f"[BevFusionPALIN] LiDAR{tuple(lidar_bev.shape)} "
                  f"+ CamBEV{tuple(camera_bev.shape)} → {tuple(fused.shape)}")

        except Exception as e:
            print(f"[BevFusionPALIN] Camera融合失败，回退到纯LiDAR: {e}")
            import traceback
            traceback.print_exc()

        return batch_dict

    # def get_training_loss(self):
    #     disp_dict = {}
    #     loss_rpn, tb_dict = self.dense_head.get_loss()
    #     tb_dict['loss_rpn'] = loss_rpn.item()
    #     return loss_rpn, tb_dict, disp_dict

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss = batch_dict['loss']
        tb_dict = batch_dict['tb_dict']
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        # TransFusionHead直接输出final_box_dicts，不走detector3d_template的post_processing
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']
            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict,
                batch_index=index,
                data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
        return final_pred_dict, recall_dict
    # def post_processing(self, batch_dict):
    #     # ── AnchorHeadSingle 路径：走父类标准 post_processing ──────────
    #     if 'final_box_dicts' not in batch_dict:
    #         print("[DEBUG] routing to super().post_processing")
    #         return super().post_processing(batch_dict)
    #
    #     # ── TransFusionHead 路径：final_box_dicts 已由头部直接写入 ──────
    #     post_process_cfg = self.model_cfg.POST_PROCESSING
    #     batch_size = batch_dict['batch_size']
    #     final_pred_dict = batch_dict['final_box_dicts']
    #     recall_dict = {}
    #     for index in range(batch_size):
    #         pred_boxes = final_pred_dict[index]['pred_boxes']
    #         recall_dict = self.generate_recall_record(
    #             box_preds=pred_boxes,
    #             recall_dict=recall_dict,
    #             batch_index=index,
    #             data_dict=batch_dict,
    #             thresh_list=post_process_cfg.RECALL_THRESH_LIST
    #         )
    #     return final_pred_dict, recall_dict