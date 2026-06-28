"""
TransFusion + Ouster Near-IR Camera Fusion 检测器
在原有 TransFusion (LiDAR-only) 基础上增加图像分支

位置: pcdet/models/detectors/transfusion_camera.py

主要改动：
1. 在 forward 中读取 batch_dict['images'] 和 shift_degrees
2. 用 YOLOv8 提取图像特征
3. 将图像特征和 shift_degrees 传入 TransFusionHead
"""

import torch
import torch.nn as nn
import sys
import os

from .detector3d_template import Detector3DTemplate
from ..utils.ouster_projection import get_shift_degrees_from_frame_id


class TransFusionCamera(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # ── 图像backbone（复用你的YOLOv8 extractor）──────────────────
        yolo_cfg = model_cfg.get('YOLO_CONFIG', None)
        if yolo_cfg is not None:
            sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
            from fusion.wrappers.yolo_extractor import YOLOv8FeatureExtractor
            self.img_backbone = YOLOv8FeatureExtractor(
                model_path_or_config=yolo_cfg.get('MODEL_PATH', None),
                pretrained_weights=yolo_cfg.get('PRETRAINED_WEIGHTS', None),
                freeze_weights=yolo_cfg.get('FREEZE_WEIGHTS', False),
                device='cuda',
                extract_layers=[15, 18, 21]   # P3, P4, P5
            )
        else:
            self.img_backbone = None

    def forward(self, batch_dict):
        # ── 1. 提取图像特征 ──────────────────────────────────────────
        if self.img_backbone is not None and 'images' in batch_dict:
            images = batch_dict['images']   # [B, 3, H, W]
            # img_features = self.img_backbone(images)
            images = batch_dict['images'].cuda()  # 或 .to(self.device)
            img_features = self.img_backbone(images)
            # img_features: [P3, P4, P5]，取 P4（stride=16）作为融合特征
            # P4 shape: [B, 128, H/16, W/16] = [B, 128, 8, 64]（针对128x1024输入）
            batch_dict['img_feat'] = img_features[1]   # P4

        # ── 2. 提取每个样本的 shift_degrees ─────────────────────────
        # frame_ids 在 batch_dict['frame_id'] 里，是一个 list[str]
        frame_ids = batch_dict.get('frame_id', [])
        shift_degrees_list = [
            get_shift_degrees_from_frame_id(fid) for fid in frame_ids
        ]
        batch_dict['shift_degrees'] = shift_degrees_list

        # ── 3. 正常走 LiDAR pipeline ─────────────────────────────────
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            return {'loss': loss}, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
        tb_dict = {'loss_trans': loss_trans.item(), **tb_dict}
        return loss_trans, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']
            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
        return final_pred_dict, recall_dict
