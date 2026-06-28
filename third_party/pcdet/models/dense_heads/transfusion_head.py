import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils
from .transfusion_img_decoder import OusterTransFusionImgDecoder
import os

class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class TransFusionHead(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(TransFusionHead, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class
        print("voxel_size type and value:", type(self.voxel_size), self.voxel_size)  # ← 加这行

        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')

        hidden_channel=self.model_cfg.HIDDEN_CHANNEL
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.bn_momentum = self.model_cfg.BN_MOMENTUM
        self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']

        self.code_size = 8

        self.fuse_img = self.model_cfg.get('FUSE_IMG', False)
        self.num_views = self.model_cfg.get('NUM_VIEWS', 1)

        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=num_class,kernel_size=3,padding=1))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)
        if self.fuse_img:
            metadata_path = model_cfg.get('METADATA_PATH', None)
            beam_altitude_angles = None
            if metadata_path and os.path.exists(metadata_path):
                try:
                    from ouster.sdk import client
                    with open(metadata_path, 'r') as f:
                        metadata = client.SensorInfo(f.read())
                    beam_altitude_angles = list(metadata.beam_altitude_angles)
                    print(f"TransFusionHead: beam_altitude_angles 加载成功")
                except Exception as e:
                    print(f"TransFusionHead: metadata 加载失败: {e}")

            img_feat_channel = self.model_cfg.get('IN_CHANNELS_IMG', 256)
            feat_stride = self.model_cfg.get('OUT_SIZE_FACTOR_IMG', 16)
            self.img_decoder = OusterTransFusionImgDecoder(
                hidden_channel=hidden_channel,
                num_heads=num_heads,
                # img_feat_channel=128,
                img_feat_channel=img_feat_channel,
                num_classes=self.num_classes,
                num_proposals=self.num_proposals,
                ffn_channel=ffn_channel,
                dropout=dropout,
                img_h=128, img_w=1024,
                feat_stride=16,
                out_size_factor=self.feature_map_stride,
                voxel_size=voxel_size[0],
                pc_range=list(point_cloud_range),
                beam_altitude_angles=beam_altitude_angles,
            )

            # 图像融合预测头，输入通道是 hidden_channel * 2
            heads_img = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
            heads_img['heatmap'] = dict(
                out_channels=self.num_classes,
                num_conv=self.model_cfg.NUM_HM_CONV
            )
            self.img_prediction_head = SeparateHead_Transfusion(
                hidden_channel * 2, 64, 1, heads_img, use_bias=bias
            )
        self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.forward_ret_dict = {}

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def predict(self, inputs):
        batch_size = inputs.shape[0]  # inputs shape: torch.Size([2, 512, 126, 128])
        # print(inputs, f"inputs shape: {inputs.shape}")
        lidar_feat = self.shared_conv(inputs)  # lidar_feat shape: torch.Size([2, 128, 126, 128])
        # print(f"lidar_feat shape: {lidar_feat.shape}")

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # bev_pos shape: torch.Size([2, 16128, 2])   128, 126
        # print(bev_pos, f"bev_pos shape: {bev_pos.shape}")

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)  #  dense_heapmap shape: torch.Size([2, 11, 126, 128])
        # print(dense_heatmap, f"dense_heapmap shape: {dense_heatmap.shape}")
        heatmap = dense_heatmap.detach().sigmoid()  #  heapmap shape: torch.Size([2, 11, 126, 128])
        # print(heatmap, f"heapmap shape: {heatmap.shape}")
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.dataset_name == "nuScenes":
            local_max[ :, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[ :, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        # for Pedestrian & Cyclist in Waymo
        elif self.dataset_name == "Waymo":
            local_max[ :, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[ :, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        elif self.dataset_name == "custom":
            # Pedestrian=1, Cyclist=7, Scooter=8, Motorcycle=10
            for cls_idx in [1, 7, 8, 10]:
                local_max[:, cls_idx] = F.max_pool2d(
                    heatmap[:, cls_idx], kernel_size=1, stride=1, padding=0
                )
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
 
        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
            ..., : self.num_proposals
        ]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        # convert to yx
        query_pos = query_pos.flip(dims=[-1])
        bev_pos = bev_pos.flip(dims=[-1])
        # print(query_pos,bev_pos,f"query_pos shape: {query_pos.shape}")    # query_pos shape: torch.Size([2, 200, 2])

        query_feat = self.decoder(
            query_feat, lidar_feat_flatten, query_pos, bev_pos
        )
        # print(query_feat, f"query_feat shape: {query_feat.shape}")  #  query_feat shape: torch.Size([2, 128, 200])
        res_layer = self.prediction_head(query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)

        if self.fuse_img and 'img_feat' in self.forward_ret_dict:
            img_feat = self.forward_ret_dict['img_feat']
            shift_degrees = self.forward_ret_dict['shift_degrees']

            # 恢复 query 的米制 3D 坐标
            center_metric = query_pos.permute(0, 2, 1).clone()
            center_metric[:, 0, :] = (center_metric[:, 0, :] *
                                      self.feature_map_stride *
                                      self.voxel_size[1] +
                                      self.point_cloud_range[1])
            center_metric[:, 1, :] = (center_metric[:, 1, :] *
                                      self.feature_map_stride *
                                      self.voxel_size[0] +
                                      self.point_cloud_range[0])
            # center_metric[:, 0, :] = (center_metric[:, 0, :] *
            #                           self.feature_map_stride *
            #                           self.voxel_size[0] +
            #                           self.point_cloud_range[0])
            # center_metric[:, 1, :] = (center_metric[:, 1, :] *
            #                           self.feature_map_stride *
            #                           self.voxel_size[1] +
            #                           self.point_cloud_range[1])
            query_pos_3d = torch.cat(
                [center_metric, res_layer['height']], dim=1
            )  # [B, 3, N_q]

            query_feat_fused, dense_heatmap_img, on_mask = self.img_decoder(
                lidar_feat_flatten=lidar_feat_flatten,
                bev_pos=bev_pos,
                query_feat=query_feat,
                query_pos=query_pos,
                query_pos_3d=query_pos_3d,
                pred_boxes=None,
                img_feat=img_feat,
                img_heatmap_lidar=dense_heatmap,
                shift_degrees_list=shift_degrees,
            )

            res_layer_img = self.img_prediction_head(query_feat_fused)
            res_layer_img["center"] = (res_layer_img["center"] +
                                       query_pos.permute(0, 2, 1))

            # 没有投影到图像的 query 保持 LiDAR 预测结果
            for key in res_layer_img:
                if key in res_layer:
                    mask = ~on_mask.unsqueeze(1).expand_as(res_layer_img[key])
                    res_layer_img[key][mask] = res_layer[key][mask]

            res_layer = res_layer_img
            res_layer["dense_heatmap"] = dense_heatmap_img
        else:
            res_layer["dense_heatmap"] = dense_heatmap

        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )
        return res_layer

    def forward(self, batch_dict):
        feats = batch_dict['spatial_features_2d']
        # 新增：把图像特征存入 forward_ret_dict
        if self.fuse_img:
            self.forward_ret_dict['img_feat'] = batch_dict.get('img_feat', None)
            self.forward_ret_dict['shift_degrees'] = batch_dict.get('shift_degrees', [])

        res = self.predict(feats)
        if not self.training:
            bboxes = self.get_bboxes(res)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d = gt_boxes[...,-1].long() - 1
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        
        num_proposals = preds_dict["center"].shape[-1]
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]["pred_boxes"]
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)

        # if gt_bboxes_3d.shape[0] > 0 and not hasattr(self, '_printed_debug'):
        #     self._printed_debug = True
        #     print("=== GT boxes (前3个) ===")
        #     print(gt_bboxes_3d[:3])
        #     print("=== Pred boxes (前3个) ===")
        #     print(bboxes_tensor[:3])
        #     print(f"GT x: {gt_bboxes_3d[:, 0].min():.2f}~{gt_bboxes_3d[:, 0].max():.2f}")
        #     print(f"GT y: {gt_bboxes_3d[:, 1].min():.2f}~{gt_bboxes_3d[:, 1].max():.2f}")
        #     print(f"GT z: {gt_bboxes_3d[:, 2].min():.2f}~{gt_bboxes_3d[:, 2].max():.2f}")
        #     print(f"GT l: {gt_bboxes_3d[:, 3].min():.2f}~{gt_bboxes_3d[:, 3].max():.2f}")
        #     print(f"Pred x: {bboxes_tensor[:, 0].min():.2f}~{bboxes_tensor[:, 0].max():.2f}")
        #     print(f"Pred y: {bboxes_tensor[:, 1].min():.2f}~{bboxes_tensor[:, 1].max():.2f}")
        #     print(f"Pred z: {bboxes_tensor[:, 2].min():.2f}~{bboxes_tensor[:, 2].max():.2f}")
        #     print(f"Pred l: {bboxes_tensor[:, 3].min():.2f}~{bboxes_tensor[:, 3].max():.2f}")

        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
            score, self.point_cloud_range,
        )
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()

        # ← 加这行
        print(f"num_pos: {len(pos_inds)}, num_gt: {gt_bboxes_3d.shape[0]}, "
              f"max_iou: {ious.max().item():.4f}, mean_pos_iou: "
              f"{ious[pos_inds].mean().item() if len(pos_inds) > 0 else 0:.4f}")

        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        else:
            # print('7')
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.code_size]).to(center.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # compute dense heatmap targets
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = (self.grid_size[:2] // self.feature_map_stride) 
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])

        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride
                # print('x, y: ',x, y, "coor_x, coor_y: ", coor_x, coor_y)

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)

        # 在get_targets_single里加打印
        print(f"feature_map_size: {feature_map_size}")
        print(f"heatmap shape: {heatmap.shape}")
        print(f"heatmap max: {heatmap.max()}")
        print(f"heatmap sum: {heatmap.sum()}")
        print(f"heatmap nonzero cells: {(heatmap > 0).sum()}")
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None])

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):
        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        ).sum() / max(heatmap.eq(1).float().sum().item(), 1)

        print("heatmap loss:", loss_heatmap.item())  # ← 就是这句

        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all

        return loss_all,loss_dict

    def encode_bbox(self, bboxes):
        code_size = 8
        targets = torch.zeros([bboxes.shape[0], code_size]).to(bboxes.device)
        # targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        # targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 0] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 1] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        if code_size == 10 and bboxes.shape[1] > 7:   # ← 加这个判断
            targets[:, 8:10] = bboxes[:, 7:]
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).cuda().float()
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        # center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        # 改后：decode后swap回[x, y]顺序
        center_y = center[:, 0, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        center_x = center[:, 1, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 0, :] = center_x  # position 0 = x
        center[:, 1, :] = center_y  # position 1 = y
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)
        # # 在 decode_bbox 里，final_box_preds 计算完之后加：
        # print("pred center range x:", final_box_preds[0, :, 0].min().item(), final_box_preds[0, :, 0].max().item())
        # print("pred center range y:", final_box_preds[0, :, 1].min().item(), final_box_preds[0, :, 1].max().item())
        # print("pred dim range:", final_box_preds[0, :, 3:6].min().item(), final_box_preds[0, :, 3:6].max().item())
        # print("=== decode_bbox debug ===")
        # print("pred x range:", final_box_preds[0, :, 0].min().item(), final_box_preds[0, :, 0].max().item())
        # print("pred y range:", final_box_preds[0, :, 1].min().item(), final_box_preds[0, :, 1].max().item())
        # print("pred z range:", final_box_preds[0, :, 2].min().item(), final_box_preds[0, :, 2].max().item())
        # print("pred l range:", final_box_preds[0, :, 3].min().item(), final_box_preds[0, :, 3].max().item())
        # print("pred w range:", final_box_preds[0, :, 4].min().item(), final_box_preds[0, :, 4].max().item())
        # print("pred h range:", final_box_preds[0, :, 5].min().item(), final_box_preds[0, :, 5].max().item())
        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh        
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        # ← 加这行
        print("boxes passing score filter:", thresh_mask[0].sum().item(),
              "boxes passing range filter:", mask[0].sum().item())

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
            }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def get_bboxes(self, preds_dicts):

        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid()
        one_hot = F.one_hot(
            self.query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
        batch_center = preds_dicts["center"]
        batch_height = preds_dicts["height"]
        batch_dim = preds_dicts["dim"]
        batch_rot = preds_dicts["rot"]
        batch_vel = None
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]

        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim,
            batch_center, batch_height, batch_vel,
            filter=True,
        )
        # ── 打印 score 分布 ──────────────────────
        print("max score:", batch_score[0].max().item())
        print("scores > 0.1:", (batch_score[0].max(0).values > 0.1).sum().item())
        print("scores > 0.3:", (batch_score[0].max(0).values > 0.3).sum().item())
        print("score distribution:", batch_score[0].max(0).values.sort(descending=True).values[:10])
        # ── 临时 debug，只打印第一个样本 ──
        if ret_dict[0]['pred_boxes'].shape[0] > 0:
            print("=== pred boxes sample 0 (前5个) ===")
            print(ret_dict[0]['pred_boxes'][:5])
            print("=== pred scores ===")
            print(ret_dict[0]['pred_scores'][:5])
        # ────────────────────────────────
        for k in range(batch_size):
            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1

        return ret_dict 
