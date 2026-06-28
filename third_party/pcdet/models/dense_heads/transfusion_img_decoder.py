"""
TransFusion 图像融合解码器 - 修正版
基于官方 MIT BEVFusion transfusion_head.py 实现，适配 Ouster OS1-128 球坐标投影

主要修正（对比第一版）：
1. 高斯 sigma 动态计算（从 box 角点投影大小推算）
2. 加入 PositionEmbeddingLearned（query 和 key 都有位置编码）
3. 加入 image-guided query initialization（图像热力图辅助初始化）
4. 加入 on_the_image_mask（只处理投影到图像范围内的 query）
5. 最终预测头输入改为 concat[query_feat, prev_query_feat]（2倍通道）
6. 加入 Image→BEV collapsed 步骤
7. 不做数据增强，无需逆变换

位置: pcdet/models/dense_heads/transfusion_img_decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from pcdet.models.utils.ouster_projection import OusterProjection, get_shift_degrees_from_frame_id


# ─────────────────────────────────────────────────────────────
# 1. 可学习位置编码（与官方完全一致）
# ─────────────────────────────────────────────────────────────
class PositionEmbeddingLearned(nn.Module):
    """
    可学习绝对位置编码，对应官方代码第25-41行
    输入: [B, N, input_channel] 的坐标
    输出: [B, num_pos_feats, N] 的位置编码
    """
    def __init__(self, input_channel, num_pos_feats=128):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1),
        )

    def forward(self, xyz):
        # xyz: [B, N, C] → transpose → [B, C, N]
        xyz = xyz.transpose(1, 2).contiguous()
        return self.position_embedding_head(xyz)  # [B, num_pos_feats, N]


# ─────────────────────────────────────────────────────────────
# 2. Transformer Decoder Layer（与官方完全一致）
# ─────────────────────────────────────────────────────────────
class TransformerDecoderLayer(nn.Module):
    """
    对应官方代码第44-122行
    支持 cross_only 模式（只做 cross-attention，跳过 self-attention）
    """
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1,
                 activation='relu', self_posembed=None, cross_posembed=None,
                 cross_only=False):
        super().__init__()
        self.cross_only = cross_only

        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = F.relu if activation == 'relu' else F.gelu

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        query:     [B, C, N_q]  (channel-first，与官方一致)
        key:       [B, C, N_k]
        query_pos: [B, N_q, 2]  (x,y坐标)
        key_pos:   [B, N_k, 2]
        """
        # 位置编码
        query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1) \
            if self.self_posembed is not None else None
        key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1) \
            if self.cross_posembed is not None else None

        # 转为 [N, B, C]（PyTorch MHA 默认格式）
        query = query.permute(2, 0, 1)
        key   = key.permute(2, 0, 1)

        # Self-attention（可选）
        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        # Cross-attention
        query2 = self.multihead_attn(
            query=self.with_pos_embed(query, query_pos_embed),
            key=self.with_pos_embed(key,   key_pos_embed),
            value=self.with_pos_embed(key,  key_pos_embed),
            attn_mask=attn_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # FFN
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # 转回 [B, C, N]
        return query.permute(1, 2, 0)


# ─────────────────────────────────────────────────────────────
# 3. 主模块：OusterTransFusionImgDecoder
# ─────────────────────────────────────────────────────────────
class OusterTransFusionImgDecoder(nn.Module):
    """
    TransFusion 图像融合完整实现，适配 Ouster OS1-128

    包含：
      A. Image→BEV collapsed cross-attention（图像特征 collapse 后投影到 BEV）
      B. Image-guided query initialization（图像热力图辅助 top-k 初始化）
      C. Camera cross-attention decoder（第二层 decoder，query 查询图像特征）
      D. on_the_image_mask（只更新投影在图像内的 query）
      E. 最终 concat[query_feat, prev_query_feat] 输出

    不做数据增强，不需要 apply_3d_transformation 逆变换。
    """

    def __init__(self,
                 hidden_channel: int,          # query 特征维度，通常 128
                 num_heads: int,               # attention 头数，通常 8
                 img_feat_channel: int,        # YOLOv8 P4 输出通道，128
                 num_classes: int,             # 11
                 num_proposals: int,           # top-k query 数量，通常 200
                 ffn_channel: int = 256,
                 dropout: float = 0.1,
                 img_h: int = 128,
                 img_w: int = 1024,
                 feat_stride: int = 16,        # YOLOv8 P4 stride
                 out_size_factor: int = 8,     # LiDAR BEV 相对 voxel grid 的 stride
                 voxel_size: float = 0.1,      # m/voxel
                 pc_range: list = None,        # [x_min, y_min, ...]
                 beam_altitude_angles: list = None,
                 ):
        super().__init__()

        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.img_h = img_h
        self.img_w = img_w
        self.feat_h = img_h // feat_stride    # 128/16 = 8
        self.feat_w = img_w // feat_stride    # 1024/16 = 64
        self.feat_stride = feat_stride
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.pc_range = pc_range or [-51.2, -60.0, -3.0, 51.2, 40.8, 5.0]

        # ── A. 图像特征投影层 ─────────────────────────────────
        # 对应官方 shared_conv_img（把图像特征 channel 统一到 hidden_channel）
        self.shared_conv_img = nn.Sequential(
            nn.Conv2d(img_feat_channel, hidden_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
        )

        # 图像特征 collapse 后的线性层（对应官方 self.fc）
        self.fc = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)

        # ── B. Image-guided query initialization ─────────────
        # 图像 BEV 热力图头（对应官方 heatmap_head_img）
        self.heatmap_head_img = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, num_classes, kernel_size=3, padding=1),
        )

        # ── C. Transformer Decoder Layers ─────────────────────
        # Layer 0: Image→BEV collapsed cross-attention（cross_only）
        self.img_to_bev_decoder = TransformerDecoderLayer(
            hidden_channel, num_heads, ffn_channel, dropout,
            self_posembed=PositionEmbeddingLearned(2, hidden_channel),
            cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            cross_only=True,   # 只做 cross-attention，跳过 self-attention
        )

        # Layer 1: Camera cross-attention（query 查询图像特征）
        self.img_cross_decoder = TransformerDecoderLayer(
            hidden_channel, num_heads, ffn_channel, dropout,
            self_posembed=PositionEmbeddingLearned(2, hidden_channel),
            cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            cross_only=False,
        )

        # ── Ouster 投影器 ──────────────────────────────────────
        assert beam_altitude_angles is not None
        self.ouster_proj = OusterProjection(
            beam_altitude_angles=beam_altitude_angles,
            img_width=img_w,
            img_height=img_h,
        )

        # 缓存位置编码（避免重复创建）
        self._img_feat_pos = None           # [1, feat_h*feat_w, 2]
        self._img_collapsed_pos = None      # [1, feat_w, 2]

    # ──────────────────────────────────────────────────────────
    # 工具函数
    # ──────────────────────────────────────────────────────────
    def _create_2d_grid(self, h, w, device):
        """创建 [1, h*w, 2] 的归一化网格坐标，对应官方 create_2D_grid"""
        ys = torch.linspace(0, h - 1, h, device=device) + 0.5
        xs = torch.linspace(0, w - 1, w, device=device) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coord = torch.stack([grid_x, grid_y], dim=-1)  # [h, w, 2]
        return coord.view(1, h * w, 2)                  # [1, h*w, 2]

    def _project_queries_to_image(self, query_pos_3d, shift_degrees_list):
        """
        将 query 3D 中心点投影到图像坐标（Ouster 球坐标投影）
        不需要数据增强逆变换（训练时不做增强）

        Args:
            query_pos_3d:      [B, 3, N_q]  (x, y, z)
            shift_degrees_list: list[float]

        Returns:
            pix_coords: [B, N_q, 2]  (col_u, row_v)，图像像素坐标
        """
        B, _, N_q = query_pos_3d.shape
        # 转为 [B, N_q, 3]
        pts = query_pos_3d.permute(0, 2, 1)
        return self.ouster_proj.project_query_centers_batch(pts, shift_degrees_list)

    def _project_box_corners_to_image(self, corners_3d, shift_degrees, device):
        """
        投影 box 8个角点到图像，用于动态计算高斯 sigma

        Args:
            corners_3d: [N_q, 8, 3]
            shift_degrees: float

        Returns:
            corners_2d: [N_q, 8, 2]
        """
        N_q = corners_3d.shape[0]
        corners_flat = corners_3d.reshape(-1, 3)  # [N_q*8, 3]
        pix = self.ouster_proj.project_points_batch(corners_flat, shift_degrees)  # [N_q*8, 2]
        return pix.reshape(N_q, 8, 2)

    def _compute_dynamic_gaussian(self, query_centers_2d, box_corners_2d,
                                  on_the_image, feat_pos):
        """
        动态计算每个 query 的高斯掩码，对应官方第988-997行

        Args:
            query_centers_2d: [N_valid, 2]  在特征图坐标系
            box_corners_2d:   [N_valid, 8, 2]  在特征图坐标系
            on_the_image:     bool mask [N_q]
            feat_pos:         [1, feat_h*feat_w, 2]

        Returns:
            attn_mask: [N_valid, feat_h*feat_w]，log-space gaussian mask
        """
        # 从 box 角点计算外接圆半径（对应官方第991-993行）
        corner_range = box_corners_2d.max(1).values - box_corners_2d.min(1).values  # [N_valid, 2]
        radius = torch.ceil(corner_range.norm(dim=-1, p=2) / 2).int()               # [N_valid]
        sigma = (radius.float() * 2 + 1) / 6.0                                      # [N_valid]

        # 计算高斯权重（对应官方第994-997行）
        # centers: [N_valid, 2], feat_pos: [1, H*W, 2]
        center_xy = query_centers_2d.unsqueeze(1)   # [N_valid, 1, 2]
        key_pos   = feat_pos - 0.5                  # [1, H*W, 2]
        distance  = (center_xy - key_pos).norm(dim=-1) ** 2   # [N_valid, H*W]
        gaussian  = (-distance / (2 * sigma[:, None] ** 2)).exp()
        gaussian[gaussian < torch.finfo(torch.float32).eps] = 0

        return gaussian.log().clamp(min=-1e4)  # log-space，对应官方 attn_mask=gaussian_mask.log()

    # ──────────────────────────────────────────────────────────
    # 主 forward
    # ──────────────────────────────────────────────────────────
    def forward(self,
                lidar_feat_flatten,   # [B, C, H_bev*W_bev]  LiDAR BEV 特征（已 flatten）
                bev_pos,              # [B, H_bev*W_bev, 2]  BEV 网格坐标
                query_feat,           # [B, C, N_q]          第一层 decoder 输出的 query
                query_pos,            # [B, N_q, 2]          query 在 BEV 网格的位置
                query_pos_3d,         # [B, 3, N_q]          query 的 3D 坐标 (x,y,z) in meters
                pred_boxes,           # list[dict]            第一层 decoder 预测的 box（含 corners）
                img_feat,             # [B, C_img, feat_h, feat_w]  YOLOv8 P4
                img_heatmap_lidar,    # [B, num_classes, H_bev, W_bev]  LiDAR 热力图（sigmoid 前）
                shift_degrees_list,   # list[float]           Bus=180.0, Boston=90.0
                ):
        """
        Returns:
            query_feat_fused: [B, 2*C, N_q]  融合图像后的 query（concat prev+new）
            dense_heatmap_img: [B, num_classes, H_bev, W_bev]  图像热力图（用于 loss）
            on_the_image_mask: [B, N_q]  bool，query 是否投影到图像内
        """
        B = query_feat.shape[0]
        device = query_feat.device

        # ══════════════════════════════════════════════════════
        # A. 图像特征预处理
        # ══════════════════════════════════════════════════════
        img_feat = self.shared_conv_img(img_feat)  # [B, C, feat_h, feat_w]
        img_h, img_w = img_feat.shape[-2], img_feat.shape[-1]

        # Collapse 高度方向（max pool），得到 [B, C, feat_w]
        img_feat_collapsed = img_feat.max(2).values                # [B, C, feat_w]
        img_feat_collapsed = self.fc(img_feat_collapsed)           # [B, C, feat_w]

        # 图像特征 flatten，用于 camera cross-attention
        img_feat_flatten = img_feat.view(B, self.hidden_channel, -1)  # [B, C, feat_h*feat_w]

        # ── 位置编码缓存 ──────────────────────────────────────
        if self._img_feat_pos is None or self._img_feat_pos.device != device:
            self._img_feat_pos = self._create_2d_grid(img_h, img_w, device)          # [1, H*W, 2]
        if self._img_collapsed_pos is None or self._img_collapsed_pos.device != device:
            self._img_collapsed_pos = self._create_2d_grid(1, img_w, device)         # [1, W, 2]

        img_feat_pos = self._img_feat_pos.expand(B, -1, -1)          # [B, H*W, 2]
        img_collapsed_pos = self._img_collapsed_pos.expand(B, -1, -1) # [B, W, 2]

        # ══════════════════════════════════════════════════════
        # B. Image → BEV collapsed cross-attention
        #    对应官方第832-833行
        #    用 collapsed 图像特征更新 LiDAR BEV 特征
        # ══════════════════════════════════════════════════════
        bev_feat = self.img_to_bev_decoder(
            query=lidar_feat_flatten,          # [B, C, H*W]
            key=img_feat_collapsed,            # [B, C, feat_w]
            query_pos=bev_pos,                 # [B, H*W, 2]
            key_pos=img_collapsed_pos,         # [B, feat_w, 2]
        )  # [B, C, H*W]

        # ══════════════════════════════════════════════════════
        # C. Image-guided query initialization
        #    对应官方第841-843行
        #    图像热力图 + LiDAR 热力图 平均 → 更好的 top-k 初始化
        # ══════════════════════════════════════════════════════
        bev_shape = img_heatmap_lidar.shape[-2:]   # (H_bev, W_bev)
        dense_heatmap_img = self.heatmap_head_img(
            bev_feat.view(B, self.hidden_channel, *bev_shape)
        )  # [B, num_classes, H_bev, W_bev]

        # 返回给外部用于图像热力图 loss 和 query 初始化
        # 外部调用者负责计算 combined_heatmap 并重新选 top-k query
        # （这里只返回 dense_heatmap_img，不在本模块内做 top-k 选取）

        # ══════════════════════════════════════════════════════
        # D. Camera Cross-Attention Decoder
        #    对应官方第902-1010行
        # ══════════════════════════════════════════════════════
        prev_query_feat = query_feat.detach().clone()  # 保存第一层输出，用于最后 concat
        new_query_feat  = torch.zeros_like(query_feat) # 新容器，只填充投影到图像内的 query

        # on_the_image_mask: -1 表示未投影到图像，>= 0 表示投影到图像
        on_the_image_mask = torch.ones(B, self.num_proposals, device=device) * -1

        for sample_idx in range(B):
            shift = shift_degrees_list[sample_idx]

            # ── 1. 投影 query 中心到图像 ──────────────────────
            q_pos_3d = query_pos_3d[sample_idx]   # [3, N_q]
            q_pix = self.ouster_proj.project_points_batch(
                q_pos_3d.permute(1, 0),            # [N_q, 3]
                shift
            )  # [N_q, 2]  (col_u, row_v) 在原图坐标

            # ── 2. 判断哪些 query 在图像范围内 ─────────────────
            on_image = (
                (q_pix[:, 0] >= 0) & (q_pix[:, 0] < self.img_w) &
                (q_pix[:, 1] >= 0) & (q_pix[:, 1] < self.img_h)
            )  # [N_q] bool

            if on_image.sum() <= 1:
                continue  # 当前样本没有 query 投影到图像，跳过

            on_the_image_mask[sample_idx, on_image] = 0  # 标记有效

            # ── 3. 投影 box 角点，计算动态 sigma ────────────────
            # 把 query 中心坐标从 BEV 网格坐标 → 米制坐标
            q_pix_feat = q_pix[on_image] / self.feat_stride  # 特征图坐标 [N_valid, 2]

            # 获取预测 box 的角点（如果有的话）
            if pred_boxes is not None and 'corners' in pred_boxes[sample_idx]:
                corners_3d = pred_boxes[sample_idx]['corners'][on_image]  # [N_valid, 8, 3]
                corners_2d = self._project_box_corners_to_image(
                    corners_3d, shift, device
                ) / self.feat_stride  # 转到特征图坐标 [N_valid, 8, 2]

                feat_pos_single = self._img_feat_pos.to(device)  # [1, H*W, 2]
                attn_mask = self._compute_dynamic_gaussian(
                    q_pix_feat, corners_2d, on_image, feat_pos_single
                )  # [N_valid, feat_h*feat_w]
            else:
                # 没有角点信息时，用固定 sigma=2.0 作为后备
                feat_pos_single = self._img_feat_pos.to(device)  # [1, H*W, 2]
                center_xy = q_pix_feat.unsqueeze(1)
                key_pos   = feat_pos_single - 0.5
                distance  = (center_xy - key_pos).norm(dim=-1) ** 2
                gaussian  = (-distance / (2 * 2.0 ** 2)).exp()
                attn_mask = gaussian.log().clamp(min=-1e4)

            # ── 4. Camera Cross-Attention ──────────────────────
            # query: [B=1, C, N_valid]
            # key/value: [B=1, C, feat_h*feat_w]
            query_feat_view = prev_query_feat[sample_idx:sample_idx+1, :, on_image]
            query_pos_view  = q_pix_feat.unsqueeze(0)   # [1, N_valid, 2]
            img_key = img_feat_flatten[sample_idx:sample_idx+1]  # [1, C, H*W]

            # attn_mask 形状需要是 [N_valid, feat_h*feat_w] 给 multihead_attn
            # 官方代码第1001行: attn_mask=attn_mask.log()（这里已经是 log 了）
            query_feat_view = self.img_cross_decoder(
                query=query_feat_view,                      # [1, C, N_valid]
                key=img_key,                                # [1, C, H*W]
                query_pos=query_pos_view,                   # [1, N_valid, 2]
                key_pos=img_feat_pos[sample_idx:sample_idx+1],  # [1, H*W, 2]
                attn_mask=attn_mask,                        # [N_valid, H*W]
            )  # [1, C, N_valid]

            new_query_feat[sample_idx, :, on_image] = query_feat_view.squeeze(0)

        # ── E. 最终输出：concat prev + new（对应官方第1005行）──
        on_the_image_mask_bool = (on_the_image_mask != -1)  # [B, N_q] bool

        # 对于没有投影到图像的 query，用第一层结果填充（官方第1007-1009行）
        # 在外部 prediction_head 里统一处理

        query_feat_fused = torch.cat([new_query_feat, prev_query_feat], dim=1)  # [B, 2C, N_q]

        return query_feat_fused, dense_heatmap_img, on_the_image_mask_bool


# ─────────────────────────────────────────────────────────────
# 接入说明（在你的 transfusion_head.py 或 OpenPCDet 版 head 里）
# ─────────────────────────────────────────────────────────────
"""
【__init__ 里初始化】

from .transfusion_img_decoder import OusterTransFusionImgDecoder

self.img_decoder = OusterTransFusionImgDecoder(
    hidden_channel=128,
    num_heads=8,
    img_feat_channel=128,          # YOLOv8 P4
    num_classes=11,
    num_proposals=200,
    ffn_channel=256,
    dropout=0.1,
    img_h=128, img_w=1024,
    feat_stride=16,
    out_size_factor=8,
    voxel_size=0.1,
    pc_range=[-51.2, -60.0, -3.0, 51.2, 40.8, 5.0],
    beam_altitude_angles=dataset.beam_altitude_angles,
)

# 最终预测头：输入通道变为 2 * hidden_channel
self.img_prediction_head = FFN(hidden_channel * 2, heads, ...)


【forward_single 里调用，在第一层 LiDAR decoder 之后】

# 第一层 decoder 已经跑完，此时有：
#   query_feat:    [B, C, N_q]
#   query_pos:     [B, N_q, 2]
#   query_pos_3d:  [B, 3, N_q]  ← 需要从 res_layer['center'] 和 res_layer['height'] 拼出来
#   pred_boxes:    decode 出的 box（含 corners）

if 'img_feat' in batch_dict and self.fuse_img:
    # 拼出 3D 坐标（米制）
    center_metric = query_pos.permute(0,2,1) * self.out_size_factor * self.voxel_size + self.pc_range[0]
    query_pos_3d  = torch.cat([center_metric, res_layer['height']], dim=1)  # [B, 3, N_q]

    query_feat_fused, dense_heatmap_img, on_mask = self.img_decoder(
        lidar_feat_flatten=lidar_feat_flatten,
        bev_pos=bev_pos,
        query_feat=query_feat,
        query_pos=query_pos,
        query_pos_3d=query_pos_3d,
        pred_boxes=pred_boxes,
        img_feat=batch_dict['img_feat'],
        img_heatmap_lidar=dense_heatmap,
        shift_degrees_list=batch_dict['shift_degrees'],
    )

    # 最终预测
    res_layer_img = self.img_prediction_head(query_feat_fused)
    res_layer_img['center'] = res_layer_img['center'] + query_pos.permute(0, 2, 1)

    # on_the_image_mask 用于 loss 加权（官方第1234-1237行）
    self.on_the_image_mask = on_mask
"""