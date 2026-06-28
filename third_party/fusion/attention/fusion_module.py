"""
多模态融合模块 - 简化版
基于空白填充策略的PointPillars + YOLOv8融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# #
# class SmartPaddingFusionModule(nn.Module):
#     """智能填充融合模块"""
#
#     def __init__(self, dim=128, num_heads=8, dropout=0.1, use_interpolation=False, ablation_config=None):
#         super().__init__()
#
#         self.dim = dim
#         self.num_heads = num_heads
#         self.use_interpolation = use_interpolation
#
#         self.ablation_config = ablation_config or {}
#         self.use_cross_attention = self.ablation_config.get('USE_CROSS_ATTENTION', True)
#         self.use_self_attention = self.ablation_config.get('USE_SELF_ATTENTION', True)
#         print('self_attention', self.use_self_attention)
#
#         if self.use_cross_attention:
#             self.img_cross_attn = nn.MultiheadAttention(
#                 embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
#             )
#             self.lidar_cross_attn = nn.MultiheadAttention(
#                 embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
#             )
#             self.norm1 = nn.LayerNorm(dim)
#             self.norm2 = nn.LayerNorm(dim)
#
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True)
#         )
#
#         if self.use_self_attention:
#             self.self_attention = nn.MultiheadAttention(
#                 embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
#             )
#             self.norm3 = nn.LayerNorm(dim)
#
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(dim * 4, dim),
#             nn.Dropout(dropout)
#         )
#
#     def _smart_padding_or_pooling(self, img_feat, target_size):
#         """智能调整图像特征尺寸"""
#         B, C, H, W = img_feat.shape
#         target_H, target_W = target_size
#
#         if H == target_H and W == target_W:
#             return img_feat
#
#         if H <= target_H and W <= target_W:
#             if self.use_interpolation:
#                 return F.interpolate(img_feat, size=(target_H, target_W), mode='bilinear', align_corners=False)
#             else:
#                 start_h = (target_H - H) // 2
#                 start_w = (target_W - W) // 2
#                 padded = torch.zeros(B, C, target_H, target_W, device=img_feat.device, dtype=img_feat.dtype)
#                 padded[:, :, start_h:start_h + H, start_w:start_w + W] = img_feat
#                 return padded
#         else:
#             return F.adaptive_avg_pool2d(img_feat, (target_H, target_W))
#
#     def _to_sequence(self, feature_map):
#         """特征图转序列"""
#         B, C, H, W = feature_map.shape
#         return feature_map.view(B, C, -1).transpose(1, 2) # (B, H*W, C)
#
#     def _to_feature_map(self, sequence, H, W):
#         """序列转特征图"""
#         B, N, C = sequence.shape
#         return sequence.transpose(1, 2).view(B, C, H, W)
#
#     def forward(self, img_feat, lidar_feat):
#         """
#         Args:
#             img_feat: (B, 128, H, W) 图像特征
#             lidar_feat: (B, 128, H_l, W_l) 点云特征
#         Returns:
#             fused_feat: (B, 128, H_l, W_l) 融合特征
#             enhanced_features: (enhanced_img, enhanced_lidar) 增强特征
#         """
#         B, C, H, W = img_feat.shape
#         B, C, H_l, W_l = lidar_feat.shape
#
#         img_seq = self._to_sequence(img_feat)
#         lidar_seq = self._to_sequence(lidar_feat)
#
#         if self.use_cross_attention:
#             import time
#             # 只测fusion_module这一步
#             torch.cuda.synchronize()
#             start = time.time()
#             # torch.cuda.reset_peak_memory_stats()
#             img_cross_output, _ = self.img_cross_attn(
#                 query=img_seq, key=lidar_seq, value=lidar_seq, need_weights=True
#             )
#             enhanced_img_seq = img_cross_output
#
#             lidar_cross_output, _ = self.lidar_cross_attn(
#                 query=lidar_seq, key=img_seq, value=img_seq, need_weights=True
#             )
#             enhanced_lidar_seq = lidar_cross_output
#             torch.cuda.synchronize()
#             # print('7777777777777777', time.time() - start)
#         else:
#             enhanced_img_seq = img_seq
#             enhanced_lidar_seq = lidar_seq
#
#         enhanced_img_feat = self._to_feature_map(enhanced_img_seq, H, W)
#         enhanced_lidar_feat = self._to_feature_map(enhanced_lidar_seq, H_l, W_l)
#
#         adjusted_img_feat = self._smart_padding_or_pooling(enhanced_img_feat, (H_l, W_l))
#
#         concat_feat = torch.cat([adjusted_img_feat, enhanced_lidar_feat], dim=1)
#         fused_feat = self.fusion_conv(concat_feat)
#
#         if self.use_self_attention:
#             fused_seq = self._to_sequence(fused_feat)
#             self_attn_output, _ = self.self_attention(
#                 query=fused_seq, key=fused_seq, value=fused_seq
#             )
#             enhanced_fused_seq = self.norm3(fused_seq + self_attn_output)
#             ffn_output = self.ffn(enhanced_fused_seq)
#             final_fused_seq = enhanced_fused_seq + ffn_output
#             fused_feat = self._to_feature_map(final_fused_seq, H_l, W_l)
#
#         return fused_feat, (enhanced_img_feat, enhanced_lidar_feat)

#
# # mamba
# class SmartPaddingFusionModule(nn.Module):
#     """智能填充融合模块"""
#
#     def __init__(self, dim=128, num_heads=8, dropout=0.1, use_interpolation=False, ablation_config=None):
#         super().__init__()
#
#         self.dim = dim
#         self.num_heads = num_heads
#         self.use_interpolation = use_interpolation
#
#         self.ablation_config = ablation_config or {}
#         self.use_cross_attention = self.ablation_config.get('USE_CROSS_ATTENTION', True)
#         self.use_self_attention = self.ablation_config.get('USE_SELF_ATTENTION', True)
#         print('self_attention', self.use_self_attention)
#
#         import sys
#         sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
#         from mamba.vmamba import CrossMambaFusionBlock
#
#         if self.use_cross_attention:
#             self.cross_mamba = CrossMambaFusionBlock(
#                 hidden_dim=dim,  # 128
#                 mlp_ratio=0.0,
#                 d_state=4,
#             )
#
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True)
#         )
#
#         if self.use_self_attention:
#             self.self_attention = nn.MultiheadAttention(
#                 embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
#             )
#             self.norm3 = nn.LayerNorm(dim)
#
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(dim * 4, dim),
#             nn.Dropout(dropout)
#         )
#
#     def _smart_padding_or_pooling(self, img_feat, target_size):
#         """智能调整图像特征尺寸"""
#         B, C, H, W = img_feat.shape
#         target_H, target_W = target_size
#
#         if H == target_H and W == target_W:
#             return img_feat
#
#         if H <= target_H and W <= target_W:
#             if self.use_interpolation:
#                 return F.interpolate(img_feat, size=(target_H, target_W), mode='bilinear', align_corners=False)
#             else:
#                 start_h = (target_H - H) // 2
#                 start_w = (target_W - W) // 2
#                 padded = torch.zeros(B, C, target_H, target_W, device=img_feat.device, dtype=img_feat.dtype)
#                 padded[:, :, start_h:start_h + H, start_w:start_w + W] = img_feat
#                 return padded
#         else:
#             return F.adaptive_avg_pool2d(img_feat, (target_H, target_W))
#
#     def _to_sequence(self, feature_map):
#         """特征图转序列"""
#         B, C, H, W = feature_map.shape
#         return feature_map.view(B, C, -1).transpose(1, 2) # (B, H*W, C)
#
#     def _to_feature_map(self, sequence, H, W):
#         """序列转特征图"""
#         B, N, C = sequence.shape
#         return sequence.transpose(1, 2).view(B, C, H, W)
#
#     def forward(self, img_feat, lidar_feat):
#         """
#         Args:
#             img_feat: (B, 128, H, W) 图像特征
#             lidar_feat: (B, 128, H_l, W_l) 点云特征
#         Returns:
#             fused_feat: (B, 128, H_l, W_l) 融合特征
#             enhanced_features: (enhanced_img, enhanced_lidar) 增强特征
#         """
#         B, C, H, W = img_feat.shape
#         B, C, H_l, W_l = lidar_feat.shape
#
#         # ⭐ 关键改动：先对齐尺寸，再做Mamba（Mamba要求两个输入尺寸相同）
#         # 原来是先做cross attention（允许不同尺寸），再padding
#         # 现在改为先padding，再Mamba
#         adjusted_img_feat = self._smart_padding_or_pooling(img_feat, (H_l, W_l))
#         # 此时 adjusted_img_feat 和 lidar_feat 都是 (B, C, H_l, W_l)
#
#         if self.use_cross_attention:
#             # Mamba要求输入格式是 (B, H, W, C)，做permute
#             img_bhwc = adjusted_img_feat.permute(0, 2, 3, 1).contiguous()
#             lidar_bhwc = lidar_feat.permute(0, 2, 3, 1).contiguous()
#
#             # CrossMambaFusionBlock：双向跨模态增强
#             enhanced_img_bhwc, enhanced_lidar_bhwc = self.cross_mamba(img_bhwc, lidar_bhwc)
#
#             # 转回 (B, C, H, W)
#             enhanced_img_feat = enhanced_img_bhwc.permute(0, 3, 1, 2).contiguous()
#             enhanced_lidar_feat = enhanced_lidar_bhwc.permute(0, 3, 1, 2).contiguous()
#         else:
#             enhanced_img_feat = adjusted_img_feat
#             enhanced_lidar_feat = lidar_feat
#
#         concat_feat = torch.cat([enhanced_img_feat, enhanced_lidar_feat], dim=1)
#         fused_feat = self.fusion_conv(concat_feat)
#
#         if self.use_self_attention:
#             fused_seq = self._to_sequence(fused_feat)
#             self_attn_output, _ = self.self_attention(
#                 query=fused_seq, key=fused_seq, value=fused_seq
#             )
#             enhanced_fused_seq = self.norm3(fused_seq + self_attn_output)
#             ffn_output = self.ffn(enhanced_fused_seq)
#             final_fused_seq = enhanced_fused_seq + ffn_output
#             fused_feat = self._to_feature_map(final_fused_seq, H_l, W_l)
#
#         return fused_feat, (enhanced_img_feat, enhanced_lidar_feat)
# mamba with concatmamba
class SmartPaddingFusionModule(nn.Module):
    """智能填充融合模块"""

    def __init__(self, dim=128, num_heads=8, dropout=0.1, use_interpolation=False, ablation_config=None):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.use_interpolation = use_interpolation

        self.ablation_config = ablation_config or {}
        self.use_cross_attention = self.ablation_config.get('USE_CROSS_ATTENTION', True)
        self.use_self_attention = self.ablation_config.get('USE_SELF_ATTENTION', True)
        print('self_attention', self.use_self_attention)

        import sys
        sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
        from mamba.vmamba import CrossMambaFusionBlock, ConcatMambaFusionBlock

        if self.use_cross_attention:
            self.cross_mamba = CrossMambaFusionBlock(
                hidden_dim=dim,  # 128
                mlp_ratio=0.0,
                d_state=4,
            )

        # self.concat_mamba = ConcatMambaFusionBlock(
        #     hidden_dim=128,
        #     mlp_ratio=0.0,
        #     d_state=4,
        # )

        self.concat_mamba = ConcatMambaFusionBlock(
            hidden_dim=128,
            mlp_ratio=0.0,
            d_state=4,
            use_gate=True,
        )

        if self.use_self_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm3 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def _smart_padding_or_pooling(self, img_feat, target_size):
        """智能调整图像特征尺寸"""
        B, C, H, W = img_feat.shape
        target_H, target_W = target_size

        if H == target_H and W == target_W:
            return img_feat

        if H <= target_H and W <= target_W:
            if self.use_interpolation:
                return F.interpolate(img_feat, size=(target_H, target_W), mode='bilinear', align_corners=False)
            else:
                start_h = (target_H - H) // 2
                start_w = (target_W - W) // 2
                padded = torch.zeros(B, C, target_H, target_W, device=img_feat.device, dtype=img_feat.dtype)
                padded[:, :, start_h:start_h + H, start_w:start_w + W] = img_feat
                return padded
        else:
            return F.adaptive_avg_pool2d(img_feat, (target_H, target_W))

    def _to_sequence(self, feature_map):
        """特征图转序列"""
        B, C, H, W = feature_map.shape
        return feature_map.view(B, C, -1).transpose(1, 2) # (B, H*W, C)

    def _to_feature_map(self, sequence, H, W):
        """序列转特征图"""
        B, N, C = sequence.shape
        return sequence.transpose(1, 2).view(B, C, H, W)

    def forward(self, img_feat, lidar_feat):
        """
        Args:
            img_feat: (B, 128, H, W) 图像特征
            lidar_feat: (B, 128, H_l, W_l) 点云特征
        Returns:
            fused_feat: (B, 128, H_l, W_l) 融合特征
            enhanced_features: (enhanced_img, enhanced_lidar) 增强特征
        """
        B, C, H, W = img_feat.shape
        B, C, H_l, W_l = lidar_feat.shape

        # ⭐ 关键改动：先对齐尺寸，再做Mamba（Mamba要求两个输入尺寸相同）
        # 原来是先做cross attention（允许不同尺寸），再padding
        # 现在改为先padding，再Mamba
        adjusted_img_feat = self._smart_padding_or_pooling(img_feat, (H_l, W_l))
        # 此时 adjusted_img_feat 和 lidar_feat 都是 (B, C, H_l, W_l)

        if self.use_cross_attention:
            # Mamba要求输入格式是 (B, H, W, C)，做permute
            img_bhwc = adjusted_img_feat.permute(0, 2, 3, 1).contiguous()
            lidar_bhwc = lidar_feat.permute(0, 2, 3, 1).contiguous()
            # import time
            # # 只测fusion_module这一步
            # torch.cuda.synchronize()
            # start = time.time()

            # CrossMambaFusionBlock：双向跨模态增强
            enhanced_img_bhwc, enhanced_lidar_bhwc = self.cross_mamba(img_bhwc, lidar_bhwc)

            # torch.cuda.synchronize()
            # print('7777777777777777',time.time() - start)

            # 转回 (B, C, H, W)
            enhanced_img_feat = enhanced_img_bhwc.permute(0, 3, 1, 2).contiguous()
            enhanced_lidar_feat = enhanced_lidar_bhwc.permute(0, 3, 1, 2).contiguous()
        else:
            enhanced_img_feat = adjusted_img_feat
            enhanced_lidar_feat = lidar_feat

        img_bhwc = enhanced_img_feat.permute(0, 2, 3, 1).contiguous()
        lidar_bhwc = enhanced_lidar_feat.permute(0, 2, 3, 1).contiguous()
        # # 只测fusion_module这一步
        # torch.cuda.synchronize()
        # start = time.time()
        fused_feat = self.concat_mamba(img_bhwc, lidar_bhwc).permute(0, 3, 1, 2).contiguous()
        # torch.cuda.synchronize()
        # print('7777777777777777', time.time() - start)

        if self.use_self_attention:
            fused_seq = self._to_sequence(fused_feat)
            self_attn_output, _ = self.self_attention(
                query=fused_seq, key=fused_seq, value=fused_seq
            )
            enhanced_fused_seq = self.norm3(fused_seq + self_attn_output)
            ffn_output = self.ffn(enhanced_fused_seq)
            final_fused_seq = enhanced_fused_seq + ffn_output
            fused_feat = self._to_feature_map(final_fused_seq, H_l, W_l)

        return fused_feat, (enhanced_img_feat, enhanced_lidar_feat)


def generate_random_noise_features(yolo_features):
    """
    生成与YOLO特征相同shape的随机噪声

    Args:
        yolo_features: List of 3 YOLO feature tensors [(B,128,H,W), (B,256,H,W), (B,512,H,W)]

    Returns:
        noise_features: List of 3 random noise tensors with same shapes
    """
    noise_features = []
    for feat in yolo_features:
        # 生成与输入特征相同shape的标准正态分布噪声
        noise = torch.randn_like(feat)
        noise_features.append(noise)

    print("🔊 使用随机噪声替代图像特征进行融合")
    return noise_features


class MultiModalFusionForPointPillars(nn.Module):
    """PointPillars多尺度融合模块"""

    def __init__(self, num_heads=8, dropout=0.1, unified_dim=128, use_interpolation=False, ablation_config=None):
        super().__init__()

        self.unified_dim = unified_dim
        self.use_interpolation = use_interpolation

        self.lidar_pre_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((32, 64)),
            nn.AdaptiveAvgPool2d((32, 64)),
            nn.AdaptiveAvgPool2d((32, 64))
        ])

        self.img_projs = nn.ModuleList([
            nn.Conv2d(128, unified_dim, 1),
            nn.Conv2d(256, unified_dim, 1),
            nn.Conv2d(512, unified_dim, 1)
        ])

        self.lidar_projs = nn.ModuleList([
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 128, 1)
        ])

        self.fusion_modules = nn.ModuleList([
            SmartPaddingFusionModule(
                dim=128, num_heads=num_heads, dropout=dropout,
                use_interpolation=use_interpolation, ablation_config=ablation_config
            )
            for _ in range(3)
        ])

        self.final_upsample = nn.ModuleList([
            nn.Upsample(size=(316, 320), mode='bilinear', align_corners=False),
            nn.Upsample(size=(316, 320), mode='bilinear', align_corners=False),
            nn.Upsample(size=(316, 320), mode='bilinear', align_corners=False)
        ])

        self.output_proj = nn.Conv2d(128, 128, 1)

    def forward(self, img_feats, lidar_up_feats):
        """
        Args:
            img_feats: YOLOv8 3尺度特征 [(B,128,H,W), (B,256,H,W), (B,512,H,W)]
            lidar_up_feats: PointPillars 3尺度特征 [(B,128,H,W), ...]
        Returns:
            fused_feats: 融合后特征 [(B,128,316,320), ...]
            enhanced_features: 增强特征列表
        """
        fused_feats = []
        enhanced_pp_yy_features = []

        for i, (img_f, lidar_f) in enumerate(zip(img_feats, lidar_up_feats)):
            lidar_pooled = self.lidar_pre_pools[i](lidar_f)
            img_unified = self.img_projs[i](img_f)
            lidar_unified = lidar_pooled

            img_flatten = F.adaptive_avg_pool2d(img_unified, (1, 1)).flatten(1)
            lidar_flatten = F.adaptive_avg_pool2d(lidar_unified, (1, 1)).flatten(1)
            # before_sim = F.cosine_similarity(img_flatten, lidar_flatten, dim=1).mean()
            # print(f'尺度{i} 融合前相似度: {before_sim:.4f}')

            fused_feat, enhanced_cross_feat = self.fusion_modules[i](img_unified, lidar_unified)

            after_img_fea, after_lidar_fea = enhanced_cross_feat
            after_img_flatten = F.adaptive_avg_pool2d(after_img_fea, (1, 1)).flatten(1)
            after_lidar_flatten = F.adaptive_avg_pool2d(after_lidar_fea, (1, 1)).flatten(1)
            after_sim = F.cosine_similarity(after_img_flatten, after_lidar_flatten, dim=1).mean()
            # print(f'尺度{i} 融合后相似度: {after_sim:.4f}')

            final_feat = self.final_upsample[i](fused_feat)
            output_feat = self.output_proj(final_feat)

            fused_feats.append(output_feat)
            enhanced_pp_yy_features.append(enhanced_cross_feat)

        return fused_feats, enhanced_pp_yy_features
# """
# 智能填充的多模态融合模块 - 完整实现
# 基于空白填充策略的PointPillars + YOLOv8融合方案
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# class SmartPaddingFusionModule(nn.Module):
#     """
#     智能填充融合模块
#     使用空白填充代替插值，保持特征真实性和空间对应关系
#     """
#
#     # def __init__(self, dim=128, num_heads=8, dropout=0.1, use_interpolation=False):
#     def __init__(self, dim=128, num_heads=8, dropout=0.1, use_interpolation=False, ablation_config=None):
#         super().__init__()
#
#         self.dim = dim
#         self.num_heads = num_heads
#         self.use_interpolation = use_interpolation
#
#         # 🔥 消融实验配置
#         self.ablation_config = ablation_config or {}
#         print('fusion_ablation_config: ',self.ablation_config)
#         self.use_cross_attention = self.ablation_config.get('USE_CROSS_ATTENTION', True)
#         self.use_self_attention = self.ablation_config.get('USE_SELF_ATTENTION', True)
#
#         # print(f"🔬 融合模块消融配置:")
#         # print(f"   交叉注意力: {self.use_cross_attention}")
#         # print(f"   自注意力: {self.use_self_attention}")
#
#         # 交叉注意力模块（可选）
#         if self.use_cross_attention:
#             self.img_cross_attn = nn.MultiheadAttention(
#                 embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
#             )
#             self.lidar_cross_attn = nn.MultiheadAttention(
#                 embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
#             )
#             self.norm1 = nn.LayerNorm(dim)
#             self.norm2 = nn.LayerNorm(dim)
#
#         # 特征融合卷积层
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
#             # nn.Conv2d(dim, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True)
#         )
#
#         # 自注意力精炼模块（可选）
#         if self.use_self_attention:
#             self.self_attention = nn.MultiheadAttention(
#                 embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
#             )
#             self.norm3 = nn.LayerNorm(dim)
#
#         # Feed Forward Network
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(dim * 4, dim),
#             nn.Dropout(dropout)
#         )
#
#     def _smart_padding_or_pooling(self, img_feat, target_size):
#         """
#         智能调整：根据图像和点云的尺寸关系选择填充或池化
#         - 图像尺寸 < 点云尺寸：使用零填充 (padding)
#         - 图像尺寸 >= 点云尺寸：使用平均池化 (average pooling)
#
#         Args:
#             img_feat: (B, C, H, W) 图像特征
#             target_size: (target_H, target_W) 目标尺寸（点云特征尺寸）
#
#         Returns:
#             adjusted_feat: (B, C, target_H, target_W) 调整后的特征
#         """
#         B, C, H, W = img_feat.shape
#         target_H, target_W = target_size
#
#         # 如果已经是目标尺寸，直接返回
#         if H == target_H and W == target_W:
#             return img_feat
#
#         # 判断是否需要填充还是池化
#         if H <= target_H and W <= target_W:
#             # 图像尺寸小于等于点云尺寸
#             if self.use_interpolation:
#                 print('use_interpolation')
#                 # 使用插值上采样
#                 return F.interpolate(img_feat, size=(target_H, target_W), mode='bilinear', align_corners=False)
#             else:
#                 # 使用零填充（居中）
#                 # 计算居中填充的位置
#                 start_h = (target_H - H) // 2
#                 start_w = (target_W - W) // 2
#
#                 # 创建零填充的目标tensor
#                 padded = torch.zeros(B, C, target_H, target_W,
#                                      device=img_feat.device, dtype=img_feat.dtype)
#
#                 # 将原始特征放在中心位置
#                 end_h = start_h + H
#                 end_w = start_w + W
#                 padded[:, :, start_h:end_h, start_w:end_w] = img_feat
#
#                 return padded
#         else:
#             # 图像尺寸大于点云尺寸，使用平均池化
#             # 使用自适应平均池化调整到目标尺寸
#             pooled = F.adaptive_avg_pool2d(img_feat, (target_H, target_W))
#             return pooled
#
#     def _to_sequence(self, feature_map):
#         """将特征图转换为序列格式"""
#         B, C, H, W = feature_map.shape
#         return feature_map.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
#
#     def _to_feature_map(self, sequence, H, W):
#         """将序列转换回特征图格式"""
#         B, N, C = sequence.shape
#         return sequence.transpose(1, 2).view(B, C, H, W)
#
#     def forward(self, img_feat, lidar_feat):
#         """
#         Args:
#             img_feat: (B, 128, H, W) - 图像特征
#             lidar_feat: (B, 128, H_l, W_l) - 点云特征
#
#         Returns:
#             fused_feat: (B, 128, H_l, W_l) - 融合后的特征，保持点云特征的空间尺寸
#         """
#         B, C, H, W = img_feat.shape
#         B, C, H_l, W_l = lidar_feat.shape
#
#         # Step 1: 转换为序列格式（保持原始尺寸）
#         img_seq = self._to_sequence(img_feat)  # (B, H*W, 128) - 原始图像尺寸
#         lidar_seq = self._to_sequence(lidar_feat)  # (B, H_l*W_l, 128) - 点云尺寸
#
#         # Step 2: 交叉注意力 - 信息注入阶段
#         if self.use_cross_attention:
#             # 图像学习点云信息（零填充区域会学到点云信息）
#             img_cross_output, img_attn_weights = self.img_cross_attn(
#                 query=img_seq,  # (B, H_l*W_l, 128)
#                 key=lidar_seq,  # (B, H_l*W_l, 128)
#                 value=lidar_seq,  # (B, H_l*W_l, 128)
#                 need_weights=True  # 关键：获取attention权重
#             )
#
#             # 应用残差连接
#             enhanced_img_seq = img_cross_output
#
#             # enhanced_img_seq = self.norm1(img_seq + img_cross_output)
#             # enhanced_img_seq = self.norm1(img_seq + img_cross_output)  # 残差连接
#
#             # 点云学习图像信息（从有效的图像区域学习语义信息）
#             lidar_cross_output, lidar_attn_weights = self.lidar_cross_attn(
#                 query=lidar_seq,  # (B, 32, 64, 128)
#                 key=img_seq,  # (B, H*W, 128)
#                 value=img_seq,  # (B, H*W, 128)
#                 need_weights=True  # 关键：获取attention权重
#             )
#
#             enhanced_lidar_seq = lidar_cross_output
#
#             enhanced_img_feat = self._to_feature_map(enhanced_img_seq, H, W)  # (B, 128, H_l, W_l)
#
#             enhanced_lidar_feat = self._to_feature_map(enhanced_lidar_seq, H_l, W_l)
#
#             # enhanced_lidar_seq = self.norm2(lidar_seq + lidar_cross_output)
#             #
#             # before_norm_lidar = lidar_seq + lidar_cross_output
#             #
#             # print(
#             #     f"LayerNorm前点云特征统计: mean={before_norm_lidar.mean().item():.4f}, std={before_norm_lidar.std().item():.4f}")
#             # print(
#             #     f"LayerNorm后点云特征统计: mean={enhanced_lidar_seq.mean().item():.4f}, std={enhanced_lidar_seq.std().item():.4f}")
#             # enhanced_lidar_seq = self.norm2(lidar_seq + lidar_cross_output)  # 残差连接
#             # print("✓ 使用交叉注意力增强特征")
#         else:
#             # 跳过交叉注意力，直接使用原始特征
#             enhanced_img_seq = img_seq
#             enhanced_lidar_seq = lidar_seq
#             print("⊗ 跳过交叉注意力")
#         # Step 3: 转换回特征图格式
#         enhanced_img_feat = self._to_feature_map(enhanced_img_seq, H, W)  # (B, 128, H_l, W_l)
#
#         enhanced_lidar_feat = self._to_feature_map(enhanced_lidar_seq, H_l, W_l)  # (B, 128, H_l, W_l)
#
#         # Step 4: 将增强后的图像特征调整到点云特征的尺寸（填充或池化）
#         adjusted_img_feat = self._smart_padding_or_pooling(enhanced_img_feat, (H_l, W_l))  # (B, 128, H_l, W_l)
#
#         # Step 5: 特征拼接与融合（此时尺寸一致且都是增强后的有效信息）
#         concat_feat = torch.cat([adjusted_img_feat, enhanced_lidar_feat], dim=1)  # (B, 256, H_l, W_l)
#         # concat_feat = adjusted_img_feat + enhanced_lidar_feat
#         fused_feat = self.fusion_conv(concat_feat)  # (B, 128, H_l, W_l)
#         print('concat_feature',concat_feat.shape,'conv_feature',fused_feat.shape)
#
#         # Step 6: 自注意力精炼（全域有效信息的相互增强）
#         if self.use_self_attention:
#             fused_seq = self._to_sequence(fused_feat)  # (B, H_l*W_l, 128)
#
#             self_attn_output, _ = self.self_attention(
#                 query=fused_seq,
#                 key=fused_seq,
#                 value=fused_seq
#             )
#             refined_seq = self.norm3(fused_seq + self_attn_output)  # 残差连接
#
#             # Feed Forward Network
#             ffn_output = self.ffn(refined_seq)
#             final_seq = refined_seq + ffn_output  # 残差连接
#
#             # 转换回特征图
#             final_feat = self._to_feature_map(final_seq, H_l, W_l)  # (B, 128, H_l, W_l)
#             # print("✓ 使用自注意力精炼特征")
#         else:
#             # # 跳过自注意力，直接使用融合特征
#             # final_feat = fused_feat
#             print("⊗ 跳过自注意力")
#             final_seq = self._to_sequence(fused_feat)  # (B, H_l, W_l, 128)
#             ffn_seq = self.ffn(final_seq)
#             final_seq = final_seq + ffn_seq  # 残差连接
#
#             # 转换回特征图
#             final_feat = self._to_feature_map(final_seq, H_l, W_l)  # (B, 128, H_l, W_l)
#             # print("⊗ 跳过自注意力")
#         # return final_feat
#         enhanced_cross_feat = [enhanced_img_feat, enhanced_lidar_feat]
#         return final_feat, enhanced_cross_feat
#
#
# class MultiModalFusionForPointPillars(nn.Module):
#     """
#     完整的多模态融合模块，适配PointPillars检测头
#     使用智能填充策略
#     """
#
#     def __init__(self, num_heads=8, dropout=0.1, unified_dim=128, use_interpolation=False,ablation_config=None):
#         super().__init__()
#
#         # self.unified_dim = 128
#         self.unified_dim = unified_dim
#         self.use_interpolation = use_interpolation
#         self.ablation_config = ablation_config or {}
#
#         # self.lidar_pre_pools = nn.ModuleList([
#         #     nn.AdaptiveAvgPool2d((16, 64)),  # Scale 0: (152,608) -> (16,64)
#         #     nn.AdaptiveAvgPool2d((16, 64)),  # Scale 1: (152,608) -> (16,64)
#         #     nn.AdaptiveAvgPool2d((16, 64))  # Scale 2: (152,608) -> (16,64)
#         # ])
#
#         self.lidar_pre_pools = nn.ModuleList([
#             nn.AdaptiveAvgPool2d((32, 64)),  # Scale 0: (152,608) -> (16,64)
#             nn.AdaptiveAvgPool2d((32, 64)),  # Scale 1: (152,608) -> (16,64)
#             nn.AdaptiveAvgPool2d((32, 64))  # Scale 2: (152,608) -> (16,64)
#         ])
#         # 图像特征投影到统一维度
#         self.img_projs = nn.ModuleList([
#             nn.Conv2d(128, self.unified_dim, 1),  # Scale 0: 128->128
#             nn.Conv2d(256, self.unified_dim, 1),  # Scale 1: 256->128
#             nn.Conv2d(512, self.unified_dim, 1)  # Scale 2: 512->128
#         ])
#
#         # 点云上采样特征投影到统一维度
#         self.lidar_projs = nn.ModuleList([
#             nn.Conv2d(128, 128, 1),  # Scale 0: 64->128
#             nn.Conv2d(128, 128, 1),  # Scale 1: 128->128
#             nn.Conv2d(128, 128, 1)  # Scale 2: 256->128
#         ])
#
#         # 每个尺度的融合模块
#         self.fusion_modules = nn.ModuleList([
#             SmartPaddingFusionModule(dim=128, num_heads=num_heads, dropout=dropout, use_interpolation=use_interpolation, ablation_config=ablation_config)
#             for _ in range(3)
#         ])
#
#         # # 最终上采样到统一尺寸的模块
#         # self.final_upsample = nn.ModuleList([
#         #     nn.Identity(),  # Scale 0: (248,216) -> (248,216) 不需要上采样
#         #     nn.Upsample(size=(16, 64), mode='bilinear', align_corners=False),  # Scale 1: (124,108) -> (248,216)
#         #     nn.Upsample(size=(16, 64), mode='bilinear', align_corners=False)  # Scale 2: (62,54) -> (248,216)
#         # ])
#
#         # self.final_upsample = nn.ModuleList([
#         #     nn.Upsample(size=(152, 608), mode='bilinear', align_corners=False),
#         #     nn.Upsample(size=(152, 608), mode='bilinear', align_corners=False),  # Scale 1: (124,108) -> (248,216)
#         #     nn.Upsample(size=(152, 608), mode='bilinear', align_corners=False)  # Scale 2: (62,54) -> (248,216)
#         # ])
#         self.final_upsample = nn.ModuleList([
#             nn.Upsample(size=(316, 320), mode='bilinear', align_corners=False),
#             nn.Upsample(size=(316, 320), mode='bilinear', align_corners=False),  # Scale 1: (124,108) -> (248,216)
#             nn.Upsample(size=(316, 320), mode='bilinear', align_corners=False)  # Scale 2: (62,54) -> (248,216)
#         ])
#
#         # 输出投影
#         self.output_proj = nn.Conv2d(128, 128, 1)
#
#     def forward(self, img_feats, lidar_up_feats):
#         """
#         Args:
#             img_feats: List[Tensor] - YOLOv8的3个尺度特征
#                 [(B,128,16,128), (B,256,8,64), (B,512,4,32)]
#             lidar_up_feats: List[Tensor] - PointPillars下采样特征
#                 [(B,64,248,216), (B,128,124,108), (B,256,62,54)]
#
#         Returns:
#             fused_feats: List[Tensor] - 融合后的3个128维特征
#                 [(B,128,248,216), (B,128,248,216), (B,128,248,216)]
#                 这些特征会被PointPillars拼接成384维送入检测头
#         """
#         fused_feats = []
#         enhanced_pp_yy_features = []
#         print('use_interpolation: ',self.use_interpolation)
#
#         for i, (img_f, lidar_f) in enumerate(zip(img_feats, lidar_up_feats)):
#             print('lidar_f size: ',lidar_f.shape)
#             # Step 0: 点云特征预处理池化(316, 320) -> (32, 64) (152, 608) -> (16, 64)
#             lidar_pooled = self.lidar_pre_pools[i](lidar_f)  # -> (B, 128, 32, 64)
#
#             # Step 1: 投影到统一维度
#             img_unified = self.img_projs[i](img_f)  # -> (B, 128, H, W)
#             # lidar_unified = self.lidar_projs[i](lidar_f)  # -> (B, 128, H_l, W_l)
#
#             lidar_unified = lidar_pooled
#
#             # print('i',i,'img_f',img_f.shape,'img_unified',img_unified.shape)
#             # print('lidar_f',lidar_f.shape,'lidar_unified',lidar_unified.shape)
#
#             img_flatten = F.adaptive_avg_pool2d(img_unified, (1, 1)).flatten(1)
#             lidar_flatten = F.adaptive_avg_pool2d(lidar_unified, (1, 1)).flatten(1)
#             before_cross_sim = F.cosine_similarity(img_flatten, lidar_flatten, dim=1).mean()
#             print('before cross attention, images and point cloud sim: ', before_cross_sim)
#             # Step 2: 智能填充融合（在点云的空间尺寸下进行）
#             # fused_feat = self.fusion_modules[i](img_unified, lidar_unified)  # -> (B, 128, H_l, W_l)
#             fused_feat, enhanced_cross_feat = self.fusion_modules[i](img_unified, lidar_unified)  # -> (B, 128, H_l, W_l)
#             after_img_fea, after_lidar_fea = enhanced_cross_feat
#             after_img_flatten = F.adaptive_avg_pool2d(after_img_fea, (1, 1)).flatten(1)
#             after_lidar_flatten = F.adaptive_avg_pool2d(after_lidar_fea, (1, 1)).flatten(1)
#             after_global_cross_sim = F.cosine_similarity(after_img_flatten, after_lidar_flatten, dim=1).mean()
#             print('after cross attention, global images and point cloud sim: ', after_global_cross_sim)
#             # # Step 3: 上采样到最终尺寸(248, 216)
#             final_feat = self.final_upsample[i](fused_feat)  # -> (B, 128, 152, 608)
#
#             # Step 4: 输出投影
#             output_feat = self.output_proj(final_feat)
#             # output_feat = self.output_proj(fused_feat)
#
#             fused_feats.append(output_feat)
#             enhanced_pp_yy_features.append(enhanced_cross_feat)
#
#         # return fused_feats
#         return fused_feats, enhanced_pp_yy_features
#
# # 测试代码
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     print("=== 测试智能填充多模态融合模块 ===")
#
#     # 模拟YOLOv8特征（3个尺度）
#     img_feats = [
#         torch.randn(16, 128, 16, 128, device=device),  # Scale 0 - 比点云大，需要池化
#         torch.randn(16, 256, 8, 64, device=device),  # Scale 1 - 比点云小，需要填充
#         torch.randn(16, 512, 4, 32, device=device)  # Scale 2 - 比点云小，需要填充
#     ]
#
#     # 模拟PointPillars上采样特征
#     # lidar_up_feats = [
#     #     torch.randn(16, 128, 16, 64, device=device),  # Scale 0
#     #     torch.randn(16, 128, 16, 64, device=device),  # Scale 1
#     #     torch.randn(16, 128, 16, 64, device=device)  # Scale 2
#     # ]
#     # lidar_up_feats = [
#     #     torch.randn(16, 128, 152, 608, device=device),  # Scale 0
#     #     torch.randn(16, 128, 152, 608, device=device),  # Scale 1
#     #     torch.randn(16, 128, 152, 608, device=device)  # Scale 2
#     # ]
#     lidar_up_feats = [
#         torch.randn(16, 128, 316, 320, device=device),  # Scale 0
#         torch.randn(16, 128, 316, 320, device=device),  # Scale 1
#         torch.randn(16, 128, 316, 320, device=device)  # Scale 2
#     ]
#
#     # 测试两种填充方式
#     for use_interp in [False, True]:
#         print(f"\n=== 测试{'插值' if use_interp else '零填充'}模式 ===")
#
#         # 创建融合模块
#         fusion_module = MultiModalFusionForPointPillars(
#             num_heads=8, dropout=0.1, unified_dim=128, use_interpolation=use_interp
#         ).to(device)
#
#         # # 创建融合模块
#         # fusion_module = MultiModalFusionForPointPillars(num_heads=8, dropout=0.1, unified_dim=128).to(device)
#
#         print(f"模型参数量: {sum(p.numel() for p in fusion_module.parameters() if p.requires_grad):,}")
#
#         # 前向传播
#         print("\n开始前向传播...")
#         try:
#             with torch.no_grad():
#                 # fused_features = fusion_module(img_feats, lidar_up_feats)
#                 fused_features, enhanced_pp_yy_features = fusion_module(img_feats, lidar_up_feats)
#
#                 print("✓ 融合成功！输出特征尺寸:")
#                 total_channels = 0
#                 for i, feat in enumerate(fused_features):
#                     print(f"  Scale {i}: {feat.shape}")
#                     total_channels += feat.shape[1]
#
#                 print(f"\n送入PointPillars检测头的总通道数: {total_channels} (3×128 = 384)")
#
#                 # 验证梯度流
#                 print("\n验证梯度计算...")
#                 dummy_loss = sum(feat.mean() for feat in fused_features)
#
#             # 测试梯度（需要开启梯度）
#             fusion_module.train()
#             fused_features, enhanced_pp_yy_features = fusion_module(img_feats, lidar_up_feats)
#             dummy_loss = sum(feat.mean() for feat in fused_features)
#             dummy_loss.backward()
#             print("✓ 梯度计算正常")
#
#             # 测试智能填充功能
#             print("\n=== 测试智能填充功能 ===")
#             padding_module = SmartPaddingFusionModule().to(device)
#
#             with torch.no_grad():
#                 # 测试池化：大图像 -> 小点云
#                 large_img = torch.randn(1, 128, 16, 128, device=device)
#                 small_lidar = torch.randn(1, 128, 16, 64, device=device)
#                 # result1 = padding_module(large_img, small_lidar)
#                 result1, enhanced_cross_feat1 = padding_module(large_img, small_lidar)
#                 print(f"大图像 {large_img.shape} + 小点云 {small_lidar.shape} -> {result1.shape} (使用池化)")
#
#                 # 测试填充：小图像 -> 大点云
#                 small_img = torch.randn(1, 128, 4, 32, device=device)
#                 large_lidar = torch.randn(1, 128, 16, 64, device=device)
#                 # result2 = padding_module(small_img, large_lidar)
#                 result2, enhanced_cross_feat2 = padding_module(large_img, small_lidar)
#                 print(f"小图像 {small_img.shape} + 大点云 {large_lidar.shape} -> {result2.shape} (使用填充)")
#
#                 print("✓ 智能填充和池化测试通过")
#
#         except Exception as e:
#             print(f"✗ 测试失败: {e}")
#             import traceback
#
#             traceback.print_exc()
#
#         print("\n=== 测试完成 ===")