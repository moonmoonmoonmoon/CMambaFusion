from .detector3d_template import Detector3DTemplate
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys

sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
from fusion.wrappers.yolo_extractor import YOLOv8FeatureExtractor
from fusion.attention.fusion_module import MultiModalFusionForPointPillars
from fusion.attention.fusion_module import generate_random_noise_features


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        self.fusion_config = self.model_cfg.get('FUSION_CONFIG', {})
        self.ablation_config = self.model_cfg.get('ABLATION_CONFIG', {})

        self.use_sigmoid_alpha = self.fusion_config.get('USE_SIGMOID_ALPHA', True)
        initial_alpha = self.fusion_config.get('INITIAL_ALPHA', 0.1)
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))

        self._init_multimodal_fusion()

    def _init_multimodal_fusion(self):
        """初始化多模态融合组件"""
        try:
            if self.model_cfg.get('ENABLE_MULTIMODAL_FUSION', False):
                print("初始化多模态融合组件...")

                yolo_config = self.model_cfg.get('YOLO_CONFIG', {})
                # print('1', yolo_config.get('MODEL_PATH'), yolo_config.get('PRETRAINED_WEIGHTS'))
                self.yolo_extractor = YOLOv8FeatureExtractor(
                    model_path_or_config=yolo_config.get('MODEL_PATH', None),
                    pretrained_weights=yolo_config.get('PRETRAINED_WEIGHTS', None),
                    freeze_weights=yolo_config.get('FREEZE_WEIGHTS', False),
                    device='cuda'
                )

                fusion_config = self.model_cfg.get('FUSION_CONFIG', {})
                self.fusion_module = MultiModalFusionForPointPillars(
                    num_heads=fusion_config.get('NUM_HEADS', 8),
                    dropout=fusion_config.get('DROPOUT', 0.1),
                    unified_dim=128,
                    use_interpolation=True,
                    ablation_config=self.ablation_config
                )
                self.use_multimodal_fusion = True

                print(f"YOLO参数数量: {sum(p.numel() for p in self.yolo_extractor.parameters())}")
                print(f"YOLO可训练参数: {sum(p.numel() for p in self.yolo_extractor.parameters() if p.requires_grad)}")
                print("多模态融合组件初始化成功")
            else:
                self.use_multimodal_fusion = False
                print("未启用多模态融合")
        except Exception as e:
            print(f"多模态融合组件初始化失败: {e}")
            self.use_multimodal_fusion = False

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            module_name = type(cur_module).__name__

            if module_name in ['AnchorHeadSingle', 'CenterHead']:
                if self.use_multimodal_fusion and 'images' in batch_dict:
                    print("执行多模态融合...")
                    batch_dict = self._apply_multimodal_fusion(batch_dict)

            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def _apply_multimodal_fusion(self, batch_dict):
        """应用多模态融合"""
        try:
            pp_features = self._extract_pointpillar_multiscale_features(batch_dict)

            images = batch_dict['images']
            device = next(self.parameters()).device

            if images.device != device:
                images = images.to(device)
                batch_dict['images'] = images

            if self.training:
                self.yolo_extractor.train()
            else:
                self.yolo_extractor.eval()

            yolo_features = self.yolo_extractor.extract_multiscale_features(images)
            # 🔥 新增：如果启用随机噪声，替换YOLO特征
            if self.model_cfg.get('USE_RANDOM_NOISE', False):
                yolo_features = generate_random_noise_features(yolo_features)

            fused_features, enhanced_pp_yy_features = self.fusion_module(yolo_features, pp_features)

            self.last_fused_features = fused_features
            self.last_enhanced_features = enhanced_pp_yy_features

            # if fused_features:
            #     upsampled_fusion = torch.cat(fused_features, dim=1)
            #     batch_dict['spatial_features_2d'] = upsampled_fusion
            #     print(f"融合后特征: {upsampled_fusion.shape}")
            if fused_features:
                upsampled_fusion = torch.cat(fused_features, dim=1)
                original = batch_dict['spatial_features_2d']
                # batch_dict['spatial_features_2d'] = 0.5 * original + 0.5 * upsampled_fusion
                # batch_dict['spatial_features_2d'] = 0.7 * original + 0.3 * upsampled_fusion
                current_alpha = self.alpha
                print('current_alpha: ', current_alpha)
                # # 🔥 添加logger记录（会写入log文件）
                # if hasattr(self, 'logger') and self.logger is not None:
                #     self.logger.info(f'current_alpha: {current_alpha}')
                # else:
                #     print('no logger')
                batch_dict['spatial_features_2d'] = current_alpha * upsampled_fusion + (
                                                1 - current_alpha) * original
                print(f"融合后特征: {upsampled_fusion.shape}")

            return batch_dict
        except Exception as e:
            print(f"融合失败: {e}")
            import traceback
            traceback.print_exc()
            return batch_dict


    def _extract_pointpillar_multiscale_features(self, batch_dict):
        """提取PointPillars真正的多尺度特征"""
        spatial_features = batch_dict['spatial_features']
        multi_scale_features = []

        x = spatial_features
        for i, block in enumerate(self.backbone_2d.blocks):
            x = block(x)

            # 每个block的上采样输出 = 不同感受野的128维特征
            if i < len(self.backbone_2d.deblocks):
                multi_scale_features.append(self.backbone_2d.deblocks[i](x).clone())

        return multi_scale_features

    def get_training_loss(self):
        """计算训练损失"""
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()

        if self.use_multimodal_fusion:
            fusion_loss = self._compute_fusion_loss()
            fusion_weight = self.model_cfg.get('FUSION_LOSS_WEIGHT', 0.1)

            tb_dict['alpha'] = self.alpha.item()  # 🔥 会自动记录

            tb_dict.update({
                'loss_rpn': loss_rpn.item(),
                'loss_fusion': fusion_loss.item() if isinstance(fusion_loss, torch.Tensor) else fusion_loss
            })

            total_loss = loss_rpn + fusion_weight * fusion_loss
        else:
            tb_dict['loss_rpn'] = loss_rpn.item()
            total_loss = loss_rpn

        return total_loss, tb_dict, disp_dict

    def _compute_fusion_loss(self):
        """计算融合损失：语义对齐"""
        if not hasattr(self, 'last_enhanced_features'):
            return torch.tensor(0.0, device=next(self.parameters()).device)

        enhanced_yolo, enhanced_pp = self.last_enhanced_features[-1]

        y_global = F.adaptive_avg_pool2d(enhanced_yolo, (1, 1)).flatten(1)
        p_global = F.adaptive_avg_pool2d(enhanced_pp, (1, 1)).flatten(1)

        semantic_sim = F.cosine_similarity(y_global, p_global, dim=1).mean()
        consistency_loss = (1.0 - semantic_sim)

        print(f'语义相似度: {semantic_sim:.4f}, 对齐损失: {consistency_loss:.4f}')

        return consistency_loss
# from .detector3d_template import Detector3DTemplate
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import sys
# sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
# from fusion.wrappers.yolo_extractor import YOLOv8FeatureExtractor
# from fusion.attention.fusion_module import MultiModalFusionForPointPillars
# from fusion.heads.yolo_official_detection_head import YOLOv8OfficialDetectionHead
#
#
# class PointPillar(Detector3DTemplate):
#     def __init__(self, model_cfg, num_class, dataset):
#         super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
#         self.module_list = self.build_networks()
#
#         # device = next(self.parameters()).device
#         # print(f"PointPillar模型设备1: {device}")
#         self.fusion_config = self.model_cfg.get('FUSION_CONFIG', {})
#
#         # 🔥 获取消融实验配置
#         self.ablation_config = self.model_cfg.get('ABLATION_CONFIG', {})
#         # print(f"PointPillar消融配置: {self.ablation_config}")
#
#         # 可选：使用Sigmoid确保alpha在[0,1]范围内
#         self.use_sigmoid_alpha = self.fusion_config.get('USE_SIGMOID_ALPHA', True)
#
#         # print('hhhhh',self.fusion_config)
#         # 初始化可学习的alpha参数
#         initial_alpha = self.fusion_config.get('INITIAL_ALPHA', 0.1)  # 从配置中读取初始值
#         # print('kkkkkkk',initial_alpha)
#         self.alpha = nn.Parameter(torch.tensor(initial_alpha))
#
#         # 初始化多模态融合组件
#         self._init_multimodal_fusion()
#         # device = next(self.parameters()).device
#         # print(f"PointPillar模型设备2: {device}")
#
#     def _init_multimodal_fusion(self):
#         """初始化多模态融合组件"""
#         try:
#             print('enable_multimodal_fusion',self.model_cfg,self.model_cfg.get('ENABLE_MULTIMODAL_FUSION'))
#             # 检查是否启用多模态融合
#             if self.model_cfg.get('ENABLE_MULTIMODAL_FUSION', False):
#                 print("初始化多模态融合组件...")
#
#                 # 初始化YOLOv8特征提取器
#                 yolo_config = self.model_cfg.get('YOLO_CONFIG', {})
#                 print('yolo_config: ',yolo_config)
#                 # cpu
#                 # print('device: ',next(self.parameters()).device)
#                 # create_yolo_extractor
#                 self.yolo_extractor = YOLOv8FeatureExtractor(
#                     model_path_or_config=yolo_config.get('MODEL_PATH', None),
#                     device='cuda'
#                 )
#                 img_size = (128, 1024)
#
#                 # # 获取特征通道信息
#                 # pp_feature_info = self._get_pointpillar_feature_info()
#                 # yolo_feature_info = self.yolo_extractor.get_feature_info()
#
#                 # 初始化融合模块
#                 fusion_config = self.model_cfg.get('FUSION_CONFIG', {})
#
#                 # 确定目标通道数
#                 unified_dim = 128
#
#                 # self.fusion_module = MultiModalFusionForPointPillars(
#                 #     num_heads=fusion_config.get('num_heads', 8),
#                 #     # num_layers=fusion_config.get('num_layers', 2),
#                 #     dropout=fusion_config.get('dropout', 0.1),
#                 #     unified_dim=unified_dim
#                 # )
#                 self.fusion_module = MultiModalFusionForPointPillars(
#                     num_heads=fusion_config.get('NUM_HEADS', 8),
#                     # num_layers=fusion_config.get('num_layers', 2),
#                     dropout=fusion_config.get('DROPOUT', 0.1),
#                     unified_dim=unified_dim,
#                     use_interpolation=True,
#                     ablation_config=self.ablation_config  # 🔥 传递消融配置
#                 )
#                 self.use_multimodal_fusion = True
#
#                 # 🔥获取图像特征的输出通道数
#                 yolo_feature_channels = [128, 256, 512]  # P3, P4, P5
#                 print('class_name', self.class_names)
#
#                 self.yolo_detection_head = YOLOv8OfficialDetectionHead(
#                     nc=len(self.class_names),  # 类别数
#                     ch=tuple(yolo_feature_channels)
#                 )
#
#                 # 确保YOLO参数被正确注册到模型中会被优化器捕获
#                 # self.yolo_extractor = self.yolo_extractor
#                 # # 如果YOLO是独立模块，需要显式注册
#                 # 显式将yolo_extractor添加为子模块
#                 # self.add_module('yolo_feature_extractor', self.yolo_extractor)
#                 # print('如果YOLO是独立模块，需要显式注册')
#                 print(f"YOLO参数数量: {sum(p.numel() for p in self.yolo_extractor.parameters())}")
#                 # 🔥 打印当前模型的所有参数
#                 # self.print_model_parameters()
#                 print("多模态融合组件初始化成功")
#
#             else:
#                 self.use_multimodal_fusion = False
#                 print("未启用多模态融合，使用原生PointPillars")
#
#         except Exception as e:
#             print(f"多模态融合组件初始化失败: {e}")
#             print("回退到原生PointPillars模式")
#             self.use_multimodal_fusion = False
#
#     def forward(self, batch_dict):
#         # print(f"PointPillar.forward()执行 - epoch: {batch_dict.get('epoch', '未知')}")
#         for cur_module in self.module_list:
#             module_name = type(cur_module).__name__
#             # print('module_name',module_name)
#
#             # 如果是DENSE_HEAD，先融合再执行
#             if module_name in ['AnchorHeadSingle', 'CenterHead']:
#                 if self.use_multimodal_fusion and 'images' in batch_dict:
#                     print("在DENSE_HEAD之前执行融合...")
#                     batch_dict = self._apply_multimodal_fusion(batch_dict)
#                     print("融合完成，DENSE_HEAD将使用融合特征")
#
#             # 执行模块
#             batch_dict = cur_module(batch_dict)
#
#         # # 多模态融合（如果启用）
#         # if self.use_multimodal_fusion and 'images' in batch_dict:
#         #     batch_dict = self._apply_multimodal_fusion(batch_dict)
#
#         if self.training:
#             loss, tb_dict, disp_dict = self.get_training_loss()
#
#             ret_dict = {
#                 'loss': loss
#             }
#             return ret_dict, tb_dict, disp_dict
#         else:
#             pred_dicts, recall_dicts = self.post_processing(batch_dict)
#             return pred_dicts, recall_dicts
#
#     def _apply_multimodal_fusion(self, batch_dict):
#         """应用多模态融合"""
#         try:
#             # 1. 提取PointPillars多尺度特征
#             pp_features = self._extract_pointpillar_multiscale_features(batch_dict)
#
#             # for i in range(len(pp_features)):
#             #     print('pp_features: ',pp_features[i].shape, 'device:', pp_features[i].device)
#             # 2. 提取图像特征
#             images = batch_dict['images']
#             # print('images: ',images.shape, 'device:', images.device)
#             #
#             # print("\n=== 调试设备信息 ===")
#             device = next(self.parameters()).device
#             # print(f"PointPillar模型设备: {device}")
#             # for key, value in batch_dict.items():
#             #     if isinstance(value, torch.Tensor):
#             #         print(f"batch_dict[{key}]: {value.shape}, 设备: {value.device}")
#
#             # 确保图像在正确的设备上
#             if images.device != device:
#                 print(f"移动图像从 {images.device} 到 {device}")
#                 images = images.to(device)
#                 batch_dict['images'] = images  # 更新batch_dict中的图像
#
#             # 关键：确保YOLO在正确的模式下运行
#             if self.training:
#                 self.yolo_extractor.train()
#             else:
#                 self.yolo_extractor.eval()
#
#             yolo_features = self.yolo_extractor.extract_multiscale_features(images)
#
#             # 🔥 保存YOLO特征用于辅助损失计算
#             self.last_yolo_features = yolo_features
#
#             # 🔥 保存图像标签（如果存在）
#             if 'image_labels' in batch_dict:
#                 self.image_labels = batch_dict['image_labels']
#             # for i in range(len(yolo_features)):
#             #     print('yolo_features: ', yolo_features[i].shape)
#
#             # 3. 多模态融合
#             # fused_features = self.fusion_module(yolo_features, pp_features)
#
#             fused_features, enhanced_pp_yy_features = self.fusion_module(yolo_features, pp_features)
#
#             # 🔥 保存融合特征
#             self.last_fused_features = fused_features
#
#             # 保存特征用于损失计算
#             self.last_enhanced_features = enhanced_pp_yy_features
#             # self.last_yolo_features = yolo_features
#             # self.last_pp_features = pp_features
#
#             # 4. 替换原始的spatial_features_2d
#             if fused_features:
#                 # batch_dict['spatial_features_2d'] = torch.cat(fused_features, dim=1)  # 使用最后一层融合特征
#                 upsampled_fusion = torch.cat(fused_features, dim=1)
#                 original_features = batch_dict['spatial_features_2d']
#                 # batch_dict['spatial_features_2d'] = torch.cat(fused_features, dim=1)  # 使用最后一层融合特征
#                 # # 简单残差连接（固定权重）
#                 # alpha = 1.0  # 可调超参数
#                 # alpha = 0.3  # 可调超参数
#                 # self.diagnose_alpha_parameter()
#                 # self.print_model_parameters()
#                 # 获取当前的alpha值
#                 if self.use_sigmoid_alpha:
#                     print('self.alpha: ',self.alpha)
#                     current_alpha = torch.sigmoid(self.alpha)  # 确保在[0,1]范围
#                     print('current_alpha: ', current_alpha)
#                 else:
#                     # current_alpha = torch.clamp(self.alpha, 0.0, 1.0)  # 硬截断到[0,1]
#                     current_alpha = self.alpha
#                     print('no sigmoid current_alpha: ', current_alpha)
#
#                 # 应用可学习的融合权重
#                 batch_dict['spatial_features_2d'] = current_alpha * upsampled_fusion + (
#                             1 - current_alpha) * original_features
#                 # batch_dict['spatial_features_2d'] = upsampled_fusion
#                 # batch_dict['spatial_features_2d'] = alpha * upsampled_fusion + (1 - alpha) * original_features
#                 # print('alpha: ',alpha)
#                 # batch_dict['spatial_features_2d'] = upsampled_fusion + original_features
#                 # # # batch_dict['spatial_features_2d'] = torch.cat(fused_features, dim=1) + batch_dict['spatial_features_2d']
#                 # print('batch_dict[spatial_features_2d]: ', batch_dict['spatial_features_2d'].shape)
#
#                 # # 验证是否真的改变了
#                 # fused_mean = batch_dict['spatial_features_2d'].mean().item()
#                 # print(f"融合后特征均值: {fused_mean:.6f}")
#                 # print(f"特征变化: {abs(fused_mean - original_mean):.6f}")
#                 print(f"相似度: {F.cosine_similarity(original_features.flatten(1), batch_dict['spatial_features_2d'].flatten(1)).mean():.4f}")
#
#                 # 可选：保存中间特征用于分析
#                 batch_dict['fused_multiscale_features'] = fused_features
#                 batch_dict['original_pp_features'] = pp_features
#                 batch_dict['yolo_features'] = yolo_features
#
#         except Exception as e:
#             print(f"多模态融合失败: {e}")
#             print("使用原始PointPillars特征")
#             # 如果融合失败，继续使用原始特征
#
#         return batch_dict
#
#     def _extract_pointpillar_multiscale_features(self, batch_dict):
#         """提取PointPillars多尺度特征"""
#         # 从backbone_2d中提取中间特征
#         spatial_features = batch_dict['spatial_features']
#         multi_scale_features = []
#         ups = []
#
#         # 通过backbone的每个block提取特征
#         x = spatial_features
#         for i, block in enumerate(self.backbone_2d.blocks):
#             x = block(x)
#
#             # 如果有上采样层，应用它们
#             if len(self.backbone_2d.deblocks) > 0 and i < len(self.backbone_2d.deblocks):
#                 ups.append(self.backbone_2d.deblocks[i](x))
#                 # print('upsample',self.backbone_2d.deblocks[i](x).shape)
#                 multi_scale_features.append(self.backbone_2d.deblocks[i](x).clone())
#             else:
#                 ups.append(x)
#
#         return multi_scale_features
#
#     def get_training_loss(self):
#         disp_dict = {}
#
#         loss_rpn, tb_dict = self.dense_head.get_loss()
#
#         # 添加融合损失
#         # fusion_loss = torch.tensor(0.0, device=loss_rpn.device, requires_grad=True)
#         # if self.use_multimodal_fusion and hasattr(self, 'last_enhanced_features'):
#         #     fusion_loss = self._compute_fusion_loss()
#         if self.use_multimodal_fusion:
#             # 融合损失（原有的语义一致性损失）
#             fusion_loss = self._compute_fusion_loss()
#             # 融合损失权重
#             fusion_weight = self.model_cfg.get('FUSION_LOSS_WEIGHT', 0.1)
#
#             # 🔥 新增：辅助损失（YOLO检测损失）
#             auxiliary_loss = torch.tensor(0.0, device=loss_rpn.device)
#             if hasattr(self, 'yolo_detection_head') and 'image_labels' in self.__dict__:
#                 auxiliary_loss = self._compute_auxiliary_loss()
#
#             # 辅助损失权重
#             auxiliary_weight = self.model_cfg.get('AUXILIARY_LOSS_WEIGHT', 0.2)
#             # fusion_loss = fusion_weight * fusion_loss
#             tb_dict = {
#                 'loss_rpn': loss_rpn.item(),
#                 'loss_fusion': fusion_loss.item() if isinstance(fusion_loss, torch.Tensor) else fusion_loss,
#                 'loss_auxiliary': auxiliary_loss.item() if isinstance(auxiliary_loss, torch.Tensor) else auxiliary_loss,
#                 **tb_dict
#             }
#
#             # loss = loss_rpn
#             # 总损失
#             total_loss = loss_rpn + fusion_weight * fusion_loss + auxiliary_weight * auxiliary_loss
#
#             # 打印YOLO梯度norm
#             if hasattr(self, 'yolo_extractor') and self.yolo_extractor.parameters().__iter__().__next__().grad is not None:
#                 total_norm = 0.0
#                 for p in self.yolo_extractor.parameters():
#                     if p.grad is not None:
#                         total_norm += p.grad.data.norm(2).item() ** 2
#                 total_norm = total_norm ** 0.5
#                 print(f"YOLO梯度norm: {total_norm:.6f}")
#         else:
#             tb_dict = {
#                 'loss_rpn': loss_rpn.item(),
#                 **tb_dict
#             }
#
#             # loss = loss_rpn
#             # 总损失
#             total_loss = loss_rpn
#
#         return total_loss, tb_dict, disp_dict
#
#     def _compute_auxiliary_loss(self):
#         """计算辅助损失：YOLO检测头损失"""
#         if not hasattr(self, 'last_yolo_features') or not hasattr(self, 'image_labels'):
#             return torch.tensor(0.0, device=next(self.parameters()).device)
#
#         # 使用YOLO检测头计算损失
#         yolo_predictions = self.yolo_detection_head(self.last_yolo_features)
#         auxiliary_loss = self.yolo_detection_head.compute_loss(yolo_predictions, self.image_labels)
#
#         print(f'辅助损失(YOLO检测): {auxiliary_loss.item():.4f}')
#         return auxiliary_loss
#
#     def _compute_fusion_loss(self):
#         """跨模态语义对齐,全局语义对齐（类似CLIP）语义一致性损失 - 使用交叉注意力后的增强特征"""
#         """确保两个模态提取相似的场景语义类似CLIP的对比学习效果"""
#         if not hasattr(self, 'last_enhanced_features'):
#             print('no_fusion_loss语义一致性损失')
#             return torch.tensor(0.0, device=next(self.parameters()).device)
#
#         enhanced_yolo, enhanced_pp = self.last_enhanced_features[-1]
#
#         # 全局语义向量
#         y_global = F.adaptive_avg_pool2d(enhanced_yolo, (1, 1)).flatten(1)
#         p_global = F.adaptive_avg_pool2d(enhanced_pp, (1, 1)).flatten(1)
#
#         # 语义一致性
#         semantic_sim = F.cosine_similarity(y_global, p_global, dim=1).mean()
#         consistency_loss = (1.0 - semantic_sim)
#         print('semantic_sim: ', semantic_sim,'consistency_loss: ',consistency_loss)
#
#         return consistency_loss
#
#
#     def diagnose_alpha_parameter(self):
#         """诊断alpha参数的状态"""
#         if not hasattr(self, 'alpha'):
#             print("❌ Alpha参数不存在")
#             return
#
#         print("\n=== Alpha参数完整诊断 ===")
#         print(f"Alpha原始值: {self.alpha.item():.6f}")
#         print(f"Alpha设备: {self.alpha.device}")
#         print(f"Alpha requires_grad: {self.alpha.requires_grad}")
#         print(f"Alpha is_leaf: {self.alpha.is_leaf}")
#         print(f"Alpha梯度: {self.alpha.grad.item() if self.alpha.grad is not None else 'None'}")
#
#         # 检查alpha是否在模型参数中
#         alpha_in_params = any(p is self.alpha for p in self.parameters())
#         print(f"Alpha在模型参数中: {alpha_in_params}")
#
#         # 检查alpha是否在named_parameters中
#         alpha_in_named = any(name == 'alpha' for name, p in self.named_parameters())
#         print(f"Alpha在named_parameters中: {alpha_in_named}")
#
#         # 检查计算图
#         if self.alpha.grad_fn is not None:
#             print(f"Alpha梯度函数: {self.alpha.grad_fn}")
#         else:
#             print("Alpha梯度函数: None (这是正常的，因为它是叶子节点)")
#
#         print("========================\n")
#
#     def check_optimizer_includes_alpha(self, optimizer):
#         """检查优化器是否包含alpha参数"""
#         alpha_included = False
#         for param_group in optimizer.param_groups:
#             for param in param_group['params']:
#                 if param is self.alpha:
#                     alpha_included = True
#                     print(f"✅ Alpha参数已包含在优化器中，学习率: {param_group['lr']}")
#                     break
#
#         if not alpha_included:
#             print("❌ Alpha参数未包含在优化器中！这是问题所在！")
#             print("解决方案：")
#             print("1. 重新创建优化器: optimizer = torch.optim.Adam(model.parameters(), lr=lr)")
#             print("2. 或手动添加: optimizer.add_param_group({'params': [model.alpha]})")
#
#         return alpha_included
#
#     def print_model_parameters(self, show_details=True):
#         """打印模型的所有参数"""
#         print("\n" + "=" * 60)
#         print("📋 模型参数清单")
#         print("=" * 60)
#
#         total_params = 0
#         trainable_params = 0
#
#         # 按模块分组显示参数
#         if show_details:
#             print("\n🔍 详细参数列表:")
#             print("-" * 60)
#             for name, param in self.named_parameters():
#                 param_count = param.numel()
#                 total_params += param_count
#                 if param.requires_grad:
#                     trainable_params += param_count
#
#                 grad_status = "✅" if param.requires_grad else "❌"
#                 shape_str = str(list(param.shape))
#
#                 print(f"{grad_status} {name:<40} {shape_str:<20} {param_count:>10,}")
#
#                 # 特别标注alpha参数
#                 if 'alpha' in name.lower():
#                     print(f"    🎯 当前值: {param.item():.6f}")
#
#         # 统计信息
#         print("-" * 60)
#         print(f"📊 参数统计:")
#         print(f"   总参数数量: {total_params:,}")
#         print(f"   可训练参数: {trainable_params:,}")
#         print(f"   冻结参数: {total_params - trainable_params:,}")
#
#         # 检查关键参数
#         print(f"\n🔍 关键参数检查:")
#         has_alpha = any('alpha' in name for name, _ in self.named_parameters())
#         has_yolo = any('yolo' in name.lower() for name, _ in self.named_parameters())
#         has_fusion = any('fusion' in name.lower() for name, _ in self.named_parameters())
#
#         print(f"   Alpha参数: {'✅ 存在' if has_alpha else '❌ 不存在'}")
#         print(f"   YOLO参数: {'✅ 存在' if has_yolo else '❌ 不存在'}")
#         print(f"   融合模块参数: {'✅ 存在' if has_fusion else '❌ 不存在'}")
#
#         # 如果有alpha，显示其详细信息
#         if hasattr(self, 'alpha'):
#             print(f"\n🎯 Alpha参数详情:")
#             print(f"   值: {self.alpha.item():.6f}")
#             print(f"   设备: {self.alpha.device}")
#             print(f"   requires_grad: {self.alpha.requires_grad}")
#             print(f"   形状: {self.alpha.shape}")
#
#         print("=" * 60 + "\n")
#
#     def print_optimizer_parameters(self, optimizer):
#         """打印优化器包含的参数"""
#         print("\n" + "=" * 60)
#         print("🚀 优化器参数清单")
#         print("=" * 60)
#
#         # 建立参数到名称的映射
#         param_to_name = {param: name for name, param in self.named_parameters()}
#
#         for group_idx, param_group in enumerate(optimizer.param_groups):
#             print(f"\n📦 参数组 {group_idx + 1}:")
#             print(f"   学习率: {param_group['lr']}")
#             print(f"   参数数量: {len(param_group['params'])}")
#
#             print("   包含的参数:")
#             for param in param_group['params']:
#                 param_name = param_to_name.get(param, "未知参数")
#                 param_count = param.numel()
#
#                 # 特别标注alpha参数
#                 if param is getattr(self, 'alpha', None):
#                     print(f"   🎯 {param_name:<40} ({param_count:,} 参数) <- ALPHA!")
#                 else:
#                     print(f"   📌 {param_name:<40} ({param_count:,} 参数)")
#
#         # 检查alpha是否在优化器中
#         alpha_in_optimizer = False
#         if hasattr(self, 'alpha'):
#             for param_group in optimizer.param_groups:
#                 if self.alpha in param_group['params']:
#                     alpha_in_optimizer = True
#                     break
#
#         print(f"\n🔍 Alpha参数检查: {'✅ 已包含' if alpha_in_optimizer else '❌ 未包含'}")
#         print("=" * 60 + "\n")