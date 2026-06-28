"""
YOLOv8特征提取器 - 支持预训练权重加载
位置: third_party/fusion/wrappers/yolo_extractor.py
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../../ultralytics'))

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("警告: Ultralytics不可用")
    ULTRALYTICS_AVAILABLE = False


class YOLOv8FeatureExtractor(nn.Module):
    """YOLOv8特征提取器，支持预训练权重加载"""

    def __init__(self, model_path_or_config=None, pretrained_weights=None,
                 freeze_weights=False, device='cuda', extract_layers=None):
        """
        Args:
            model_path_or_config: YOLO模型配置文件路径
            pretrained_weights: 预训练权重路径（.pt文件）
            freeze_weights: 是否冻结权重
            device: 设备
            extract_layers: 要提取特征的层索引
        """
        super().__init__()

        # torch.manual_seed(42)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(42)

        self.device = device
        self.features = {}
        self.extract_layers = extract_layers or [15, 18, 21]
        self.freeze_weights = freeze_weights

        if ULTRALYTICS_AVAILABLE:
            self._load_yolo_model(model_path_or_config, pretrained_weights)
        else:
            raise ImportError("需要ultralytics库")

        print(f"YOLOv8特征提取器初始化完成")
        print(f"提取层: {self.extract_layers}")
        print(f"权重冻结: {freeze_weights}")

    def _load_yolo_model(self, model_path_or_config, pretrained_weights):
        """加载YOLO模型"""
        try:
            if pretrained_weights and os.path.exists(pretrained_weights):
                print(f"从预训练权重加载: {pretrained_weights}")
                yolo = YOLO(pretrained_weights)
            else:
                if model_path_or_config is None:
                    model_path_or_config = '/home/yanan/Downloads/projects/multimodal_detection/config/customed_yolov8s.yaml'
                print(f"从配置加载YOLO: {model_path_or_config}")
                yolo = YOLO(model_path_or_config)

            self.model = yolo.model.to(self.device)

            if self.freeze_weights:
                print("冻结YOLO权重...")
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()

            self._register_hooks()

        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            raise

    def _register_hooks(self):
        """注册特征提取钩子"""

        def get_activation(name):
            def hook(module, input, output):
                self.features[name] = output

            return hook

        model_layers = self.model.model if hasattr(self.model, 'model') else self.model
        layer_names = ['P3', 'P4', 'P5']

        for i, layer_idx in enumerate(self.extract_layers):
            if layer_idx < len(model_layers):
                layer_name = layer_names[i] if i < len(layer_names) else f'layer_{layer_idx}'
                model_layers[layer_idx].register_forward_hook(get_activation(layer_name))
                print(f"注册钩子: {layer_name} (层{layer_idx})")

    def forward(self, images):
        """前向传播提取特征"""
        self.features.clear()

        # 🔥 打印输入尺寸
        print(f"输入YOLOv8的图像尺寸: {images.shape}")

        if self.freeze_weights:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(images)
        else:
            if self.training:
                self.model.train()
            else:
                self.model.eval()
            _ = self.model(images)

        multi_scale_features = []
        for scale_name in ['P3', 'P4', 'P5']:
            if scale_name in self.features:
                feat = self.features[scale_name]
                print(f"{scale_name} 特征尺寸: {feat.shape}")
                multi_scale_features.append(self.features[scale_name])

        return multi_scale_features

    def extract_multiscale_features(self, images):
        """提取多尺度特征"""
        return self.forward(images)