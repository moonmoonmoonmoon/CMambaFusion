from ultralytics.nn.modules.conv import autopad
import torch.nn as nn
import torch

class LidarInputLayer(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # 添加YOLOv8所需的属性
        self.f = -1  # 表示从前一层获取输入
        self.i = 0  # 通常是第一层
        self.type = 'LidarInputLayer'  # 类型名称

        # 如果需要可学习的权重，可以取消下面的注释

        # 添加可学习的通道权重
        self.channel_weights = nn.Parameter(torch.ones(c1))

    def forward(self, x):
        # 如果需要调试，可以取消下面的注释
        # print(f"LidarInputLayer 输入形状: {x.shape}")
        # is_training = self.training
        # print(f"阶段: {'训练' if is_training else '验证'}, 输入形状: {x.shape}")

        # # 应用通道权重
        weighted_x = x * self.channel_weights.view(1, -1, 1, 1)

        # 卷积后打印输出尺寸
        out = self.act(self.bn(self.conv(weighted_x)))

        return out