# import torch
# import torch.nn as nn
# from ultralytics.nn.modules.conv import autopad
# class LidarInputLayerAsymDil(nn.Module):
#     default_act = nn.SiLU()  # default activation
#
#     def __init__(self, c1, c2, k=3, s=2, d=[1, 4]):
#         super().__init__()
#         self.c2 = c2
#         self.k = k
#         self.s = s
#         self.d = d
#         # 计算非对称padding
#         p_h = ((d[0] * (k - 1)) + 1) // 2
#         p_w = ((d[1] * (k - 1)) + 1) // 2
#
#         self.conv = nn.Conv2d(3, c2, k, s, padding=(p_h, p_w), dilation=(d[0], d[1]), bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU()
#         # 可学习的通道权重
#         self.channel_weights = nn.Parameter(torch.ones(3))
#
#     def forward(self, x):
#         # 应用通道权重
#         weighted_x = x * self.channel_weights.view(1, -1, 1, 1)
#         return self.act(self.bn(self.conv(weighted_x)))

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import autopad
class LidarInputLayerAsymDil(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=[2,2], d=[1, 4]):
        super().__init__()
        self.c2 = c2
        self.k = k
        self.s = s
        self.d = d
        # 计算非对称padding
        p_h = ((d[0] * (k - 1)) + 1) // 2
        p_w = ((d[1] * (k - 1)) + 1) // 2

        self.conv = nn.Conv2d(3, c2, k, s, padding=(p_h, p_w), dilation=(d[0], d[1]), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        # 可学习的通道权重
        self.channel_weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # 应用通道权重
        weighted_x = x * self.channel_weights.view(1, -1, 1, 1)
        return self.act(self.bn(self.conv(weighted_x)))