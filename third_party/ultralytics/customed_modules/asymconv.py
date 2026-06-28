import torch
import  torch.nn as nn

class AsymmetricDownsample(nn.Module):
    """非对称下采样，高度和宽度方向使用不同的策略"""

    def __init__(self, c1, c2, k=3, s=[1,2]):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s,
                              padding=(k // 2, k // 2))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))