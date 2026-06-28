import torch
import torch.nn as nn
from ..nn.modules.conv import Conv, CBAM
from ..nn.modules.block import Bottleneck

# 在C2f类中添加CBAM
class C2f_CBAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.bottlenecks = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
        self.cbam = CBAM(c2)  # 添加CBAM

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        out = self.cv2(torch.cat(y, 1))
        return self.cbam(out)  # 应用CBAM