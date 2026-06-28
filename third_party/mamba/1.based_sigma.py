import torch
import torch.nn as nn
import torch.nn.functional as F
from vmamba import CrossMambaFusionBlock, ConcatMambaFusionBlock





class SigmaFusion(nn.Module):
    def __init__(self, embed_dim=768):

        super().__init__()
        """ sigma blocks   """
        self.embed_dim = embed_dim
        self.cross_mamba = CrossMambaFusionBlock(
                hidden_dim=self.embed_dim,
                mlp_ratio=0.0,
                d_state=4,
            )

        self.channel_attn_mamba = ConcatMambaFusionBlock(
                hidden_dim=self.embed_dim,
                mlp_ratio=0.0,
                d_state=4,
            )

        self.sigma_norm = norm_layer(embed_dim)


    def forward(self, modal_rgb, modal_lidar):
        ## 大部分都有提升
        x, x_modal = self.cross_mamba(modal_rgb, modal_lidar) ## 1. 要有提升
        x_fuse = self.channel_attn_mamba(x, x_modal) ## 2. 要有提升
        x = x + x_fuse + x_modal
        return x
