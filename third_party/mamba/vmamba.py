import os
import time
import math
import copy
from datetime import datetime
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    pass

# cross selective scan ===============================
if True:
    import selective_scan_cuda_core as selective_scan_cuda


    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)


    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y

        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None


    class CrossScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:H * W].view(B, -1, H, W), ys[:, :, H * W:2 * H * W].view(B, -1, H, W)


    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            return ys[:, :, 0:L // 2], ys[:, :, L // 2:L]

        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape    out: (b, k, d, l)
            # H, W = ctx.shape
            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2 * L))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2 * L)
            return xs, None, None


    def cross_selective_scan(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)

        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def selective_scan_1d(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape  # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)

        y = CrossMerge.apply(ys)  # todo this line has changed

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def cross_selective_scan_multimodal_k1(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2 * H * W)

        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L // 2] + ys[:, 0, :, L // 2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)

        return y


    def cross_selective_scan_multimodal_k2(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm1: torch.nn.Module = None,
            out_norm2: torch.nn.Module = None,  # softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e)  # B, C, H, W -> B, 2, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2 * H * W)

        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)

        return y_rgb, y_e

    # share A/D
    def cross_selective_scan_4dir(
            x_rgb: torch.Tensor = None,  # (B, D, H, W) — 已经过 conv2d 处理
            x_e: torch.Tensor = None,  # (B, D, H, W)
            # rgb 和 e 分别有独立的 x_proj，用于生成各自的 B、C、delta
            x_proj_weight_rgb: torch.Tensor = None,  # (4, dt_rank+2*N, d_inner)
            x_proj_weight_e: torch.Tensor = None,  # (4, dt_rank+2*N, d_inner)
            # dt_proj、A、D 两模态共享（类似 Sigma 的 Siamese 权重共享）
            dt_projs_weight: torch.Tensor = None,  # (4, d_inner, dt_rank)
            dt_projs_bias: torch.Tensor = None,  # (4, d_inner)
            A_logs: torch.Tensor = None,  # (4*d_inner, N)
            Ds: torch.Tensor = None,  # (4*d_inner,)
            out_norm_rgb: torch.nn.Module = None,
            out_norm_e: torch.nn.Module = None,
            nrows: int = -1,
            delta_softplus: bool = True,
    ):
        """
        真正的 4 方向 2D Cross Mamba 扫描。

        核心思想（来自 Sigma CroMB）：
          - 每个模态各自在 4 个方向做 SS2D 扫描，得到独立的隐状态 h_rgb / h_e
          - 解码时交换 C 矩阵：rgb 用 C_e 解码，e 用 C_rgb 解码
          - 这样两模态互相引导，实现跨模态信息增强

        与原来 cross_selective_scan_multimodal_k2（K=2 双向）的区别：
          - 原来：把 rgb+e 拼成 2*H*W 序列，只有行主序 正/反 两个方向
          - 现在：每个模态独立做 4 方向扫（水平正/反 + 垂直转置正/反），
                  空间结构保留更完整，BEV 特征的 2D 上下文建模更充分
        """
        B, D, H, W = x_rgb.shape
        # A_logs: (4*D, N)  → 取 N
        N = A_logs.shape[1] if A_logs.dim() == 2 else A_logs.shape[-1]
        K, D_inner, R = dt_projs_weight.shape  # K=4
        L = H * W

        # 自动选 nrows（和其他 scan 函数保持一致）
        if nrows < 1:
            if D_inner % 4 == 0:
                nrows = 4
            elif D_inner % 3 == 0:
                nrows = 3
            elif D_inner % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # ── Step 1: 4 方向展开 ──────────────────────────────────────
        # CrossScan.apply: (B, D, H, W) → (B, 4, D, H*W)
        # 4 个方向: 行正向、列正向(转置)、行反向、列反向
        xs_rgb = CrossScan.apply(x_rgb)  # (B, 4, D, L)
        xs_e = CrossScan.apply(x_e)  # (B, 4, D, L)

        # ── Step 2: 各自投影得到 dt、B、C ───────────────────────────
        x_dbl_rgb = torch.einsum("b k d l, k c d -> b k c l", xs_rgb, x_proj_weight_rgb)
        x_dbl_e = torch.einsum("b k d l, k c d -> b k c l", xs_e, x_proj_weight_e)

        dts_rgb, Bs_rgb, Cs_rgb = torch.split(x_dbl_rgb, [R, N, N], dim=2)
        dts_e, Bs_e, Cs_e = torch.split(x_dbl_e, [R, N, N], dim=2)

        # dt 投影（共享权重）
        dts_rgb = torch.einsum("b k r l, k d r -> b k d l", dts_rgb, dt_projs_weight)
        dts_e = torch.einsum("b k r l, k d r -> b k d l", dts_e, dt_projs_weight)

        # ── Step 3: 准备共享的 A、D ─────────────────────────────────
        As = -torch.exp(A_logs.float())  # (4*D, N)
        Ds_ = Ds.float()  # (4*D,)
        delta_bias = dt_projs_bias.view(-1).float()  # (4*D,)

        def selective_scan(u, delta, A, B, C, D, db):
            return SelectiveScan.apply(u, delta, A, B, C, D, db, delta_softplus, nrows)

        # ── Step 4: Cross 扫描（关键：C 矩阵互换）──────────────────
        # rgb 的隐状态由 rgb 自身驱动(A_rgb, B_rgb, dt_rgb)，
        # 但解码时用 C_e → rgb 输出受 e 的语义引导
        ys_rgb = selective_scan(
            xs_rgb.view(B, -1, L).float(),
            dts_rgb.contiguous().view(B, -1, L).float(),
            As,
            Bs_rgb.contiguous().float(),
            Cs_e.contiguous().float(),  # ← 用 e 的 C 解码 rgb 隐状态
            Ds_,
            delta_bias,
        ).view(B, K, -1, H, W)

        # e 的隐状态由 e 自身驱动，但解码时用 C_rgb
        ys_e = selective_scan(
            xs_e.view(B, -1, L).float(),
            dts_e.contiguous().view(B, -1, L).float(),
            As,
            Bs_e.contiguous().float(),
            Cs_rgb.contiguous().float(),  # ← 用 rgb 的 C 解码 e 隐状态
            Ds_,
            delta_bias,
        ).view(B, K, -1, H, W)

        # ── Step 5: 合并 4 方向 ─────────────────────────────────────
        # CrossMerge.apply: (B, 4, D, H, W) → (B, D, H*W)
        y_rgb = CrossMerge.apply(ys_rgb)  # (B, D, L)
        y_e = CrossMerge.apply(ys_e)

        # reshape → (B, H, W, D)，然后 LayerNorm
        y_rgb = y_rgb.transpose(1, 2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(1, 2).contiguous().view(B, H, W, -1)

        y_rgb = out_norm_rgb(y_rgb).to(x_rgb.dtype)
        y_e = out_norm_e(y_e).to(x_e.dtype)

        return y_rgb, y_e

    # no share A/D
    # def cross_selective_scan_4dir(
    #         x_rgb, x_e,
    #         x_proj_weight_rgb, x_proj_weight_e,
    #         dt_projs_weight, dt_projs_bias,
    #         A_logs_rgb, A_logs_e,  # ← 拆分
    #         Ds_rgb, Ds_e,  # ← 拆分
    #         out_norm_rgb, out_norm_e,
    #         nrows: int = -1,
    #         delta_softplus: bool = True,
    # ):
    #     """
    #     真正的 4 方向 2D Cross Mamba 扫描。
    #
    #     核心思想（来自 Sigma CroMB）：
    #       - 每个模态各自在 4 个方向做 SS2D 扫描，得到独立的隐状态 h_rgb / h_e
    #       - 解码时交换 C 矩阵：rgb 用 C_e 解码，e 用 C_rgb 解码
    #       - 这样两模态互相引导，实现跨模态信息增强
    #
    #     与原来 cross_selective_scan_multimodal_k2（K=2 双向）的区别：
    #       - 原来：把 rgb+e 拼成 2*H*W 序列，只有行主序 正/反 两个方向
    #       - 现在：每个模态独立做 4 方向扫（水平正/反 + 垂直转置正/反），
    #               空间结构保留更完整，BEV 特征的 2D 上下文建模更充分
    #     """
    #     B, D, H, W = x_rgb.shape
    #     # A_logs: (4*D, N)  → 取 N
    #     # N = A_logs.shape[1] if A_logs.dim() == 2 else A_logs.shape[-1]
    #     N = A_logs_rgb.shape[1] if A_logs_rgb.dim() == 2 else A_logs_rgb.shape[-1]
    #     K, D_inner, R = dt_projs_weight.shape  # K=4
    #     L = H * W
    #
    #     # 自动选 nrows（和其他 scan 函数保持一致）
    #     if nrows < 1:
    #         if D_inner % 4 == 0:
    #             nrows = 4
    #         elif D_inner % 3 == 0:
    #             nrows = 3
    #         elif D_inner % 2 == 0:
    #             nrows = 2
    #         else:
    #             nrows = 1
    #
    #     # ── Step 1: 4 方向展开 ──────────────────────────────────────
    #     # CrossScan.apply: (B, D, H, W) → (B, 4, D, H*W)
    #     # 4 个方向: 行正向、列正向(转置)、行反向、列反向
    #     xs_rgb = CrossScan.apply(x_rgb)  # (B, 4, D, L)
    #     xs_e = CrossScan.apply(x_e)  # (B, 4, D, L)
    #
    #     # ── Step 2: 各自投影得到 dt、B、C ───────────────────────────
    #     x_dbl_rgb = torch.einsum("b k d l, k c d -> b k c l", xs_rgb, x_proj_weight_rgb)
    #     x_dbl_e = torch.einsum("b k d l, k c d -> b k c l", xs_e, x_proj_weight_e)
    #
    #     dts_rgb, Bs_rgb, Cs_rgb = torch.split(x_dbl_rgb, [R, N, N], dim=2)
    #     dts_e, Bs_e, Cs_e = torch.split(x_dbl_e, [R, N, N], dim=2)
    #
    #     # dt 投影（共享权重）
    #     dts_rgb = torch.einsum("b k r l, k d r -> b k d l", dts_rgb, dt_projs_weight)
    #     dts_e = torch.einsum("b k r l, k d r -> b k d l", dts_e, dt_projs_weight)
    #
    #     # ── Step 3: 准备共享的 A、D ─────────────────────────────────
    #     As_rgb = -torch.exp(A_logs_rgb.float())  # (4*D, N)
    #     As_e = -torch.exp(A_logs_e.float())  # (4*D, N)
    #     Ds_rgb_ = Ds_rgb.float()  # (4*D,)
    #     Ds_e_ = Ds_e.float()  # (4*D,)
    #     delta_bias = dt_projs_bias.view(-1).float()
    #
    #     def selective_scan(u, delta, A, B, C, D, db):
    #         return SelectiveScan.apply(u, delta, A, B, C, D, db, delta_softplus, nrows)
    #
    #     # ── Step 4: Cross 扫描（关键：C 矩阵互换）──────────────────
    #     # rgb 的隐状态由 rgb 自身驱动(A_rgb, B_rgb, dt_rgb)，
    #     # 但解码时用 C_e → rgb 输出受 e 的语义引导
    #     ys_rgb = selective_scan(
    #         xs_rgb.view(B, -1, L).float(),
    #         dts_rgb.contiguous().view(B, -1, L).float(),
    #         As_rgb,  # rgb 独立
    #         Bs_rgb.contiguous().float(),
    #         Cs_e.contiguous().float(),
    #         Ds_rgb_,  # rgb 独立
    #         delta_bias,
    #     ).view(B, K, -1, H, W)
    #
    #     # e 的隐状态由 e 自身驱动，但解码时用 C_rgb
    #     ys_e = selective_scan(
    #         xs_e.view(B, -1, L).float(),
    #         dts_e.contiguous().view(B, -1, L).float(),
    #         As_e,  # e 独立
    #         Bs_e.contiguous().float(),
    #         Cs_rgb.contiguous().float(),
    #         Ds_e_,  # e 独立
    #         delta_bias,
    #     ).view(B, K, -1, H, W)
    #
    #     # ── Step 5: 合并 4 方向 ─────────────────────────────────────
    #     # CrossMerge.apply: (B, 4, D, H, W) → (B, D, H*W)
    #     y_rgb = CrossMerge.apply(ys_rgb)  # (B, D, L)
    #     y_e = CrossMerge.apply(ys_e)
    #
    #     # reshape → (B, H, W, D)，然后 LayerNorm
    #     y_rgb = y_rgb.transpose(1, 2).contiguous().view(B, H, W, -1)
    #     y_e = y_e.transpose(1, 2).contiguous().view(B, H, W, -1)
    #
    #     y_rgb = out_norm_rgb(y_rgb).to(x_rgb.dtype)
    #     y_e = out_norm_e(y_e).to(x_e.dtype)
    #
    #     return y_rgb, y_e

    # ================================================================
    # 消融实验用：1-dir 和 2-dir Cross Selective Scan
    # 与 cross_selective_scan_4dir 保持一致：
    #   - x_proj: rgb/e 各自独立
    #   - A/D: 共享
    #   - C矩阵: 跨模态交换
    # ================================================================

    def cross_selective_scan_1dir(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight_rgb: torch.Tensor = None,  # (1, dt_rank+2*N, d_inner)
            x_proj_weight_e: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,  # (1, d_inner, dt_rank)
            dt_projs_bias: torch.Tensor = None,  # (1, d_inner)
            A_logs: torch.Tensor = None,  # (1*d_inner, N)
            Ds: torch.Tensor = None,  # (1*d_inner,)
            out_norm_rgb: torch.nn.Module = None,
            out_norm_e: torch.nn.Module = None,
            nrows: int = -1,
            delta_softplus: bool = True,
    ):
        """
        单方向 1D Cross Mamba（行正向展平，无空间扫描）
        和 4-dir 版本唯一区别：K=1，不做 CrossScan，直接 flatten
        """
        B, D, H, W = x_rgb.shape
        N = A_logs.shape[1]
        K, D_inner, R = dt_projs_weight.shape  # K=1
        L = H * W

        if nrows < 1:
            if D_inner % 4 == 0:
                nrows = 4
            elif D_inner % 3 == 0:
                nrows = 3
            elif D_inner % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # ── Step 1: 单方向展开（行正向 flatten）──────────────────────
        xs_rgb = x_rgb.flatten(2, 3).unsqueeze(1)  # (B, 1, D, L)
        xs_e = x_e.flatten(2, 3).unsqueeze(1)

        # ── Step 2: 各自投影 ─────────────────────────────────────────
        x_dbl_rgb = torch.einsum("b k d l, k c d -> b k c l", xs_rgb, x_proj_weight_rgb)
        x_dbl_e = torch.einsum("b k d l, k c d -> b k c l", xs_e, x_proj_weight_e)

        dts_rgb, Bs_rgb, Cs_rgb = torch.split(x_dbl_rgb, [R, N, N], dim=2)
        dts_e, Bs_e, Cs_e = torch.split(x_dbl_e, [R, N, N], dim=2)

        dts_rgb = torch.einsum("b k r l, k d r -> b k d l", dts_rgb, dt_projs_weight)
        dts_e = torch.einsum("b k r l, k d r -> b k d l", dts_e, dt_projs_weight)

        # ── Step 3: 共享 A、D ────────────────────────────────────────
        As = -torch.exp(A_logs.float())
        Ds_ = Ds.float()
        delta_bias = dt_projs_bias.view(-1).float()

        def selective_scan(u, delta, A, B, C, D, db):
            return SelectiveScan.apply(u, delta, A, B, C, D, db, delta_softplus, nrows)

        # ── Step 4: C矩阵跨模态交换 ──────────────────────────────────
        ys_rgb = selective_scan(
            xs_rgb.view(B, -1, L).float(),
            dts_rgb.contiguous().view(B, -1, L).float(),
            As,
            Bs_rgb.contiguous().float(),
            Cs_e.contiguous().float(),  # ← 用 e 的 C 解码 rgb
            Ds_, delta_bias,
        )  # (B, D_inner, L)

        ys_e = selective_scan(
            xs_e.view(B, -1, L).float(),
            dts_e.contiguous().view(B, -1, L).float(),
            As,
            Bs_e.contiguous().float(),
            Cs_rgb.contiguous().float(),  # ← 用 rgb 的 C 解码 e
            Ds_, delta_bias,
        )  # (B, D_inner, L)

        # ── Step 5: reshape + norm ───────────────────────────────────
        y_rgb = ys_rgb.transpose(1, 2).contiguous().view(B, H, W, -1)
        y_e = ys_e.transpose(1, 2).contiguous().view(B, H, W, -1)

        y_rgb = out_norm_rgb(y_rgb).to(x_rgb.dtype)
        y_e = out_norm_e(y_e).to(x_e.dtype)

        return y_rgb, y_e


    def cross_selective_scan_2dir(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight_rgb: torch.Tensor = None,  # (2, dt_rank+2*N, d_inner)
            x_proj_weight_e: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,  # (2, d_inner, dt_rank)
            dt_projs_bias: torch.Tensor = None,  # (2, d_inner)
            A_logs: torch.Tensor = None,  # (2*d_inner, N)
            Ds: torch.Tensor = None,  # (2*d_inner,)
            out_norm_rgb: torch.nn.Module = None,
            out_norm_e: torch.nn.Module = None,
            nrows: int = -1,
            delta_softplus: bool = True,
    ):
        """
        双方向 2D Cross Mamba（行正向 + 行反向，水平双向扫描）
        和 4-dir 版本唯一区别：K=2，只取水平方向（0和2），不做列方向扫描
        """
        B, D, H, W = x_rgb.shape
        N = A_logs.shape[1]
        K, D_inner, R = dt_projs_weight.shape  # K=2
        L = H * W

        if nrows < 1:
            if D_inner % 4 == 0:
                nrows = 4
            elif D_inner % 3 == 0:
                nrows = 3
            elif D_inner % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # ── Step 1: 2方向展开（行正向 + 行反向）──────────────────────
        # CrossScan产生4个方向: [行正, 列正, 行反, 列反]
        # 只取 0(行正) 和 2(行反) 这一对
        xs_rgb_all = CrossScan.apply(x_rgb)  # (B, 4, D, L)
        xs_e_all = CrossScan.apply(x_e)
        xs_rgb = torch.stack([xs_rgb_all[:, 0], xs_rgb_all[:, 2]], dim=1)  # (B, 2, D, L)
        xs_e = torch.stack([xs_e_all[:, 0], xs_e_all[:, 2]], dim=1)

        # ── Step 2: 各自投影 ─────────────────────────────────────────
        x_dbl_rgb = torch.einsum("b k d l, k c d -> b k c l", xs_rgb, x_proj_weight_rgb)
        x_dbl_e = torch.einsum("b k d l, k c d -> b k c l", xs_e, x_proj_weight_e)

        dts_rgb, Bs_rgb, Cs_rgb = torch.split(x_dbl_rgb, [R, N, N], dim=2)
        dts_e, Bs_e, Cs_e = torch.split(x_dbl_e, [R, N, N], dim=2)

        dts_rgb = torch.einsum("b k r l, k d r -> b k d l", dts_rgb, dt_projs_weight)
        dts_e = torch.einsum("b k r l, k d r -> b k d l", dts_e, dt_projs_weight)

        # ── Step 3: 共享 A、D ────────────────────────────────────────
        As = -torch.exp(A_logs.float())
        Ds_ = Ds.float()
        delta_bias = dt_projs_bias.view(-1).float()

        def selective_scan(u, delta, A, B, C, D, db):
            return SelectiveScan.apply(u, delta, A, B, C, D, db, delta_softplus, nrows)

        # ── Step 4: C矩阵跨模态交换 ──────────────────────────────────
        ys_rgb = selective_scan(
            xs_rgb.contiguous().view(B, -1, L).float(),
            dts_rgb.contiguous().view(B, -1, L).float(),
            As,
            Bs_rgb.contiguous().float(),
            Cs_e.contiguous().float(),  # ← 用 e 的 C 解码 rgb
            Ds_, delta_bias,
        ).view(B, K, D_inner, L)

        ys_e = selective_scan(
            xs_e.contiguous().view(B, -1, L).float(),
            dts_e.contiguous().view(B, -1, L).float(),
            As,
            Bs_e.contiguous().float(),
            Cs_rgb.contiguous().float(),  # ← 用 rgb 的 C 解码 e
            Ds_, delta_bias,
        ).view(B, K, D_inner, L)

        # ── Step 5: 合并2方向（正向 + 翻转反向）─────────────────────
        # ys[:, 0]: 行正向输出  ys[:, 1]: 行反向输出（需 flip 回来）
        y_rgb = ys_rgb[:, 0] + ys_rgb[:, 1].flip(dims=[-1])  # (B, D_inner, L)
        y_e = ys_e[:, 0] + ys_e[:, 1].flip(dims=[-1])

        y_rgb = y_rgb.transpose(1, 2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(1, 2).contiguous().view(B, H, W, -1)

        y_rgb = out_norm_rgb(y_rgb).to(x_rgb.dtype)
        y_e = out_norm_e(y_e).to(x_e.dtype)

        return y_rgb, y_e


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossMambaFusionBlock(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CrossMambaFusion_SS2D_SSM(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb_cross, x_e_cross = self.op(x_rgb, x_e)
        x_rgb = x_rgb + self.drop_path1(x_rgb_cross)
        x_e = x_e + self.drop_path2(x_e_cross)
        return x_rgb, x_e

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)


# 4d selective scan, share A/D.
# ================================================================
# CrossMambaFusion_SS2D_SSM 消融版
# 切换方式：只需改 self.K 和 forward 里的函数调用（取消注释对应行）
# ================================================================
class CrossMambaFusion_SS2D_SSM(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    改为真正的 4 方向 2D 扫描（原来是 K=2 双向）。

    变化：
      - K: 2 → 4（四方向扫描）
      - x_proj: 共享 1 个 → rgb/e 各自独立（x_proj_weight_rgb / x_proj_weight_e）
      - dt_projs、A_logs、Ds 仍然共享
      - forward 调用 cross_selective_scan_4dir，C 矩阵跨模态交换
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            d_conv=3,
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影：rgb 和 e 各自独立
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # 深度可分离卷积：rgb 和 e 各自独立
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_e = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # 输出投影
        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # ── 关键改动：K=4，rgb/e 分别有独立的 x_proj ──────────────
        # ============================================================
        # ⬇ 消融实验：只需修改这一行 K 的值
        # 1-dir: self.K = 1
        # 2-dir: self.K = 2
        # 4-dir: self.K = 4
        # ============================================================
        self.K = 4

        # rgb 的 x_proj：生成 rgb 自己的 B_rgb、C_rgb、dt_rgb
        x_proj_rgb = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight_rgb = nn.Parameter(
            torch.stack([t.weight for t in x_proj_rgb], dim=0)
        )  # (4, dt_rank+2*N, d_inner)
        del x_proj_rgb

        # e 的 x_proj：生成 e 自己的 B_e、C_e、dt_e
        x_proj_e = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                      bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight_e = nn.Parameter(
            torch.stack([t.weight for t in x_proj_e], dim=0)
        )  # (4, dt_rank+2*N, d_inner)
        del x_proj_e

        # dt_projs、A_logs、Ds：K=4，两模态共享
        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                         dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (4, d_inner, dt_rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (4, d_inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner,
                                      copies=self.K, merge=True)  # (4*d_inner, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)   # (4*d_inner,)

        self.out_norm_rgb = nn.LayerNorm(self.d_inner)
        self.out_norm_e = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        # 输入格式: (B, H, W, C)，来自 CrossMambaFusionBlock
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        B, H, W, D = x_rgb.shape

        if self.d_conv > 1:
            # (B,H,W,D) → (B,D,H,W) 供 Conv2d 使用
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans   = x_e.permute(0, 3, 1, 2).contiguous()

            # rgb 和 e 各自用独立的 conv2d（保持特征独立性）
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans))    # (B, d_inner, H, W)
            x_e_conv   = self.act(self.conv2d_e(x_e_trans))    # (B, d_inner, H, W)

            # ============================================================
            # ⬇ 消融实验：取消注释对应行，注释掉其他两行
            # ============================================================

            # # 1-dir（单方向行正向）:
            # y_rgb, y_e = cross_selective_scan_1dir(
            #     x_rgb_conv, x_e_conv,
            #     self.x_proj_weight_rgb, self.x_proj_weight_e,
            #     self.dt_projs_weight, self.dt_projs_bias,
            #     self.A_logs, self.Ds,
            #     self.out_norm_rgb, self.out_norm_e,
            # )

            # # 2-dir（水平正向+反向）:
            #
            # y_rgb, y_e = cross_selective_scan_2dir(
            #     x_rgb_conv, x_e_conv,
            #     self.x_proj_weight_rgb, self.x_proj_weight_e,
            #     self.dt_projs_weight, self.dt_projs_bias,
            #     self.A_logs, self.Ds,
            #     self.out_norm_rgb, self.out_norm_e,
            # )

            # # ⭐ 调用 4 方向 2D Cross Scan（替换原来的 K=2 双向）
            # # x_proj_weight_rgb/e 分别提取各模态的 B、C、delta
            # # C 矩阵在函数内部跨模态交换：rgb 用 C_e 解码，e 用 C_rgb 解码
            y_rgb, y_e = cross_selective_scan_4dir(
                x_rgb_conv, x_e_conv,
                self.x_proj_weight_rgb,    # rgb 独立 x_proj
                self.x_proj_weight_e,      # e 独立 x_proj
                self.dt_projs_weight,      # 共享 dt
                self.dt_projs_bias,
                self.A_logs,               # 共享 A
                self.Ds,                   # 共享 D
                self.out_norm_rgb,
                self.out_norm_e,
            )
            # y_rgb, y_e 已经是 (B, H, W, d_inner)

        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        out_e   = self.dropout_e(self.out_proj_e(y_e))
        return out_rgb, out_e

# # 4d selective scan with A/D indepandent, no share A/D.
# class CrossMambaFusion_SS2D_SSM(nn.Module):
#     '''
#     Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
#     改为真正的 4 方向 2D 扫描（原来是 K=2 双向）。
#
#     变化：
#       - K: 2 → 4（四方向扫描）
#       - x_proj: 共享 1 个 → rgb/e 各自独立（x_proj_weight_rgb / x_proj_weight_e）
#       - dt_projs、A_logs、Ds 仍然共享
#       - forward 调用 cross_selective_scan_4dir，C 矩阵跨模态交换
#     '''
#
#     def __init__(
#             self,
#             # basic dims ===========
#             d_model=96,
#             d_state=16,
#             ssm_ratio=2,
#             dt_rank="auto",
#             # dwconv ===============
#             d_conv=3,
#             conv_bias=True,
#             # ======================
#             dropout=0.,
#             bias=False,
#             # dt init ==============
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             # ======================
#             softmax_version=False,
#             # ======================
#             **kwargs,
#     ):
#         factory_kwargs = {"device": None, "dtype": None}
#         super().__init__()
#         self.softmax_version = softmax_version
#         self.d_model = d_model
#         self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
#         self.d_conv = d_conv
#         self.expand = ssm_ratio
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#
#         # 输入投影：rgb 和 e 各自独立
#         self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#         self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#
#         # 深度可分离卷积：rgb 和 e 各自独立
#         if self.d_conv > 1:
#             self.conv2d = nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 padding=(d_conv - 1) // 2,
#                 **factory_kwargs,
#             )
#             self.conv2d_e = nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 padding=(d_conv - 1) // 2,
#                 **factory_kwargs,
#             )
#             self.act = nn.SiLU()
#
#         # 输出投影
#         self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#         self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#
#         # ── 关键改动：K=4，rgb/e 分别有独立的 x_proj ──────────────
#         self.K = 4
#
#         # rgb 的 x_proj：生成 rgb 自己的 B_rgb、C_rgb、dt_rgb
#         x_proj_rgb = [
#             nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
#                       bias=False, **factory_kwargs)
#             for _ in range(self.K)
#         ]
#         self.x_proj_weight_rgb = nn.Parameter(
#             torch.stack([t.weight for t in x_proj_rgb], dim=0)
#         )  # (4, dt_rank+2*N, d_inner)
#         del x_proj_rgb
#
#         # e 的 x_proj：生成 e 自己的 B_e、C_e、dt_e
#         x_proj_e = [
#             nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
#                       bias=False, **factory_kwargs)
#             for _ in range(self.K)
#         ]
#         self.x_proj_weight_e = nn.Parameter(
#             torch.stack([t.weight for t in x_proj_e], dim=0)
#         )  # (4, dt_rank+2*N, d_inner)
#         del x_proj_e
#
#         # dt_projs、A_logs、Ds：K=4，两模态共享
#         self.dt_projs = [
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
#                          dt_min, dt_max, dt_init_floor, **factory_kwargs)
#             for _ in range(self.K)
#         ]
#         self.dt_projs_weight = nn.Parameter(
#             torch.stack([t.weight for t in self.dt_projs], dim=0)
#         )  # (4, d_inner, dt_rank)
#         self.dt_projs_bias = nn.Parameter(
#             torch.stack([t.bias for t in self.dt_projs], dim=0)
#         )  # (4, d_inner)
#         del self.dt_projs
#
#         # ✅ 改为：各自独立
#         self.A_logs_rgb = self.A_log_init(self.d_state, self.d_inner,
#                                           copies=self.K, merge=True)  # (4*d_inner, N)
#         self.A_logs_e = self.A_log_init(self.d_state, self.d_inner,
#                                         copies=self.K, merge=True)  # (4*d_inner, N)
#         self.Ds_rgb = self.D_init(self.d_inner, copies=self.K, merge=True)  # (4*d_inner,)
#         self.Ds_e = self.D_init(self.d_inner, copies=self.K, merge=True)  # (4*d_inner,)
#
#         self.out_norm_rgb = nn.LayerNorm(self.d_inner)
#         self.out_norm_e = nn.LayerNorm(self.d_inner)
#
#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
#
#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#
#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         # dt_proj.bias._no_reinit = True
#
#         return dt_proj
#
#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 0:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log
#
#     @staticmethod
#     def D_init(d_inner, copies=-1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 0:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D
#
#     def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
#         # 输入格式: (B, H, W, C)，来自 CrossMambaFusionBlock
#         x_rgb = self.in_proj(x_rgb)
#         x_e = self.in_proj_modalx(x_e)
#         B, H, W, D = x_rgb.shape
#
#         if self.d_conv > 1:
#             # (B,H,W,D) → (B,D,H,W) 供 Conv2d 使用
#             x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
#             x_e_trans   = x_e.permute(0, 3, 1, 2).contiguous()
#
#             # rgb 和 e 各自用独立的 conv2d（保持特征独立性）
#             x_rgb_conv = self.act(self.conv2d(x_rgb_trans))    # (B, d_inner, H, W)
#             x_e_conv   = self.act(self.conv2d_e(x_e_trans))    # (B, d_inner, H, W)
#
#             # ⭐ 调用 4 方向 2D Cross Scan（替换原来的 K=2 双向）
#             # x_proj_weight_rgb/e 分别提取各模态的 B、C、delta
#             # C 矩阵在函数内部跨模态交换：rgb 用 C_e 解码，e 用 C_rgb 解码
#             # ✅ 改为
#             y_rgb, y_e = cross_selective_scan_4dir(
#                 x_rgb_conv, x_e_conv,
#                 self.x_proj_weight_rgb,
#                 self.x_proj_weight_e,
#                 self.dt_projs_weight,
#                 self.dt_projs_bias,
#                 self.A_logs_rgb,  # rgb 独立
#                 self.A_logs_e,  # e 独立
#                 self.Ds_rgb,  # rgb 独立
#                 self.Ds_e,  # e 独立
#                 self.out_norm_rgb,
#                 self.out_norm_e,
#             )
#             # y_rgb, y_e 已经是 (B, H, W, d_inner)
#
#         out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
#         out_e   = self.dropout_e(self.out_proj_e(y_e))
#         return out_rgb, out_e

# class CrossMambaFusion_SS2D_SSM(nn.Module):
#     '''
#     Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
#     '''
#
#     def __init__(
#             self,
#             # basic dims ===========
#             d_model=96,
#             d_state=16,
#             ssm_ratio=2,
#             dt_rank="auto",
#             # dwconv ===============
#             # d_conv=-1, # < 2 means no conv
#             d_conv=3,  # < 2 means no conv
#             conv_bias=True,
#             # ======================
#             dropout=0.,
#             bias=False,
#             # dt init ==============
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             # ======================
#             softmax_version=False,
#             # ======================
#             **kwargs,
#     ):
#         factory_kwargs = {"device": None, "dtype": None}
#         super().__init__()
#         self.softmax_version = softmax_version
#         self.d_model = d_model
#         self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
#         self.d_conv = d_conv
#         self.expand = ssm_ratio
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#
#         self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#         self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#
#         # conv =======================================
#         if self.d_conv > 1:
#             self.conv2d = nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 padding=(d_conv - 1) // 2,
#                 **factory_kwargs,
#             )
#             self.act = nn.SiLU()
#
#         self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#         self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#
#         self.CMA_ssm = Cross_Mamba_Attention_SSM(
#             d_model=self.d_model,
#             d_state=self.d_state,
#             ssm_ratio=ssm_ratio,
#             dt_rank=dt_rank,
#             dt_min=dt_min,
#             dt_max=dt_max,
#             dt_init=dt_init,
#             dt_scale=dt_scale,
#             dt_init_floor=dt_init_floor,
#             **kwargs,
#         )
#
#     def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
#         x_rgb = self.in_proj(x_rgb)
#         x_e = self.in_proj_modalx(x_e)
#         B, H, W, D = x_rgb.shape
#         if self.d_conv > 1:
#             x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
#             x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
#             x_rgb_conv = self.act(self.conv2d(x_rgb_trans))  # (b, d, h, w)
#             x_e_conv = self.act(self.conv2d(x_e_trans))  # (b, d, h, w)
#             x_rgb_conv = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
#             x_e_conv = rearrange(x_e_conv, "b d h w -> b (h w) d")
#             y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv)
#             # to b, d, h, w
#             y_rgb = y_rgb.view(B, H, W, -1)
#             y_e = y_e.view(B, H, W, -1)
#
#         out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
#         out_e = self.dropout_e(self.out_proj_e(y_e))
#         return out_rgb, out_e

# class CrossMambaFusion_SS2D_SSM(nn.Module):
#     '''
#     Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
#     '''
#
#     def __init__(
#             self,
#             # basic dims ===========
#             d_model=96,
#             d_state=16,
#             ssm_ratio=2,
#             dt_rank="auto",
#             # dwconv ===============
#             # d_conv=-1, # < 2 means no conv
#             d_conv=3,  # < 2 means no conv
#             conv_bias=True,
#             # ======================
#             dropout=0.,
#             bias=False,
#             # dt init ==============
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             # ======================
#             softmax_version=False,
#             # ======================
#             **kwargs,
#     ):
#         factory_kwargs = {"device": None, "dtype": None}
#         super().__init__()
#         self.softmax_version = softmax_version
#         self.d_model = d_model
#         self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
#         self.d_conv = d_conv
#         self.expand = ssm_ratio
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#
#         self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#         self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#
#         # conv =======================================
#         if self.d_conv > 1:
#             self.conv2d = nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 padding=(d_conv - 1) // 2,
#                 **factory_kwargs,
#             )
#             self.act = nn.SiLU()
#
#         self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#         self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#
#         self.K = 2
#         self.x_proj = [
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2),
#                       bias=False, **factory_kwargs)
#             for _ in range(self.K)
#         ]
#         self.x_proj_weight = nn.Parameter(
#             torch.stack([t.weight for t in self.x_proj], dim=0)
#         )
#         del self.x_proj
#
#         self.dt_projs = [
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
#                          dt_min, dt_max, dt_init_floor, **factory_kwargs)
#             for _ in range(self.K)
#         ]
#         self.dt_projs_weight = nn.Parameter(
#             torch.stack([t.weight for t in self.dt_projs], dim=0)
#         )
#         self.dt_projs_bias = nn.Parameter(
#             torch.stack([t.bias for t in self.dt_projs], dim=0)
#         )
#         del self.dt_projs
#
#         self.A_logs = self.A_log_init(self.d_state, self.d_inner,
#                                       copies=self.K, merge=True)
#         self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)
#
#         self.out_norm_rgb = nn.LayerNorm(self.d_inner)
#         self.out_norm_e = nn.LayerNorm(self.d_inner)
#
#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
#
#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#
#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         # dt_proj.bias._no_reinit = True
#
#         return dt_proj
#
#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 0:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log
#
#     @staticmethod
#     def D_init(d_inner, copies=-1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 0:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D
#
#     def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
#         x_rgb = self.in_proj(x_rgb)
#         x_e = self.in_proj_modalx(x_e)
#         B, H, W, D = x_rgb.shape
#         if self.d_conv > 1:
#             x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
#             x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
#             x_rgb_conv = self.act(self.conv2d(x_rgb_trans))  # (b, d, h, w)
#             x_e_conv = self.act(self.conv2d(x_e_trans))  # (b, d, h, w)
#
#             # ⭐ 关键改动：调用2D空间扫描替代原来的1D SSM
#             y_rgb, y_e = cross_selective_scan_multimodal_k2(
#                 x_rgb_conv, x_e_conv,
#                 self.x_proj_weight, None,
#                 self.dt_projs_weight, self.dt_projs_bias,
#                 self.A_logs, self.Ds,
#                 self.out_norm_rgb, self.out_norm_e,
#                 softmax_version=self.softmax_version,
#             )
#             # 输出已经是 (B, H, W, d_inner)
#
#         out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
#         out_e = self.dropout_e(self.out_proj_e(y_e))
#         return out_rgb, out_e


class Cross_Mamba_Attention_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)

        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)
        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()

        y_rgb = selective_scan(
            x_rgb, dt_rgb,
            A_rgb, B_rgb, C_e, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        y_e = selective_scan(
            x_e, dt_e,
            A_e, B_e, C_rgb, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        return y_rgb, y_e


# class ConcatMambaFusionBlock(nn.Module):
#     '''
#     Concat Mamba (ConMB) fusion, with 2d SSM
#     '''
#
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             attn_drop_rate: float = 0,
#             d_state: int = 4,
#             dt_rank: Any = "auto",
#             ssm_ratio=2.0,
#             shared_ssm=False,
#             softmax_version=False,
#             use_checkpoint: bool = False,
#             mlp_ratio=0.0,
#             act_layer=nn.GELU,
#             drop: float = 0.0,
#             **kwargs,
#     ):
#         super().__init__()
#         self.use_checkpoint = use_checkpoint
#         # self.norm = norm_layer(hidden_dim)
#         self.op = ConMB_SS2D(
#             d_model=hidden_dim,
#             dropout=attn_drop_rate,
#             d_state=d_state,
#             ssm_ratio=ssm_ratio,
#             dt_rank=dt_rank,
#             shared_ssm=shared_ssm,
#             softmax_version=softmax_version,
#             **kwargs
#         )
#         self.drop_path = DropPath(drop_path)
#
#         self.mlp_branch = mlp_ratio > 0
#         if self.mlp_branch:
#             self.norm2 = norm_layer(hidden_dim)
#             mlp_hidden_dim = int(hidden_dim * mlp_ratio)
#             self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)
#
#     def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
#         x = x_rgb + x_e + self.drop_path(self.op(x_rgb, x_e))
#         if self.mlp_branch:
#             x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
#         return x
#
#     def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
#         '''
#         B C H W, B C H W -> B C H W
#         '''
#         if self.use_checkpoint:
#             return checkpoint.checkpoint(self._forward, x_rgb, x_e)
#         else:
#             return self._forward(x_rgb, x_e)
#
#
# DEV = False


import torch
import torch.nn as nn
class ConcatMambaFusionBlock(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            use_gate: bool = True,
            alpha_init: float = 0.0,
            gate_bias_init: float = -1.0,
            # gate_bias_init: float = -2.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = ConMB_SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        ) #单流
        # self.op = ConMB_SS2D_2L(
        #     d_model=hidden_dim,
        #     dropout=attn_drop_rate,
        #     d_state=d_state,
        #     ssm_ratio=ssm_ratio,
        #     dt_rank=dt_rank,
        #     shared_ssm=shared_ssm,
        #     softmax_version=softmax_version,
        #     **kwargs
        # )  # 2L拼接双流
        self.drop_path = DropPath(drop_path)
        # merge_conv 替代直接相加，让模型学习如何融合两个模态
        self.merge_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # self.merge_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False)
        # self._log_counter = 0

        # 门控：自适应控制ConMB的贡献量
        self.ssm_alpha = nn.Parameter(torch.tensor(alpha_init))
        self.use_gate = use_gate
        # condition 0: x = x0 + g * y, g= sigmod(conv(cat[x_rgb, xe]))
        # if use_gate:
        #     self.gate_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=True)
        #     nn.init.zeros_(self.gate_conv.weight)
        #     nn.init.constant_(self.gate_conv.bias, gate_bias_init)  # sigmoid(-2)≈0.12，初始近似关闭
        # condition 1: 用模态间的差异作为gate输入，而不是拼接, x = x0 + g * y, but g is diff, not cat.
        if use_gate:
            # 改后
            self.gate_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
            nn.init.constant_(self.gate_conv.bias, 0.0)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        # merge_conv替代x_rgb + x_e，避免直接相加的冗余问题
        x_rgb_chw = x_rgb.permute(0, 3, 1, 2).contiguous()
        x_e_chw = x_e.permute(0, 3, 1, 2).contiguous()
        x0 = self.merge_conv(torch.cat([x_rgb_chw, x_e_chw], dim=1)).permute(0, 2, 3, 1).contiguous()

        y = self.drop_path(self.op(x_rgb, x_e))

        if self.use_gate:
            # gate_in = torch.cat([x_rgb_chw, x_e_chw], dim=1)  # condition 0
            gate_in = x_rgb_chw - x_e_chw  # C   #condition 1
            g = torch.sigmoid(self.gate_conv(gate_in).permute(0, 2, 3, 1).contiguous())
            print(
                f"g_mean: {g.mean().item():.4f}, g_min: {g.min().item():.4f}, g_max: {g.max().item():.4f}, "
                f"g_std: {g.std().item():.4f}, bias_mean: {self.gate_conv.bias.mean().item():.4f}, weight_abs_mean: {self.gate_conv.weight.abs().mean().item():.6f}")
            x = x0 + g * y
        else:
            x = x0 + y
        # if self.use_gate:
        #     gate_in = torch.cat([x_rgb_chw, x_e_chw], dim=1)
        #     g = torch.sigmoid(self.gate_conv(gate_in).permute(0, 2, 3, 1).contiguous())
        #     print(
        #         f"ssm_alpha: {self.ssm_alpha.item():.4f}, g_mean: {g.mean().item():.4f}, g_min: {g.min().item():.4f}, g_max: {g.max().item():.4f}")
        #     x = x0 + self.ssm_alpha * g * y
        # else:
        #     x = x0 + self.ssm_alpha * y

        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)


DEV = False

# class ConMB_SS2D(nn.Module):
#     '''
#     Multimodal Mamba Selective Scan 2D
#     '''
#
#     def __init__(
#             self,
#             # basic dims ===========
#             d_model=96,
#             d_state=4,
#             ssm_ratio=2,
#             dt_rank="auto",
#             # dwconv ===============
#             # d_conv=-1, # < 2 means no conv
#             d_conv=3,  # < 2 means no conv
#             conv_bias=True,
#             # ======================
#             dropout=0.,
#             bias=False,
#             # dt init ==============
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             # ======================
#             softmax_version=False,
#             # ======================
#             **kwargs,
#     ):
#         if DEV:
#             d_conv = -1
#
#         factory_kwargs = {"device": None, "dtype": None}
#         super().__init__()
#         self.softmax_version = softmax_version
#         self.d_model = d_model
#         self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
#         self.d_conv = d_conv
#         self.expand = ssm_ratio
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#
#         self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#         self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
#
#         # conv =======================================
#         if self.d_conv > 1:
#             self.conv2d = nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 padding=(d_conv - 1) // 2,
#                 **factory_kwargs,
#             )
#             self.conv2d_modalx = nn.Conv2d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 groups=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 padding=(d_conv - 1) // 2,
#                 **factory_kwargs,
#             )
#             self.act = nn.SiLU()
#
#         # x proj; dt proj ============================
#         self.K = 2
#         self.x_proj = [
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
#             for _ in range(self.K)
#         ]
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
#         del self.x_proj
#
#         self.dt_projs = [
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
#             for _ in range(self.K)
#         ]
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
#         del self.dt_projs
#
#         # A, D =======================================
#         self.K2 = self.K
#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K * D, N)
#         self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K * D)
#
#         # out proj =======================================
#         if not self.softmax_version:
#             self.out_norm1 = nn.LayerNorm(self.d_inner)
#             self.out_norm2 = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Sequential(
#             nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
#             nn.SiLU(inplace=True),
#             nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
#             nn.Sigmoid(),
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
#             nn.SiLU(inplace=True),
#             nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
#             nn.Sigmoid(),
#         )
#
#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
#
#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#
#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         # dt_proj.bias._no_reinit = True
#
#         return dt_proj
#
#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 0:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log
#
#     @staticmethod
#     def D_init(d_inner, copies=-1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 0:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D
#
#     def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
#         return cross_selective_scan_multimodal_k2(
#             x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
#             self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None), self.softmax_version,
#             nrows=nrows,
#         )
#
#     def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
#         x_rgb = self.in_proj(x_rgb)
#         x_e = self.in_proj_modalx(x_e)
#         if self.d_conv > 1:
#             x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
#             x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
#             x_rgb_conv = self.act(self.conv2d(x_rgb_trans))  # (b, d, h, w)
#             x_e_conv = self.act(self.conv2d_modalx(x_e_trans))  # (b, d, h, w)
#             y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv)  # b, d, h, w -> b, h, w, d
#             # SE to get attention, scale
#             b, d, h, w = x_rgb_trans.shape
#             x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
#             x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
#             x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()  # b, 1, 1, d
#             x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
#             y_rgb = y_rgb * x_e_exitation
#             y_e = y_e * x_rgb_exitation
#             y = torch.concat([y_rgb, y_e], dim=-1)
#         out = self.dropout(self.out_proj(y))
#         return out
#

# single sequnce ssm
class ConMB_SS2D(nn.Module):
    '''
    Concat Mamba SS2D - 改进版
    先用 fusion conv 将两模态压缩为单一特征，再用单流 SSM（4方向扫描）建模长程依赖
    相比原版（直接2L拼接）：序列同质，无模态边界问题，SSM更容易建模
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 两模态分别投影（保留各自语义）
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # 新增：1x1 fusion conv，在进入 SSM 前先跨模态融合
        # 输入是 cat([x_rgb, x_e], dim=1) → 2*d_inner → d_inner
        self.pre_fusion_conv = nn.Sequential(
            nn.Conv2d(self.d_model * 2, self.d_inner, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.d_inner),
            # nn.SiLU()
            nn.ReLU(inplace=True)
        )

        # DWConv：局部空间建模（在 fusion conv 之后，SSM 之前）
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,       # depthwise
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # SSM 参数：K=4（4方向扫描，单流输入）
        self.K = 4
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))      # (K, inner)
        del self.dt_projs

        # A, D：K=4
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K*D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)                        # (K*D)

        # 输出层：单流输出 d_inner → d_model（原版是 d_inner*2，现在是单流）
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    # ---- 静态方法（和原版一致，直接复用）----

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # ---- 核心扫描：单流 4方向 SSM ----

    def forward_corev2_single(self, x: torch.Tensor, nrows=-1):
        """
        单流 4方向 SS2D
        输入 x: (B, d_inner, H, W)
        输出 y: (B, H, W, d_inner)
        复用 cross_selective_scan（已有的4方向单流实现）
        """
        return cross_selective_scan(
            x,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            getattr(self, "out_norm", None),
            self.softmax_version,
            nrows=nrows,
        )

    # def forward_corev2_single(self, x: torch.Tensor, nrows=-1):
    #     # 单流 K=2 双向扫描，不再是跨模态  cross_selective_scan_d1
    #     return cross_selective_scan_multimodal_k2(
    #         x, self.x_proj_weight, None,
    #         self.dt_projs_weight, self.dt_projs_bias,
    #         self.A_logs, self.Ds,
    #         getattr(self, "out_norm1", None),
    #         self.softmax_version,
    #         nrows=nrows,
    #     )

    # ---- forward ----

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        """
        输入：
            x_rgb: (B, H, W, d_model)  来自 CrossMamba 输出的 rgb 特征
            x_e:   (B, H, W, d_model)  来自 CrossMamba 输出的 e   特征
        输出：
            out:   (B, H, W, d_model)  融合后的特征
        """

        # # Step 1: 两模态分别线性投影 → (B, H, W, d_inner)
        # x_rgb = self.in_proj(x_rgb)
        # x_e = self.in_proj_modalx(x_e)

        # Step 2: 转为 (B, d_inner, H, W) 方便 Conv
        x_rgb_chw = x_rgb.permute(0, 3, 1, 2).contiguous()
        x_e_chw = x_e.permute(0, 3, 1, 2).contiguous()

        # Step 3: 1x1 fusion conv，跨模态 channel 融合 → (B, d_inner, H, W)
        x_fused = self.pre_fusion_conv(
            torch.cat([x_rgb_chw, x_e_chw], dim=1)
        )

        # Step 4: DWConv，局部空间建模 → (B, d_inner, H, W)
        if self.d_conv > 1:
            x_fused = self.act(self.conv2d(x_fused))

        # Step 5: 单流 4方向 SSM，全局长程依赖 → (B, H, W, d_inner)
        y = self.forward_corev2_single(x_fused)

        # Step 6: 投影回 d_model → (B, H, W, d_model)
        out = self.dropout(self.out_proj(y))
        return out

class ConMB_SS2D_2L(nn.Module):
    '''
    Concat Mamba SS2D - 改进版
    先用 fusion conv 将两模态压缩为单一特征，再用单流 SSM（4方向扫描）建模长程依赖
    相比原版（直接2L拼接）：序列同质，无模态边界问题，SSM更容易建模
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 两模态分别投影（保留各自语义）
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # DWConv：局部空间建模（在 fusion conv 之后，SSM 之前）
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,       # depthwise
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_modalx = nn.Conv2d(
                        in_channels=self.d_inner,
                        out_channels=self.d_inner,
                        groups=self.d_inner,
                        bias=conv_bias,
                        kernel_size=d_conv,
                        padding=(d_conv - 1) // 2,
                        **factory_kwargs,
                    )
            self.act = nn.SiLU()

        # SSM 参数：K=4（4方向扫描，单流输入）
        self.K = 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))      # (K, inner)
        del self.dt_projs

        # A, D：K=4
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K*D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)                        # (K*D)

        # 输出层：单流输出 d_inner → d_model（原版是 d_inner*2，现在是单流）
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)

    # ---- 静态方法（和原版一致，直接复用）----

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    # 2 scan

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        """
        输入：
            x_rgb: (B, H, W, d_model)  来自 CrossMamba 输出的 rgb 特征
            x_e:   (B, H, W, d_model)  来自 CrossMamba 输出的 e   特征
        输出：
            out:   (B, H, W, d_model)  融合后的特征
        """

        # Step 1: 两模态分别线性投影 → (B, H, W, d_inner)
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)

        # Step 2: 转为 (B, d_inner, H, W) 方便 Conv
        x_rgb_chw = x_rgb.permute(0, 3, 1, 2).contiguous()
        x_e_chw = x_e.permute(0, 3, 1, 2).contiguous()

        # Step 3: 不合并，各自 DWConv
        x_rgb_conv = self.act(self.conv2d(x_rgb_chw))
        x_e_conv = self.act(self.conv2d_modalx(x_e_chw))  # 需要加 self.conv2d_e
        # Step 4: 2L 拼接扫描
        y_rgb, y_e = cross_selective_scan_multimodal_k2(
            x_rgb_conv, x_e_conv,
            self.x_proj_weight, None,
            self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            self.out_norm1, self.out_norm2,
        )
        # Step 5: concat → out_proj
        y = self.out_proj(torch.cat([y_rgb, y_e], dim=-1))
        return y  # 直接返回，不走外面的 out_proj