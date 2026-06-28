"""
Ouster OS1-128 球坐标投影模块
严格按照 extract_labels_from_json.py 中的 point_to_pix() 逻辑实现

Bus数据:    shift_degrees=180.0
Boston数据: shift_degrees=90.0

位置: pcdet/models/utils/ouster_projection.py
"""

import torch
import numpy as np


# ===== 根据 frame_id 前缀自动判断 shift_degrees =====
def get_shift_degrees_from_frame_id(frame_id: str) -> float:
    """
    根据 frame_id 前缀判断 shift_degrees
    bus_30_01_pcd_out_XXXXXX  → 180.0
    boston_01_pcd_out_XXXXXX  → 90.0
    """
    frame_id = str(frame_id)  # ← 加这一行
    if frame_id.startswith('boston'):
        return 90.0
    else:
        return 180.0  # bus数据默认


class OusterProjection:
    """
    Ouster OS1-128 球坐标投影
    将3D点（或3D box中心）投影到 Near-IR range image 的像素坐标

    完全对应 extract_labels_from_json.py 的 point_to_pix() 函数
    支持批量 PyTorch tensor 操作，用于 TransFusion 第二层 decoder
    """

    def __init__(self, beam_altitude_angles, img_width=1024, img_height=128):
        """
        Args:
            beam_altitude_angles: list[float], 128个beam的仰角（度），
                                  从 metadata.beam_altitude_angles 读取
            img_width:  图像宽度，默认1024
            img_height: 图像高度，默认128
        """
        self.img_w = img_width
        self.img_h = img_height
        # 转为固定的numpy数组，推理时再搬到GPU
        self.altitude_angles_np = np.array(beam_altitude_angles, dtype=np.float32)

    def project_points_batch(self, points_3d: torch.Tensor,
                             shift_degrees: float) -> torch.Tensor:
        """
        批量投影3D点到图像坐标（与 point_to_pix 完全一致）

        Args:
            points_3d:     [N, 3] float tensor (x, y, z)，LiDAR坐标系
            shift_degrees: float，Bus=180.0, Boston=90.0

        Returns:
            pix_coords: [N, 2] float tensor (col=u, row=v)，图像像素坐标
                        col ∈ [0, img_w), row ∈ [0, img_h)
        """
        device = points_3d.device
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]

        l = torch.sqrt(x * x + y * y + z * z).clamp(min=1e-6)

        # ── 垂直方向（行坐标）──────────────────────────────
        # 严格对应: sita128 = arcsin(z/l) * 180/π
        # num128 = argmin(|list128 - sita128|)
        sita128 = torch.arcsin((z / l).clamp(-1.0, 1.0)) * 180.0 / np.pi  # [N]

        alt_tensor = torch.tensor(
            self.altitude_angles_np, dtype=torch.float32, device=device
        )  # [128]

        # [N, 128] 距离矩阵，找最近beam
        diff = torch.abs(sita128.unsqueeze(1) - alt_tensor.unsqueeze(0))  # [N, 128]
        row = torch.argmin(diff, dim=1).float()  # [N]

        # ── 水平方向（列坐标）──────────────────────────────
        # 严格对应:
        #   target_angle = -arctan2(y, x) * 180/π
        #   if target_angle < 0: target_angle += 360
        #   target_angle = (target_angle + shift_degrees) % 360
        #   col = int(target_angle / 360 * IMG_WIDTH)
        target_angle = -torch.arctan2(y, x) * 180.0 / np.pi  # [N]
        target_angle = torch.where(
            target_angle < 0, target_angle + 360.0, target_angle
        )
        target_angle = torch.fmod(target_angle + shift_degrees, 360.0)
        col = target_angle / 360.0 * self.img_w  # [N]，float（不取int，保留亚像素）

        pix_coords = torch.stack([col, row], dim=1)  # [N, 2]
        return pix_coords

    def project_query_centers_batch(self, query_centers: torch.Tensor,
                                    shift_degrees_list: list) -> torch.Tensor:
        """
        为 TransFusion 第二层 decoder 批量投影 object query 中心点

        Args:
            query_centers:      [B, N_q, 3] float tensor，每个query的3D中心 (x,y,z)
            shift_degrees_list: list[float]，长度为B，每个样本对应的shift_degrees
                                （Bus=180.0, Boston=90.0）

        Returns:
            pix_coords: [B, N_q, 2] float tensor (u=col, v=row)
        """
        B, N_q, _ = query_centers.shape
        results = []

        for b in range(B):
            pts = query_centers[b]           # [N_q, 3]
            shift = shift_degrees_list[b]    # float
            coords = self.project_points_batch(pts, shift)  # [N_q, 2]
            results.append(coords)

        return torch.stack(results, dim=0)  # [B, N_q, 2]