"""
保存Ouster OS1-128的beam_altitude_angles到numpy文件
在训练BevFusion前必须先运行此脚本

使用方法：
  python tools/save_beam_angles.py \
    --metadata /media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.json \
    --output /home/yanan/Downloads/projects/multimodal_detection/data/beam_altitude_angles.npy
"""

import argparse
import numpy as np


def save_beam_angles(metadata_path, output_path):
    from ouster.sdk import client

    print(f"加载sensor metadata: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())

    beam_angles = np.array(metadata.beam_altitude_angles)  # [128]，单位：度
    print(f"beam_altitude_angles: {len(beam_angles)} 个beam")
    print(f"  最大仰角: {beam_angles.max():.2f}°")
    print(f"  最小仰角: {beam_angles.min():.2f}°")
    print(f"  前5个: {beam_angles[:5]}")
    print(f"  后5个: {beam_angles[-5:]}")

    np.save(output_path, beam_angles)
    print(f"✓ 已保存: {output_path}")

    return beam_angles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str,
                        default='/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.json',
                        help='Ouster sensor metadata JSON路径')
    parser.add_argument('--output', type=str,
                        default='/home/yanan/Downloads/projects/multimodal_detection/data/beam_altitude_angles.npy',
                        help='输出numpy文件路径')
    args = parser.parse_args()

    save_beam_angles(args.metadata, args.output)
    print("\n注意：OS1-128的beam_altitude_angles在所有子集间是相同的（同一型号传感器）。")
    print("只需运行一次即可，Bus和Boston子集共用同一个文件。")