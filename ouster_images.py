import json
import time
import numpy as np
import cv2
import os
import argparse

from ouster.sdk import client
from ouster.sdk.client import ChanField
from ouster.sdk import open_source


# 获取扫描尺寸和帧速率
def get_scan_size_and_fps(sensor_info: client.SensorInfo):
    w = sensor_info.format.columns_per_frame
    h = sensor_info.format.pixels_per_column
    # fps = sensor_info.format.udp_profile_lidar.frequency
    fps = 10
    # fps = 20
    return w, h, fps


# 获取每帧图像数据
def get_frame_from_scan(scan, channel, metadata):
    image = scan.field(channel).astype(np.float32)
    image = client.destagger(metadata, image)
    return image

def enhance_image_contrast(image, low_percent=1, high_percent=93.3, brightness_boost=1.1):
    # 排除零值（通常是无效数据）
    non_zero_mask = image > 0
    if np.sum(non_zero_mask) > 0:
        valid_pixels = image[non_zero_mask]

        # 计算有效值的范围（排除极端值）
        min_val = np.percentile(valid_pixels, low_percent)
        max_val = np.percentile(valid_pixels, high_percent)

        # 应用线性拉伸
        if max_val > min_val:
            normalized = np.zeros_like(image)
            normalized[non_zero_mask] = 255 * (valid_pixels - min_val) / (max_val - min_val)

            # # 增加亮度
            # normalized = normalized * brightness_boost
            normalized = np.clip(normalized, 0, 255)

            # # 添加降噪：双边滤波保留边缘
            # # normalized = cv2.bilateralFilter(normalized.astype(np.uint8), d=5, sigmaColor=50, sigmaSpace=50)
            # normalized = cv2.medianBlur(normalized.astype(np.uint8), 5)

            return normalized

    return image


def run(output_dir, save_combined=True, apply_contrast=True, brightness_boost=2):
    fields = [ChanField.RANGE, ChanField.REFLECTIVITY, ChanField.NEAR_IR]
    field_names = ["range", "reflectivity", "near_ir"]

    scan_source = open_source(pcap_file_path, sensor_idx=0, cycle=False)

    frame_count = 0

    for scan in scan_source:
        print(f"\r处理第 {frame_count}帧...", end="")

        # 处理扫描
        start = time.time()
        raw_images = [None] * len(fields)  # 存储原始图像
        processed_images = [None] * len(fields)  # 存储处理后的图像

        # 首先获取所有原始图像
        for i, field in enumerate(fields):
            # 获取原始图像
            image = get_frame_from_scan(scan, field, scan_source.metadata)
            raw_images[i] = image.copy()

        # 处理每个图像通道
        for i, field in enumerate(fields):
            image = raw_images[i].copy()

            if field == ChanField.NEAR_IR and (apply_contrast):
                if apply_contrast:
                    image = enhance_image_contrast(image, brightness_boost=brightness_boost)

            # 存储处理后的图像
            processed_images[i] = image.copy()

        # # 保存各个通道图像
        # for i, image in enumerate(processed_images):
        #     frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}_{field_names[i]}.png")
        #     cv2.imwrite(frame_filename, image)

        # 保存三通道合并图像（如果需要）
        if save_combined:
            # 创建RGB合成图像（使用NEAR_IR, SIGNAL和RANGE）
            combined_image = np.zeros((scan_height, scan_width, 3), dtype=np.uint8)
            # R通道 - NEAR_IR
            near_ir_scaled = processed_images[2]
            combined_image[:, :, 0] = near_ir_scaled.astype(np.uint8)
            # G通道 - REFLECTIVITY
            combined_image[:, :, 1] = near_ir_scaled.astype(np.uint8)
            # B通道 - RANGE（可能需要缩放）
            combined_image[:, :, 2] = near_ir_scaled.astype(np.uint8)

            # 保存合并图像
            combined_filename = os.path.join(output_dir, f"frame_{frame_count:05d}_combined.png")
            cv2.imwrite(combined_filename, combined_image)

        end = time.time()
        sleep_period = 1.0 / scan_fps - (end - start)
        if sleep_period > 0:
            time.sleep(sleep_period)

        # 在处理第一帧后打印图像信息
        if frame_count == 0:
            for i, field_name in enumerate(field_names):
                print(f"{field_name} 图像形状: {raw_images[i].shape}")
                print(f"{field_name} 最小值: {np.min(raw_images[i])}, 最大值: {np.max(raw_images[i])}")

                print(f"combined_{field_name} 图像形状: {combined_image[:, :, 2-i].shape}")
                print(f"combined_{field_name} 最小值: {np.min(combined_image[:, :, 2-i])}, 最大值: {np.max(combined_image[:, :, 2-i])}")

        frame_count += 1

    print(f"\n处理完成，共 {frame_count} 帧图像")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Generate Ouster LiDAR images from pcap file')
    parser.add_argument('--output', type=str, default='output_frames',
                        help='Output directory for frame images (default: output_frames)')
    parser.add_argument('--combined', action='store_true',
                        help='Save combined RGB image using different channels')
    parser.add_argument('--apply_contrast', action='store_true',
                        help='Apply image contrast enhancement')
    # parser.add_argument('--brightness', type=float, default=1.70,
    parser.add_argument('--brightness', type=float, default=2,
                        help='Brightness boost factor (1.0-3.0, default: 3.0)')
    # 30_02: python ouster_images.py --combined --apply_contrast --brightness 1.7
    # enhance_image_contrast(image, low_percent=0.5, high_percent=99.5, brightness_boost=1.1)
    # parser.add_argument('--pcap', type=str,
    #                     default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_02/Left/20250124_1300_OS-1-128_122211001778-002_split_30_02.pcap",
    #                     help='Path to pcap file')
    # parser.add_argument('--metadata', type=str,
    #                     default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_02/Left/20250124_1300_OS-1-128_122211001778-002_split_30_02.json",
    #                     help='Path to metadata JSON file')
    # bus 30_01:  python ouster_images.py --combined --apply_contrast --brightness 2
    # enhance_image_contrast(image, low_percent=1, high_percent=99, brightness_boost=1.1)
    # parser.add_argument('--pcap', type=str,
    #                     default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.pcap",
    #                     help='Path to pcap file')
    # parser.add_argument('--metadata', type=str,
    #                     default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.json",
    #                     help='Path to metadata JSON file')
    # # bus30_03: python ouster_images.py - -combined - -apply_contrast
    # parser.add_argument('--pcap', type=str,
    #                                         default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_03/left/20250124_1310_OS-1-128_122211001778-003.pcap",
    #                                         help='Path to pcap file')
    # parser.add_argument('--metadata', type=str,
    #                                         default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_03/left/20250124_1310_OS-1-128_122211001778-003.json",
    #                                         help='Path to metadata JSON file')
    # # # # bus31_01: python ouster_images.py - -combined - -apply_contrast
    # parser.add_argument('--pcap', type=str,
    #                     default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/31_01/left/20250124_1432_OS-1-128_122211001778.pcap",
    #                     help='Path to pcap file')
    # parser.add_argument('--metadata', type=str,
    #                     default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/31_01/left/20250124_1432_OS-1-128_122211001778.json",
    #                     help='Path to metadata JSON file')
    # # boston: python ouster_images.py - -combined - -apply_contrast
    # parser.add_argument('--pcap', type=str,
    #                     default="/media/yanan/MA2023-2/Ouster_LiDAR/Boston/scence_1/OS-1-128_122426001161_1024x20_20250918_074251454060.pcap",
    #                     help='Path to pcap file')
    # parser.add_argument('--metadata', type=str,
    #                     default="/media/yanan/MA2023-2/Ouster_LiDAR/Boston/scence_1/OS-1-128_122426001161_1024x20_20250918_074251312008.json",
    #                     help='Path to metadata JSON file')

    # # # bus38_02: python ouster_images.py - -combined - -apply_contrast
    parser.add_argument('--pcap', type=str,
                        default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/38_02/left/20250124_1235_OS-1-128_122211001778-002.pcap",
                        help='Path to pcap file')
    parser.add_argument('--metadata', type=str,
                        default="/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/38_02/left/20250124_1235_OS-1-128_122211001778-002.json",
                        help='Path to metadata JSON file')

    # 解析命令行参数
    args = parser.parse_args()

    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 打开pcap文件
    pcap_file_path = args.pcap
    metadata_path = args.metadata

    # 读取元数据
    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())

    # 获取扫描尺寸和帧率
    scan_source = open_source(pcap_file_path, sensor_idx=0, cycle=False)
    scan_width, scan_height, scan_fps = get_scan_size_and_fps(scan_source.metadata)

    # 使用XYZ查找表
    xyzlut = client.XYZLut(scan_source.metadata)

    print(f"开始处理 {pcap_file_path}")
    print(f"输出目录: {output_dir}")
    print(f"图像尺寸: {scan_width}x{scan_height}")
    print(f"应用图像对比度增强: {'是' if args.apply_contrast else '否'}")
    print(f"生成合并图像: {'是' if args.combined else '否'}")

    # 运行主程序
    run(output_dir,
        save_combined=args.combined,
        apply_contrast=args.apply_contrast,
        brightness_boost=args.brightness
        )

    scan_source.close()
    cv2.destroyAllWindows()