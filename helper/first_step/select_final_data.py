"""
超简洁版 - 单数据集筛选
直接修改下面的配置路径即可使用
"""

import json
import os
import shutil

# ========== 配置（改这里）==========
# JSON_FILE = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/All_Route_30_01_PointCloud-bus_30_01.json"
# BIN_DIR = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/bin"
# PNG_DIR = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/images/30_01"
# OUTPUT_DIR = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_01"  # 输出到这个文件夹
# ==================================
# JSON_FILE = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/All_Route_31_01_PointCloud-bus_31_01.json"
# BIN_DIR = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/31_01/bin"
# PNG_DIR = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/31_01/images/31_01"
# OUTPUT_DIR = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/31_01"  # 输出到这个文件夹

JSON_FILE = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/Boston_01_PointCloud-Boston_01.json"
BIN_DIR = "/media/yanan/MA2023-2/Ouster_LiDAR/Boston/label_data/boston_01/Boston_bin/1_0742"
PNG_DIR = "/media/yanan/MA2023-2/Ouster_LiDAR/Boston/label_data/boston_01/images"
OUTPUT_DIR = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/Boston_01"
# 读取JSON获取有效帧
print("读取JSON...")
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

valid_frames = set()
for sample in data['dataset']['samples']:
    # 提取帧号: "pcd_out_000007.bin" -> "00007"
    name = sample['name'].replace('.bin', '')
    frame_num = name.split('pcd_out_')[-1].lstrip('0') or '0'
    valid_frames.add(frame_num)

print(f"✓ JSON中有 {len(valid_frames)} 个有效帧\n")

# 筛选bin文件
if os.path.exists(BIN_DIR):
    print(f"筛选bin文件...")
    bin_output = os.path.join(OUTPUT_DIR, 'bin')
    os.makedirs(bin_output, exist_ok=True)

    bin_count = 0
    for filename in os.listdir(BIN_DIR):
        if filename.endswith('.bin'):
            # 提取帧号
            frame_num = filename.replace('.bin', '').split('pcd_out_')[-1].lstrip('0') or '0'

            if frame_num in valid_frames:
                src = os.path.join(BIN_DIR, filename)
                PREFIX = 'boston_01_'  # 配置区添加
                new_name = PREFIX + filename
                # dst = os.path.join(bin_output, filename)
                dst = os.path.join(bin_output, new_name)
                shutil.copy2(src, dst)
                bin_count += 1

    print(f"✓ 复制了 {bin_count} 个bin文件\n")

# 筛选png文件
if os.path.exists(PNG_DIR):
    print(f"筛选png文件...")
    png_output = os.path.join(OUTPUT_DIR, 'images')
    os.makedirs(png_output, exist_ok=True)

    png_count = 0
    for filename in os.listdir(PNG_DIR):
        if filename.endswith('.png'):
            # 提取帧号: "frame_00007_combined.png" -> "00007"
            frame_num = filename.split('_')[1].lstrip('0') or '0'

            if frame_num in valid_frames:
                src = os.path.join(PNG_DIR, filename)
                PREFIX = 'boston_01_'  # 配置区添加
                new_name = PREFIX + filename
                dst = os.path.join(png_output, new_name)
                shutil.copy2(src, dst)
                png_count += 1

    print(f"✓ 复制了 {png_count} 个png文件\n")

print("=" * 50)
print("完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 50)