"""
从标注JSON中提取Label信息 - 完整版
功能：
1. 提取3D box用于OpenPCDet训练
2. 生成YOLO格式txt用于图像检测训练
3. 从JSON读取难度等级

使用方法：
python extract_labels_from_json.py
"""

import json
import numpy as np
import os
from pathlib import Path
from ouster.sdk import client

# ========== 配置区域（修改这里） ==========
# DATASETS = [
#     {
#         'name': '30_01',
#         'prefix': 'bus_30_01_',
#         'json_path': '/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/All_Route_30_01_PointCloud-bus_30_01.json',
#         'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.json',
#     },
#     {
#         'name': '30_02',
#         'prefix': 'bus_30_02_',
#         'json_path': '/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/All_Route_30_02_PointCloud-bus_30_02.json',
#         'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_02/Left/20250124_1300_OS-1-128_122211001778-002_split_30_02.json',
#     },
# ]
# DATASETS = [
#     {
#         'name': '30_01',
#         'prefix': 'bus_30_01_',
#         'json_path': '/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/All_Route_30_01_PointCloud-bus_30_01.json',
#         'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.json',
#     },
#     {
#         'name': '30_02',
#         'prefix': 'bus_30_02_',
#         'json_path': '/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/All_Route_30_02_PointCloud-bus_30_02.json',
#         'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_02/Left/20250124_1300_OS-1-128_122211001778-002_split_30_02.json',
#     },
#     {
#         'name': '30_03',
#         'prefix': 'bus_30_03_',
#         'json_path': '/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/All_Route_30_03-bus_30_03.json',
#         'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_03/left/20250124_1310_OS-1-128_122211001778-003.json',
#     },
#     {
#         'name': '31_01',
#         'prefix': 'bus_31_01_',
#         'json_path': '/home/yanan/Downloads/projects/multimodal_detection/data/dataset/label_json/All_Route_31_01_PointCloud-bus_31_01.json',
#         'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/31_01/left/20250124_1432_OS-1-128_122211001778.json',
#     },
# ]

DATASETS = [
    {
        'name': '30_01',
        'prefix': 'bus_30_01_',
        'json_path': '/home/yanan/Downloads/All_Route_30_01_PointCloud-Bus_30_01.json',
        'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.json',
    },
    {
        'name': '30_02',
        'prefix': 'bus_30_02_',
        'json_path': '/home/yanan/Downloads/All_Route_30_02_PointCloud-Bus_30_02.json',
        'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_02/Left/20250124_1300_OS-1-128_122211001778-002_split_30_02.json',
    },
    {
        'name': '30_03',
        'prefix': 'bus_30_03_',
        'json_path': '/home/yanan/Downloads/All_Route_30_03-Bus_30_03.json',
        'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_03/left/20250124_1310_OS-1-128_122211001778-003.json',
    },
    {
        'name': '31_01',
        'prefix': 'bus_31_01_',
        'json_path': '/home/yanan/Downloads/All_Route_31_01_PointCloud-Bus_31_01.json',
        'metadata_path': '/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/31_01/left/20250124_1432_OS-1-128_122211001778.json',
    },
]

# 输出目录
OUTPUT_LABEL_DIR = '/home/yanan/Downloads/projects/multimodal_detection/data/test/label2'
OUTPUT_YOLO_DIR = '/home/yanan/Downloads/projects/multimodal_detection/data/test/labels2'  # YOLO txt输出目录
# # 输出目录
# OUTPUT_LABEL_DIR = '/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/label2'
# OUTPUT_YOLO_DIR = '/home/yanan/Downloads/projects/multimodal_detection/data/Bus/train/labels2'  # YOLO txt输出目录

# 类别映射
CATEGORY_MAPPING = {
    'car': 'Car',
    'bus': 'Bus',
    'person': 'Pedestrian',
    'semi-truck': 'SemiTruck',
    'pickup truck': 'PickupTruck',
    'medium-sized truck': 'MediumTruck',
    'cyclist': 'Cyclist',
    'bicycle': 'Bicycle',
    'motorcycle': 'Motorcycle',
    'towed object': 'TowedObject',
    'semi truck': 'SemiTruck',
    'pickup-truck': 'PickupTruck',
}

# 类别ID映射（用于YOLO）
CATEGORY_TO_ID = {
    'Car': 0,
    'Bus': 1,
    'Pedestrian': 2,
    'SemiTruck': 3,
    'PickupTruck': 4,
    'MediumTruck': 5,
    'Cyclist': 6,
    'Bicycle': 7,
    'Motorcycle': 8,
    'TowedObject': 9,
    # 'semi truck': 3,
    # 'pickup-truck': 4,
}

# 难度映射
DEGREE_TO_CONFIDENCE = {
    'easy': 3,
    'medium': 2,
    'hard': 1
}

# 图像尺寸
IMG_WIDTH = 1024
IMG_HEIGHT = 128


# ==========================================


def load_metadata(metadata_path):
    """加载Ouster传感器元数据"""
    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())
    return metadata


def point_to_pix(metadata, x, y, z):
    """将3D点投影到2D图像坐标"""
    list128 = metadata.beam_altitude_angles
    l = np.sqrt(x * x + y * y + z * z)

    if l == 0:
        return [0, 0]

    x, y, z = x / l, y / l, z / l
    sita128 = np.arcsin(z) * 180 / np.pi
    sita2048 = np.arctan(y / x) * 180 / np.pi

    num2048 = 0
    if x < 0 and y > 0:
        num2048 = (-sita2048) * 2048 / 360
    elif x > 0 and y > 0:
        num2048 = (180 - sita2048) * 2048 / 360
    elif x > 0 and y < 0:
        num2048 = (180 - sita2048) * 2048 / 360
    else:
        num2048 = (360 - sita2048) * 2048 / 360

    num128 = 0
    min_dist = 2000

    for ind in range(128):
        dist = np.sqrt((sita128 - list128[ind]) * (sita128 - list128[ind]))
        if dist < min_dist:
            min_dist = dist
            num128 = ind

    if sita128 > list128[0]:
        num128 = 0 + (sita128 - list128[0]) / (list128[0] - list128[127]) * 128
    if sita128 < list128[127]:
        num128 = 127 - (list128[0] - sita128) / (list128[0] - list128[127]) * 128

    return [int(num2048 / 2), int(num128)]


def get_box_corners(center, size, rotation):
    """计算3D box的8个角点"""
    cx, cy, cz = center
    l, w, h = size
    rot = rotation

    corners = np.array([
        [cx - l / 2, cy - w / 2, cz - h / 2],
        [cx - l / 2, cy + w / 2, cz - h / 2],
        [cx + l / 2, cy + w / 2, cz - h / 2],
        [cx + l / 2, cy - w / 2, cz - h / 2],
        [cx - l / 2, cy - w / 2, cz + h / 2],
        [cx - l / 2, cy + w / 2, cz + h / 2],
        [cx + l / 2, cy + w / 2, cz + h / 2],
        [cx + l / 2, cy - w / 2, cz + h / 2],
    ])

    rotation_matrix = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot), np.cos(rot), 0],
        [0, 0, 1]
    ])

    rotated_corners = np.dot(corners - center, rotation_matrix.T) + center
    return rotated_corners


def project_to_image(corners, metadata):
    """将8个3D角点投影到2D图像"""
    corners_pix = []
    for corner in corners:
        x, y, z = corner
        pix = point_to_pix(metadata, x, y, z)
        corners_pix.append(pix)
    return np.array(corners_pix)


def compute_2d_bbox(projected_corners):
    """
    从投影的角点计算2D bounding box (轴对齐矩形)
    返回 [x_min, y_min, x_max, y_max]
    """
    # 过滤无效点
    valid_mask = projected_corners[:, 0] != 0
    if not np.any(valid_mask):
        return None

    valid_corners = projected_corners[valid_mask]

    # 处理360度图像的边界情况
    if np.max(valid_corners[:, 0]) > 728 and np.min(valid_corners[:, 0]) < 256:
        valid_corners[:, 0][valid_corners[:, 0] < 256] = 1024

    # 🔥 简化版：直接取min/max（不需要cv2.minAreaRect的复杂处理）
    x_min = max(0, min(int(np.min(valid_corners[:, 0])), IMG_WIDTH - 1))
    y_min = max(0, min(int(np.min(valid_corners[:, 1])), IMG_HEIGHT - 1))
    x_max = max(0, min(int(np.max(valid_corners[:, 0])), IMG_WIDTH - 1))
    y_max = max(0, min(int(np.max(valid_corners[:, 1])), IMG_HEIGHT - 1))

    return [x_min, y_min, x_max, y_max]


def bbox_to_yolo_format(bbox_2d, class_id):
    """
    将2D bbox转换为YOLO格式
    YOLO格式: class_id center_x center_y width height (归一化到0-1)
    """
    if bbox_2d is None:
        return None

    x_min, y_min, x_max, y_max = bbox_2d

    # 归一化
    center_x = ((x_min + x_max) / 2) / IMG_WIDTH
    center_y = ((y_min + y_max) / 2) / IMG_HEIGHT
    width = (x_max - x_min) / IMG_WIDTH
    height = (y_max - y_min) / IMG_HEIGHT

    # 确保在[0, 1]范围内
    center_x = max(0, min(center_x, 1))
    center_y = max(0, min(center_y, 1))
    width = max(0, min(width, 1))
    height = max(0, min(height, 1))

    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"


def read_boxes_from_json(json_path):
    """
    从JSON文件读取所有帧的boxes
    返回: {frame_num: [boxes]}, category_map
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 读取类别映射
    categories = data.get('dataset', {}).get('task_attributes', {}).get('categories', [])
    category_map = {cat['id']: cat['name'] for cat in categories}

    all_frames_boxes = {}

    samples = data.get('dataset', {}).get('samples', [])
    for sample in samples:
        frame_name = sample.get('name', '')

        # 提取帧编号
        if 'pcd_out_' in frame_name:
            frame_num = frame_name.split('pcd_out_')[-1].replace('.bin', '').lstrip('0') or '0'
        else:
            frame_num = frame_name.replace('.bin', '')

        boxes = []
        labels = sample.get('labels') or {}
        ground_truth = labels.get('ground-truth') or {}
        attributes = ground_truth.get('attributes') or {}
        annotations = attributes.get('annotations', [])

        for annotation in annotations:
            if annotation.get('type') == 'cuboid':
                position = annotation.get('position', {})
                dimensions = annotation.get('dimensions', {})
                yaw = annotation.get('yaw', 0)
                category_id = annotation.get('category_id', 0)

                # 🔥 新增：读取难度等级
                ann_attributes = annotation.get('attributes', {})
                degree = ann_attributes.get('degree', 'easy')  # 默认easy
                confidence = DEGREE_TO_CONFIDENCE.get(degree, 3)

                box = {
                    'center': [position.get('x', 0), position.get('y', 0), position.get('z', 0)],
                    'size': [dimensions.get('x', 0), dimensions.get('y', 0), dimensions.get('z', 0)],
                    'rotation': yaw,
                    'category_id': category_id,
                    'category_name': category_map.get(category_id, 'unknown'),
                    'confidence': confidence  # 🔥 从JSON读取
                }
                boxes.append(box)

        all_frames_boxes[frame_num] = boxes

    return all_frames_boxes, category_map


def convert_to_training_format(box, category_name, bbox_2d):
    """
    将box转换为OpenPCDet训练格式
    """
    # 映射类别名
    mapped_name = CATEGORY_MAPPING.get(category_name.lower(), category_name)

    box_dict = {
        'type': f"TYPE_{mapped_name}",
        'position3d': {
            'x': box['center'][0],
            'y': box['center'][1],
            'z': box['center'][2]
        },
        'size3d': {
            'x': box['size'][0],
            'y': box['size'][1],
            'z': box['size'][2]
        },
        'heading': box['rotation'],
        'tag': {
            'confidence': str(box['confidence'])  # 🔥 使用从JSON读取的confidence
        }
    }

    # 添加2D bbox信息
    if bbox_2d is not None:
        box_dict['bbox_2d'] = {
            'x_min': bbox_2d[0],
            'y_min': bbox_2d[1],
            'x_max': bbox_2d[2],
            'y_max': bbox_2d[3]
        }

    return box_dict


def process_dataset(dataset_config):
    """处理单个数据集"""
    name = dataset_config['name']
    prefix = dataset_config['prefix']
    json_path = dataset_config['json_path']
    metadata_path = dataset_config['metadata_path']

    print(f"\n{'=' * 60}")
    print(f"处理数据集: {name}")
    print(f"JSON: {json_path}")
    print(f"{'=' * 60}\n")

    # 加载元数据
    metadata = load_metadata(metadata_path)

    # 读取boxes
    all_frames_boxes, category_map = read_boxes_from_json(json_path)
    print(f"总帧数: {len(all_frames_boxes)}")
    print(f"类别: {list(category_map.values())}")
    empty_frames = [f for f, b in all_frames_boxes.items() if not b]
    print("JSON里空标注帧数:", len(empty_frames))
    print("前50个空标注帧:", empty_frames[:50])

    # 创建输出目录
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_YOLO_DIR, exist_ok=True)

    total_boxes = 0
    empty_label_written = 0

    for frame_num, boxes in sorted(all_frames_boxes.items()):
        # if not boxes:
        #     continue

        output_boxes = []
        yolo_lines = []

        # ====== 有boxes就正常处理；没有boxes则保持空列表 ======
        if boxes:
            for box in boxes:
                # 计算8个角点
                corners = get_box_corners(box['center'], box['size'], box['rotation'])

                # 投影到2D
                projected_corners = project_to_image(corners, metadata)

                # 计算2D bbox
                bbox_2d = compute_2d_bbox(projected_corners)

                # 转换为OpenPCDet格式
                box_dict = convert_to_training_format(
                    box,
                    box['category_name'],
                    bbox_2d
                )
                output_boxes.append(box_dict)

                # 🔥 新增：生成YOLO格式
                if bbox_2d is not None:
                    mapped_name = CATEGORY_MAPPING.get(box['category_name'].lower(), box['category_name'])
                    class_id = CATEGORY_TO_ID.get(mapped_name, 0)
                    yolo_line = bbox_to_yolo_format(bbox_2d, class_id)
                    if yolo_line:
                        yolo_lines.append(yolo_line)

                total_boxes += 1
        else:
            empty_label_written += 1

        # 保存OpenPCDet格式的json
        output_filename = f"{prefix}pcd_out_{frame_num.zfill(6)}.json"
        output_path = os.path.join(OUTPUT_LABEL_DIR, output_filename)
        with open(output_path, 'w') as f:
            json.dump(output_boxes, f, indent=4)

        # 🔥 新增：保存YOLO格式的txt
        yolo_filename = f"{prefix}frame_{frame_num.zfill(5)}_combined.txt"
        yolo_path = os.path.join(OUTPUT_YOLO_DIR, yolo_filename)
        with open(yolo_path, 'w') as f:
            if yolo_lines:
                f.writelines(yolo_lines)
            else:
                # 空标注：写空文件即可
                pass
            # f.writelines(yolo_lines)

    print(f"✓ 完成 {len(all_frames_boxes)} 帧，共 {total_boxes} 个boxes")
    print(f"  - 空标注帧（已生成空label文件）: {empty_label_written}")
    print(f"  - OpenPCDet labels: {OUTPUT_LABEL_DIR}")
    print(f"  - YOLO labels: {OUTPUT_YOLO_DIR}")

    return len(all_frames_boxes), total_boxes


def main():
    print("=" * 60)
    print("从JSON提取Label信息（包含YOLO格式）")
    print("=" * 60)

    total_frames = 0
    total_boxes = 0

    for dataset_config in DATASETS:
        frames, boxes = process_dataset(dataset_config)
        total_frames += frames
        total_boxes += boxes

    print("\n" + "=" * 60)
    print("提取完成！")
    print("=" * 60)
    print(f"总帧数: {total_frames}")
    print(f"总boxes: {total_boxes}")
    print(f"\n输出目录:")
    print(f"  - OpenPCDet: {OUTPUT_LABEL_DIR}")
    print(f"  - YOLO: {OUTPUT_YOLO_DIR}")


if __name__ == '__main__':
    main()