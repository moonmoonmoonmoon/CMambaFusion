import json
import numpy as np
import cv2
import os
from ouster.sdk import client

# ==============================================================================
# 配置区域 - 修改这里
# ==============================================================================
metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.json"
# metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_02/Left/20250124_1300_OS-1-128_122211001778-002_split_30_02.json"
# metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_03/left/20250124_1310_OS-1-128_122211001778-003.json"
# metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/31_01/left/20250124_1432_OS-1-128_122211001778.json"
# metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/Boston/raw_data/scence_1/OS-1-128_122426001161_1024x20_20250918_074251312008.json"

# # Boston sensor 安装方向偏转了90度
# SHIFT_DEGREES = 90.0
# Bus数据集安装方向偏转了180度
SHIFT_DEGREES = 180.0
# ==============================================================================

with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())


def read_boxes_from_new_json(file_path):
    """
    从JSON文件中读取所有帧的boxes
    返回: dict {frame_name: [boxes]}, category_map
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    categories = data.get('dataset', {}).get('task_attributes', {}).get('categories', [])
    category_map = {cat['id']: cat['name'] for cat in categories}
    all_frames_boxes = {}

    samples = data.get('dataset', {}).get('samples', [])
    for sample in samples:
        frame_name = sample.get('name', '')
        frame_num = frame_name.replace('pcd_out_0', '').replace('.bin', '')
        # frame_num = frame_name.replace('bus_30_02_pcd_out_0', '').replace('.bin', '')

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

                box = {
                    'center': [position.get('x', 0), position.get('y', 0), position.get('z', 0)],
                    'size': [dimensions.get('x', 0), dimensions.get('y', 0), dimensions.get('z', 0)],
                    'rotation': yaw,
                    'category_id': category_id
                }
                boxes.append(box)

        all_frames_boxes[frame_num] = boxes

    return all_frames_boxes, category_map


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
        [np.sin(rot),  np.cos(rot), 0],
        [0,            0,           1]
    ])
    rotated_corners = np.dot(corners - center, rotation_matrix.T) + center

    return rotated_corners


def point_To_pix(metadataT, x, y, z, img_w, img_h):
    """
    将3D点投影到图像像素坐标。
    改动说明（相比bus版本）：
      1. 新增 img_w, img_h 参数，支持任意分辨率图像
      2. 水平角度改用 arctan2，消除手动象限判断误差
      3. 加入 SHIFT_DEGREES 补偿 Boston sensor 的安装偏转
    """
    list128 = metadataT.beam_altitude_angles

    l = np.sqrt(x * x + y * y + z * z)
    if l == 0:
        return [0, 0]
        # return [-1, -1]

    # --- 垂直方向 ---
    sita128 = np.arcsin(z / l) * 180 / np.pi
    row_idx = int(np.argmin(np.abs(np.array(list128) - sita128)))
    target_v = int((row_idx / 128.0) * img_h)

    # --- 水平方向（改动：用 arctan2 替代手动象限判断）---
    target_angle = -np.arctan2(y, x) * 180 / np.pi
    if target_angle < 0:
        target_angle += 360

    # 补偿 sensor 安装偏转角度（bus数据集 SHIFT_DEGREES=0 则无影响）
    target_angle = (target_angle + SHIFT_DEGREES) % 360

    # 映射到图像宽度（改动：用 img_w 替代硬编码的 1024）
    target_u = int((target_angle / 360.0) * img_w)

    return [int(target_u), int(target_v)]


def project_to_image2(corners, img_w, img_h):
    """将8个角点投影到图像像素坐标（改动：新增 img_w, img_h 参数）"""
    cornersPix = []
    for ind in range(8):
        x, y, z = corners[ind]
        cornersPix.append(point_To_pix(metadata, x, y, z, img_w, img_h))
    return np.array(cornersPix)


# 和我之前直接保留右边box的方法一样
def draw_box(image, projected_corners, category_name="", frame_num='1'):
    h, w = image.shape[:2]

    valid_mask = projected_corners[:, 0] != 0
    if not np.any(valid_mask):
        return
    valid_corners = projected_corners[valid_mask].astype(np.float32)

    # 和标签生成代码完全一致的跨边界处理：左边的点推到w，clip后只剩右侧
    if np.max(valid_corners[:, 0]) > w * 0.71 and np.min(valid_corners[:, 0]) < w * 0.25:
        valid_corners[:, 0][valid_corners[:, 0] < w * 0.25] = w  # 推到w，clip后变成w-1

    x_min = max(0, min(int(np.min(valid_corners[:, 0])), w - 1))
    y_min = max(0, min(int(np.min(valid_corners[:, 1])), h - 1))
    x_max = max(0, min(int(np.max(valid_corners[:, 0])), w - 1))
    y_max = max(0, min(int(np.max(valid_corners[:, 1])), h - 1))

    if x_min >= x_max or y_min >= y_max:
        return

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)

    if category_name:
        label_pos = (x_min, y_min - 5 if y_min > 10 else 15)
        cv2.putText(image, category_name, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)



# # 用最小外接矩阵保留最右边的box，如果不存在，则不要。
# def draw_box(image, projected_corners, category_name="", frame_num='1'):
#     """在图像上绘制2D旋转最小外接矩形，处理跨边界情况"""
#     h, w = image.shape[:2]
#
#     valid_mask = projected_corners[:, 0] != 0
#     if not np.any(valid_mask):
#         return
#     valid_corners = projected_corners[valid_mask].astype(np.float32)
#
#     if np.max(valid_corners[:, 0]) > w * 0.71 and np.min(valid_corners[:, 0]) < w * 0.25:
#         valid_corners[:, 0][valid_corners[:, 0] < w * 0.25] = w
#
#     x_min = max(0, min(int(np.min(valid_corners[:, 0])), w - 1))
#     y_min = max(0, min(int(np.min(valid_corners[:, 1])), h - 1))
#     x_max = max(0, min(int(np.max(valid_corners[:, 0])), w - 1))
#     y_max = max(0, min(int(np.max(valid_corners[:, 1])), h - 1))
#
#     if x_min >= x_max or y_min >= y_max:
#         return
#
#     if x_min + 11 >= w-1 or y_min + 4 >= h-1:
#         return
#
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
#
#     if category_name:
#         label_pos = (x_min, y_min - 5 if y_min > 10 else 15)
#         cv2.putText(image, category_name, label_pos,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

# # 用最小外接矩阵保留左右两边的box，适用于可视化。
# def draw_box(image, projected_corners, category_name="", frame_num='1'):
#     """在图像上绘制2D旋转最小外接矩形，处理跨边界情况"""
#     h, w = image.shape[:2]
#
#     valid_mask = projected_corners[:, 0] != 0
#     # valid_mask = projected_corners[:, 0] != -1
#     if not np.any(valid_mask):
#         return
#     valid_corners = projected_corners[valid_mask].astype(np.float32)
#
#     # 跨边界检测：若点同时出现在图像最左和最右，说明box跨越了全景图拼接缝
#     xs = valid_corners[:, 0]
#     # 原来硬编码 728/256，改成按比例自适应 img_w
#     is_split = (np.max(xs) > w * 0.71) and (np.min(xs) < w * 0.25)
#
#     if is_split:
#         # 把左侧的点推到右侧虚拟空间，让 minAreaRect 画出连续的框
#         valid_corners[:, 0][valid_corners[:, 0] < (w / 2)] += w
#
#     rect = cv2.minAreaRect(valid_corners)
#     box = cv2.boxPoints(rect)
#     box = box.astype(np.int32)
#
#     if is_split:
#         # 画左侧部分（虚拟坐标减回去）
#         box_left = box.copy()
#         box_left[:, 0] -= w
#         cv2.drawContours(image, [box_left], 0, (255, 255, 0), 1)
#         # 画右侧部分
#         cv2.drawContours(image, [box], 0, (255, 255, 0), 1)
#
#         label_x = int(min(box[:, 0]))
#         if label_x >= w:
#             label_x -= w
#         label_pos = (label_x, int(min(box[:, 1])) - 5)
#     else:
#         box = np.clip(box, [0, 0], [w - 1, h - 1])
#         cv2.drawContours(image, [box], 0, (255, 255, 0), 1)
#         label_pos = (int(min(box[:, 0])), int(min(box[:, 1])) - 5)
#
#     if category_name:
#         if label_pos[1] < 10:
#             label_pos = (label_pos[0], 15)
#         cv2.putText(image, category_name, label_pos,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)


def run():
    # ==========================================================================
    # 路径配置
    # ==========================================================================
    #     # 配置路径
    json_file_path = "/home/yanan/Downloads/All_Route_30_01_PointCloud-Bus_30_01.json"  # 修改为你的JSON文件路径
    input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_01/images"  # 修改为你的输入图像文件夹
    output_dir = "./output/label_image_box/30_01"
    #     # json_file_path = "/home/yanan/Downloads/All_Route_30_01_PointCloud-bus3001.json"  # 修改为你的JSON文件路径
    #     # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_01/images"  # 修改为你的输入图像文件夹
    #     # output_dir = "./output/label_image_box/3001"
    # json_file_path = "/home/yanan/Downloads/All_Route_30_02_PointCloud-Bus_30_02.json"  # 修改为你的JSON文件路径
    # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_02/images"  # 修改为你的输入图像文件夹
    # output_dir = "./output/label_image_box/30_02"
    # json_file_path = "/home/yanan/Downloads/All_Route_30_03-Bus_30_03.json"  # 修改为你的JSON文件路径
    # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_03/images"  # 修改为你的输入图像文件夹
    # output_dir = "./output/label_image_box/30_03"
    # json_file_path = "/home/yanan/Downloads/All_Route_31_01_PointCloud-Bus_31_01.json"
    # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/output/images/output_combined_images/upload/31_01_upload"
    # output_dir = "./output/label_image_box/31_01"
    # json_file_path = "/home/yanan/Downloads/Boston_01_PointCloud-Boston_01.json"
    # input_image_dir = "/media/yanan/MA2023-2/Ouster_LiDAR/Boston/label_data/boston_01/images"
    # output_dir = "./output/label_image_box/Boston_01"
    #
    field_names = ["combined"]
    # ==========================================================================

    os.makedirs(output_dir, exist_ok=True)

    print("读取JSON文件...")
    all_frames_boxes, category_map = read_boxes_from_new_json(json_file_path)
    print(f"总共有 {len(all_frames_boxes)} 帧数据")

    for frame_num, boxes in sorted(all_frames_boxes.items()):
        print(f"处理帧 {frame_num}...")

        images = []
        current_h, current_w = 128, 1024  # 默认尺寸，实际以读到的图像为准

        for field_name in field_names:
            image_path = os.path.join(input_image_dir, f"frame_{frame_num}_{field_name}.png")

            if not os.path.exists(image_path):
                # 模糊匹配文件名
                possible = [f for f in os.listdir(input_image_dir)
                            if frame_num in f and field_name in f]
                if possible:
                    image_path = os.path.join(input_image_dir, possible[0])
                else:
                    print(f"警告: 图像文件不存在 {image_path}")
                    continue

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"警告: 无法读取图像 {image_path}")
                continue

            # 记录实际图像尺寸（改动：传给投影函数使用）
            current_h, current_w = image.shape[:2]
            images.append(image)

        if not images:
            print(f"跳过帧 {frame_num}: 没有找到图像")
            continue

        if boxes:
            for box in boxes:
                center = box['center']
                size = box['size']
                rotation = box['rotation']
                category_id = box.get('category_id', 0)
                category_name = category_map.get(category_id, 'unknown')

                corners = get_box_corners(center, size, rotation)

                # 改动：传入实际图像尺寸
                projected_corners = project_to_image2(corners, current_w, current_h)

                for image in images:
                    draw_box(image, projected_corners, category_name, frame_num)

        for i, image in enumerate(images):
            output_filename = os.path.join(
                output_dir, f"frame_{frame_num}_{field_names[i]}_boxed.png"
            )
            cv2.imwrite(output_filename, image)

        print(f"帧 {frame_num} 处理完成，绘制了 {len(boxes)} 个框")

    print("所有帧处理完成！")


if __name__ == "__main__":
    run()


# 我本来对bus数据集的mapping box的代码
# import json
# import numpy as np
# import cv2
# import os
# from ouster.sdk import client
#
# # metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_01/Left/20250124_1250_OS-1-128_122211001778.json"
# # metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_02/Left/20250124_1300_OS-1-128_122211001778-002_split_30_02.json"
# # metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/30_03/left/20250124_1310_OS-1-128_122211001778-003.json"
# metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/31_01/left/20250124_1432_OS-1-128_122211001778.json"
# with open(metadata_path, 'r') as f:
#     metadata = client.SensorInfo(f.read())
#
#
# def read_boxes_from_new_json(file_path):
#     """
#     从新格式的JSON文件中读取所有帧的boxes
#     返回: dict {frame_name: [boxes]}
#     """
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#
#     # 读取类别映射
#     categories = data.get('dataset', {}).get('task_attributes', {}).get('categories', [])
#     category_map = {cat['id']: cat['name'] for cat in categories}
#     all_frames_boxes = {}
#
#     # 遍历所有samples
#     samples = data.get('dataset', {}).get('samples', [])
#     for sample in samples:
#         frame_name = sample.get('name', '')
#         # 提取帧编号，例如 "pcd_out_000007.bin" -> "00007"
#         frame_num = frame_name.replace('pcd_out_0', '').replace('.bin', '')
#         # frame_num = frame_name.replace('bus_30_02_pcd_out_0', '').replace('.bin', '')
#
#         boxes = []
#         # annotations = sample.get('labels', {}).get('ground-truth', {}).get('attributes', {}).get('annotations', [])
#         # 修改这里，处理None的情况
#         labels = sample.get('labels') or {}
#         ground_truth = labels.get('ground-truth') or {}
#         attributes = ground_truth.get('attributes') or {}
#         annotations = attributes.get('annotations', [])
#
#         for annotation in annotations:
#             if annotation.get('type') == 'cuboid':
#                 position = annotation.get('position', {})
#                 dimensions = annotation.get('dimensions', {})
#                 yaw = annotation.get('yaw', 0)
#                 category_id = annotation.get('category_id', 0)
#
#                 box = {
#                     'center': [position.get('x', 0), position.get('y', 0), position.get('z', 0)],
#                     'size': [dimensions.get('x', 0), dimensions.get('y', 0), dimensions.get('z', 0)],
#                     'rotation': yaw,
#                     'category_id': category_id
#                 }
#                 boxes.append(box)
#
#         all_frames_boxes[frame_num] = boxes
#
#     return all_frames_boxes, category_map
#
#
# # 计算box的八个角点
# def get_box_corners(center, size, rotation):
#     cx, cy, cz = center
#     l, w, h = size
#     rot = rotation
#
#     # 计算八个角点
#     corners = np.array([
#         [cx - l / 2, cy - w / 2, cz - h / 2],
#         [cx - l / 2, cy + w / 2, cz - h / 2],
#         [cx + l / 2, cy + w / 2, cz - h / 2],
#         [cx + l / 2, cy - w / 2, cz - h / 2],
#         [cx - l / 2, cy - w / 2, cz + h / 2],
#         [cx - l / 2, cy + w / 2, cz + h / 2],
#         [cx + l / 2, cy + w / 2, cz + h / 2],
#         [cx + l / 2, cy - w / 2, cz + h / 2],
#     ])
#
#     # 旋转角点
#     rotation_matrix = np.array([
#         [np.cos(rot), -np.sin(rot), 0],
#         [np.sin(rot), np.cos(rot), 0],
#         [0, 0, 1]
#     ])
#     rotated_corners = np.dot(corners - center, rotation_matrix.T) + center
#
#     return rotated_corners
#
# def point_To_pix(metadataT, x, y, z):
#     list128 = metadata.beam_altitude_angles
#     l = np.sqrt(x * x + y * y + z * z)
#     x, y, z = x / l, y / l, z / l
#     sita128 = np.arcsin(z) * 180 / np.pi
#     sita2048 = np.arctan(y / x) * 180 / np.pi
#     num2048 = 0
#     if x < 0 and y > 0:
#         num2048 = (-sita2048) * 2048 / 360
#     elif x > 0 and y > 0:
#         num2048 = (180 - sita2048) * 2048 / 360
#     elif x > 0 and y < 0:
#         num2048 = (180 - sita2048) * 2048 / 360
#     else:
#         num2048 = (360 - sita2048) * 2048 / 360
#     num128 = 0
#     l = 2000
#     # print('sita128',sita128)
#
#     for ind in range(128):
#         if l > np.sqrt((sita128 - list128[ind]) * (sita128 - list128[ind])):
#             l = np.sqrt((sita128 - list128[ind]) * (sita128 - list128[ind]))
#             num128 = ind
#
#     if sita128 > list128[0]:
#         num128 = 0 + (sita128 - list128[0]) / (list128[0] - list128[127]) * 128
#     if sita128 < list128[127]:
#         num128 = 127 - (list128[0] - sita128) / (list128[0] - list128[127]) * 128
#     # print('128',num128)
#
#     return [int(num2048 / 2), int(num128)]
#
#
# def project_to_image2(corners):
#     cornersPix = []
#     # print('corners',corners)
#     for ind in range(8):
#         x, y, z = corners[ind]
#         cornersPix.append(point_To_pix(metadata, x, y, z))
#     return np.array(cornersPix)
#
#
# # 在图像上绘制2D box（旋转最小外接矩形）
# def draw_box(image, projected_corners, category_name="", frame_num='1'):
#     # 过滤掉无效的点
#     valid_mask = projected_corners[:, 0] != 0
#     if not np.any(valid_mask):
#         return
#     valid_corners = projected_corners[valid_mask].astype(np.float32)
#     if max(valid_corners[:, 0]) > 728 and min(valid_corners[:, 0]) < 256:
#         valid_corners[:, 0][valid_corners[:, 0] < 256] = 1024
#     # # 如果box跨越了左右边界
#     # if max(valid_corners[:, 0]) > 728 and min(valid_corners[:, 0]) < 256:
#     #     left_points = valid_corners[:, 0][valid_corners[:, 0] < 256]
#     #     right_points = valid_corners[:, 0][valid_corners[:, 0] > 728]
#     #
#     #     if len(left_points) > 0 and len(right_points) > 0:
#     #         left_width = max(left_points)
#     #         right_width = 1023 - min(right_points)
#     #
#     #         # 左边占比 > 3/4，则保留左边，否则保留右边
#     #         if left_width / (left_width + right_width + 1e-6) > 0.75:
#     #             valid_corners[:, 0][valid_corners[:, 0] > 728] = 0
#     #         else:
#     #             valid_corners[:, 0][valid_corners[:, 0] < 256] = 1024
#
#     h, w = image.shape[:2]
#
#     # if frame_num == '02761':
#     #     print('p', valid_corners)
#     # 计算旋转最小外接矩形
#     rect = cv2.minAreaRect(valid_corners)
#     box = cv2.boxPoints(rect)  # 获取矩形的4个角点
#     # box[:, 0] = np.clip(box[:, 0], 0, w - 1)
#     # box[:, 1] = np.clip(box[:, 1], 0, h - 1)
#     # if frame_num == '02761':
#     #     print('b', box)
#
#     box = np.clip(box, [0, 0], [w - 1, h - 1])
#
#     # box = np.int0(box)  # 转换为整数
#     box_int = np.int0(box)
#
#     # 处理x坐标
#     x_indices = np.argsort(box_int[:, 0])
#     min_x_small = min(box_int[x_indices[0], 0], box_int[x_indices[1], 0])
#     box[x_indices[0], 0] = min_x_small
#     box[x_indices[1], 0] = min_x_small
#     min_x_large = min(box_int[x_indices[2], 0], box_int[x_indices[3], 0])
#     box[x_indices[2], 0] = min_x_large
#     box[x_indices[3], 0] = min_x_large
#
#     # 处理y坐标
#     y_indices = np.argsort(box_int[:, 1])
#     min_y_small = min(box_int[y_indices[0], 1], box_int[y_indices[1], 1])
#     box[y_indices[0], 1] = min_y_small
#     box[y_indices[1], 1] = min_y_small
#     min_y_large = min(box_int[y_indices[2], 1], box_int[y_indices[3], 1])
#     box[y_indices[2], 1] = min_y_large
#     box[y_indices[3], 1] = min_y_large
#
#     box = box.astype(np.int32)
#     # if frame_num == '00007':
#     #     print(box)
#
#     # 绘制矩形
#     cv2.drawContours(image, [box], 0, (255, 255, 0), 1)
#
#     # 添加类别标签（添加这部分）
#     if category_name:
#         # 在box的左上角显示类别名称
#         label_pos = (int(min(box[:, 0])), int(min(box[:, 1])) - 5)
#         cv2.putText(image, category_name, label_pos,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
#
# def run():
#     # 配置路径
#     # json_file_path = "/home/yanan/Downloads/All_Route_30_01_PointCloud-Bus_30_01.json"  # 修改为你的JSON文件路径
#     # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_01/images"  # 修改为你的输入图像文件夹
#     # output_dir = "./output/label_image_box/30_01"
#     # json_file_path = "/home/yanan/Downloads/All_Route_30_01_PointCloud-bus3001.json"  # 修改为你的JSON文件路径
#     # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_01/images"  # 修改为你的输入图像文件夹
#     # output_dir = "./output/label_image_box/3001"
#     # json_file_path = "/home/yanan/Downloads/All_Route_30_02_PointCloud-Bus_30_02.json"  # 修改为你的JSON文件路径
#     # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_02/images"  # 修改为你的输入图像文件夹
#     # output_dir = "./output/label_image_box/30_02"
#     # json_file_path = "/home/yanan/Downloads/All_Route_30_03-Bus_30_03.json"  # 修改为你的JSON文件路径
#     # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/30_03/images"  # 修改为你的输入图像文件夹
#     # output_dir = "./output/label_image_box/30_03"
#     json_file_path = "/home/yanan/Downloads/All_Route_31_01_PointCloud-Bus_31_01.json"  # 修改为你的JSON文件路径
#     # input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/output/images/output_combined_images/upload/31_01_upload"  # 修改为你的输入图像文件夹
#     input_image_dir = "/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/31_01/images"
#     output_dir = "./output/label_image_box/31_01"
#
#     field_names = ["combined"]
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 读取所有帧的boxes
#     print("读取JSON文件...")
#     all_frames_boxes, category_map = read_boxes_from_new_json(json_file_path)
#     print(f"总共有 {len(all_frames_boxes)} 帧数据")
#
#     # 遍历所有帧
#     for frame_num, boxes in sorted(all_frames_boxes.items()):
#         print(f"处理帧 {frame_num}...")
#
#         # 读取该帧的所有图像
#         images = []
#         for field_name in field_names:
#             # 假设图像文件命名格式为: frame_00007_range.png
#             image_path = os.path.join(input_image_dir, f"bus_31_01_frame_{frame_num}_{field_name}.png")
#
#             if not os.path.exists(image_path):
#                 print(f"警告: 图像文件不存在 {image_path}")
#                 continue
#
#             image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#             if image is None:
#                 print(f"警告: 无法读取图像 {image_path}")
#                 continue
#
#             images.append(image)
#
#         # 如果没有成功读取图像，跳过该帧
#         if len(images) == 0:
#             print(f"跳过帧 {frame_num}: 没有找到图像")
#             continue
#
#         # 在所有图像上绘制boxes
#         if boxes:
#             for box in boxes:
#                 center = box['center']
#                 size = box['size']
#                 rotation = box['rotation']
#                 category_id = box.get('category_id', 0)  # 添加这行
#                 category_name = category_map.get(category_id, 'unknown')
#                 corners = get_box_corners(center, size, rotation)
#                 projected_corners = project_to_image2(corners)
#
#                 # 在每张图像上绘制box
#                 for image in images:
#                     draw_box(image, projected_corners, category_name, frame_num)
#
#         # 保存绘制后的图像
#         for i, image in enumerate(images):
#             output_filename = os.path.join(output_dir, f"frame_{frame_num}_{field_names[i]}_boxed.png")
#             cv2.imwrite(output_filename, image)
#
#             print(f"帧 {frame_num} 处理完成，绘制了 {len(boxes)} 个框")
#
#     print("所有帧处理完成！")
#
#
# if __name__ == "__main__":
#     run()


# # Amit本来对Boston数据集的mapping box的代码
# # Boston Mapping
# import json
# import numpy as np
# import cv2
# import os
# from ouster.sdk import client
#
# # ==========================================================================================
# # FINAL CONFIGURATION
# # ==========================================================================================
# # THIS IS NOW THE SINGLE SOURCE OF TRUTH FOR YOUR JSON FILE
# metadata_path = "/media/yanan/MA2023-2/Ouster_LiDAR/Boston/raw_data/scence_1/OS-1-128_122426001161_1024x20_20250918_074251312008.json"
#
# # Based on your test, the sensor is rotated 90 degrees.
# # We will apply this shift in DEGREES so it works for any image resolution (1024 or 2048).
# SHIFT_DEGREES = 90.0
#
#
# # ==========================================================================================
#
#
# # --- Safe Metadata Loading ---
# class GenericMetadata:
#     def __init__(self):
#         # We only need the beam angles here. Width/Height will come from the image itself.
#         self.beam_altitude_angles = np.linspace(22.5, -22.5, 128).tolist()
#
#
# # Load real metadata or fallback to generic beams
# try:
#     with open(metadata_path, 'r') as f:
#         metadata = client.SensorInfo(f.read())
# except Exception as e:
#     print(f"Note: Using Generic Beam Angles (Fallback mode)")
#     metadata = GenericMetadata()
#
#
# def read_boxes_from_new_json(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#
#     categories = data.get('dataset', {}).get('task_attributes', {}).get('categories', [])
#     category_map = {cat['id']: cat['name'] for cat in categories}
#     all_frames_boxes = {}
#
#     samples = data.get('dataset', {}).get('samples', [])
#     for sample in samples:
#         frame_name = sample.get('name', '')
#         frame_num = frame_name.replace('pcd_out_0', '').replace('.bin', '')
#
#         boxes = []
#         labels = sample.get('labels') or {}
#         ground_truth = labels.get('ground-truth') or {}
#         attributes = ground_truth.get('attributes') or {}
#         annotations = attributes.get('annotations', [])
#
#         for annotation in annotations:
#             if annotation.get('type') == 'cuboid':
#                 position = annotation.get('position', {})
#                 dimensions = annotation.get('dimensions', {})
#                 yaw = annotation.get('yaw', 0)
#                 category_id = annotation.get('category_id', 0)
#
#                 box = {
#                     'center': [position.get('x', 0), position.get('y', 0), position.get('z', 0)],
#                     'size': [dimensions.get('x', 0), dimensions.get('y', 0), dimensions.get('z', 0)],
#                     'rotation': yaw,
#                     'category_id': category_id
#                 }
#                 boxes.append(box)
#
#         all_frames_boxes[frame_num] = boxes
#
#     return all_frames_boxes, category_map
#
#
# def get_box_corners(center, size, rotation):
#     cx, cy, cz = center
#     l, w, h = size
#     rot = rotation
#
#     corners = np.array([
#         [cx - l / 2, cy - w / 2, cz - h / 2],
#         [cx - l / 2, cy + w / 2, cz - h / 2],
#         [cx + l / 2, cy + w / 2, cz - h / 2],
#         [cx + l / 2, cy - w / 2, cz - h / 2],
#         [cx - l / 2, cy - w / 2, cz + h / 2],
#         [cx - l / 2, cy + w / 2, cz + h / 2],
#         [cx + l / 2, cy + w / 2, cz + h / 2],
#         [cx + l / 2, cy - w / 2, cz + h / 2],
#     ])
#
#     rotation_matrix = np.array([
#         [np.cos(rot), -np.sin(rot), 0],
#         [np.sin(rot), np.cos(rot), 0],
#         [0, 0, 1]
#     ])
#     rotated_corners = np.dot(corners - center, rotation_matrix.T) + center
#
#     return rotated_corners
#
#
# def point_To_pix(metadataT, x, y, z, img_w, img_h):
#     """
#     Projects point to pixel using the ACTUAL image dimensions.
#     """
#     list128 = metadataT.beam_altitude_angles
#
#     l = np.sqrt(x * x + y * y + z * z)
#     if l == 0: return [0, 0]
#
#     # --- Vertical Calculation ---
#     sita128 = np.arcsin(z / l) * 180 / np.pi
#     diffs = np.abs(np.array(list128) - sita128)
#     row_idx = np.argmin(diffs)
#
#     # Map row index to image height (assumes image height corresponds to beam count)
#     target_v = int((row_idx / 128.0) * img_h)
#
#     # --- Horizontal Calculation ---
#     sita_h = np.arctan2(y, x) * 180 / np.pi
#
#     # Invert and normalize to 0-360
#     target_angle = -sita_h
#     if target_angle < 0:
#         target_angle += 360
#
#     # Apply the 90 Degree Shift
#     target_angle += SHIFT_DEGREES
#
#     # Normalize back to 0-360 range
#     target_angle = target_angle % 360
#
#     # Map degrees to Image Width (Resolution Agnostic)
#     target_u = (target_angle / 360.0) * img_w
#
#     return [int(target_u), int(target_v)]
#
#
# def project_to_image2(corners, img_w, img_h):
#     cornersPix = []
#     for ind in range(8):
#         x, y, z = corners[ind]
#         # Pass the image dimensions down to the projection function
#         cornersPix.append(point_To_pix(metadata, x, y, z, img_w, img_h))
#     return np.array(cornersPix)
#
#
# def draw_box(image, projected_corners, category_name="", frame_num='1'):
#     h, w = image.shape[:2]
#
#     valid_mask = projected_corners[:, 0] != 0
#     if not np.any(valid_mask):
#         return
#     valid_corners = projected_corners[valid_mask].astype(np.float32)
#
#     # 1. Detect Seam Wrapping
#     # If the object spans more than 50% of the image width, it's crossing the edge
#     xs = valid_corners[:, 0]
#     is_split = (np.max(xs) - np.min(xs)) > (w / 2)
#
#     if is_split:
#         # Normalize points: Shift left-side points to the right 'virtual' space
#         valid_corners[:, 0][valid_corners[:, 0] < (w / 2)] += w
#
#     rect = cv2.minAreaRect(valid_corners)
#     box = cv2.boxPoints(rect)
#     box = box.astype(np.int32)
#
#     if is_split:
#         # DRAW TWICE Strategy:
#         # 1. Draw normally (points > W will be clipped, points < W will show)
#         # Note: We need to wrap the points that are > W back to < W for the "right side" pass
#         box_right = box.copy()
#         box_right[:, 0] = box_right[:, 0] % w
#
#         # However, purely modulo isn't enough for drawing lines correctly.
#         # It's safer to draw it in "virtual space" and let CV2 clip, then shift and draw again.
#
#         # Simpler approach for visual correctness:
#         # Draw on the Right Edge (points > W need to stay > W? No, CV2 won't draw offscreen)
#         # We need two sets of valid coordinates.
#
#         # Set 1: The part on the right edge
#         box_pass1 = box.copy()
#         # No change needed, just let it draw. The parts > W won't show.
#         # But wait, if ALL parts are > W (shifted), nothing shows.
#         # Actually, we shifted "left" points (small x) to "right" (large x).
#         # So now the box is unified around W (e.g. from 2000 to 2100).
#
#         # To draw the LEFT part (x=20 to x=50): Shift everything by -W
#         box_left = box.copy()
#         box_left[:, 0] -= w
#         cv2.drawContours(image, [box_left], 0, (255, 255, 0), 1)
#
#         # To draw the RIGHT part (x=2000 to x=2040): Keep as is?
#         # If the box is 2000 to 2100, we want to draw the 2000 part.
#         # The 2100 part is off screen.
#         cv2.drawContours(image, [box], 0, (255, 255, 0), 1)
#
#         # Label position
#         label_x = min(box[:, 0])
#         if label_x >= w: label_x -= w
#         label_pos = (int(label_x), int(min(box[:, 1])) - 5)
#
#     else:
#         # Standard case
#         box = np.clip(box, [0, 0], [w - 1, h - 1])
#         cv2.drawContours(image, [box], 0, (255, 255, 0), 1)
#         label_pos = (int(min(box[:, 0])), int(min(box[:, 1])) - 5)
#
#     if category_name:
#         if label_pos[1] < 10: label_pos = (label_pos[0], 15)
#         cv2.putText(image, category_name, label_pos,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
#
#
# def run():
#     # =========================================================================
#     # CONFIGURATION
#     # =========================================================================
#     # FIXED: This now points to metadata_path at the top of the file!
#     json_file_path = '/home/yanan/Downloads/Boston_01_PointCloud-Boston_01.json'
#     input_image_dir = "/media/yanan/MA2023-2/Ouster_LiDAR/Boston/label_data/boston_01/images"
#     output_dir = "./output/label_image_box/Boston_017"
#
#     field_names = ["combined"]
#     # =========================================================================
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     print("Reading JSON file...")
#     all_frames_boxes, category_map = read_boxes_from_new_json(json_file_path)
#     print(f"Total frames found: {len(all_frames_boxes)}")
#
#     for frame_num, boxes in sorted(all_frames_boxes.items()):
#         print(f"Processing frame {frame_num}...")
#
#         images = []
#         # Default dimensions in case first read fails (fallback)
#         current_h, current_w = 128, 1024
#
#         for field_name in field_names:
#             image_path = os.path.join(input_image_dir, f"frame_{frame_num}_{field_name}.png")
#
#             if not os.path.exists(image_path):
#                 possible = [f for f in os.listdir(input_image_dir) if frame_num in f and field_name in f]
#                 if possible:
#                     image_path = os.path.join(input_image_dir, possible[0])
#                 else:
#                     print(f"Warning: Image not found {image_path}")
#                     continue
#
#             image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#             if image is None:
#                 print(f"Warning: Cannot read {image_path}")
#                 continue
#
#             # Update dimensions based on actual loaded image
#             current_h, current_w = image.shape[:2]
#             images.append(image)
#
#         if not images: continue
#
#         if boxes:
#             for box in boxes:
#                 center = box['center']
#                 size = box['size']
#                 rotation = box['rotation']
#                 category_id = box.get('category_id', 0)
#                 category_name = category_map.get(category_id, 'unknown')
#
#                 corners = get_box_corners(center, size, rotation)
#
#                 # UPDATED: Pass the actual image width/height to the projection
#                 projected_corners = project_to_image2(corners, current_w, current_h)
#
#                 for image in images:
#                     draw_box(image, projected_corners, category_name, frame_num)
#
#         for i, image in enumerate(images):
#             output_filename = os.path.join(output_dir, f"frame_{frame_num}_{field_names[i]}_boxed.png")
#             cv2.imwrite(output_filename, image)
#
#         print(f"Frame {frame_num} complete. Boxes drawn: {len(boxes)}")
#
#     print("All processing complete.")
#
#
# if __name__ == "__main__":
#     run()
