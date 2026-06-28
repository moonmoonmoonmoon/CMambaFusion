# import os
# import json
#
# def filter_labels_by_range(label_txt_dir, label_json_dir,
#                            output_txt_dir, output_json_dir, pc_range):
#     """过滤range外的标注，确保2D和3D标注一致"""
#
#     for json_file in os.listdir(label_json_dir):
#         if not json_file.endswith('.json'):
#             continue
#
#         # 读取3D标注
#         with open(os.path.join(label_json_dir, json_file)) as f:
#             boxes_3d = json.load(f)
#
#         # 过滤range内的boxes
#         valid_boxes = []
#         valid_indices = []
#         for i, box in enumerate(boxes_3d):
#             x = box['position3d']['x']
#             y = box['position3d']['y']
#             z = box['position3d']['z']
#
#             if (pc_range[0] <= x <= pc_range[3] and
#                     pc_range[1] <= y <= pc_range[4] and
#                     pc_range[2] <= z <= pc_range[5]):
#                 valid_boxes.append(box)
#                 valid_indices.append(i)
#
#         # 保存过滤后的3D标注
#         with open(os.path.join(output_json_dir, json_file), 'w') as f:
#             json.dump(valid_boxes, f, indent=2)
#
#         # 过滤对应的2D标注
#         txt_file = json_file.replace('.json', '_combined.txt')
#         with open(os.path.join(label_txt_dir, txt_file)) as f:
#             lines = f.readlines()
#
#         valid_lines = [lines[i] for i in valid_indices if i < len(lines)]
#
#         with open(os.path.join(output_txt_dir, txt_file), 'w') as f:
#             f.writelines(valid_lines)
#
#         print(f"{json_file}: {len(boxes_3d)} -> {len(valid_boxes)}")
#
#
# # 使用
# filter_labels_by_range(
#     label_txt_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/620_bus/train/labels',
#     label_json_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/620_custom/training/label',  # 注意是label不是labels
#     output_txt_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/bus_labels_filtered',
#     output_json_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/custom_label_filtered',
#     pc_range=[-51.2, -60, -3, 51.2, 41.12, 1]
# )

import os
import json
import re


def extract_frame_number(filename):
    """
    从文件名提取帧号
    pcd_out_000251 -> 251
    frame_00251 -> 251
    """
    # 提取所有数字序列
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # 返回最后一个数字序列（通常是帧号）
        # print('1', int(numbers[-1]))
        return int(numbers[-1])
    return None


def find_matching_txt(json_file, txt_dir):
    """
    为json文件找到匹配的txt文件
    """
    # 提取json的帧号
    json_frame_num = extract_frame_number(json_file)
    # print('json_frame_num: ', json_frame_num)
    if json_frame_num is None:
        return None

    # 提取前缀（bus_30_01部分）
    prefix = '_'.join(json_file.split('_')[:-3])  # bus_30_01
    # prefix = ''
    # print('prefix: ', prefix)

    # 在txt目录中查找匹配的文件
    for txt_file in os.listdir(txt_dir):
        if not txt_file.endswith('_combined.txt'):
            continue

        # 检查前缀是否匹配
        if not txt_file.startswith(prefix):
            continue

        # 检查帧号是否匹配
        txt_frame_num = extract_frame_number(txt_file)
        # print('txt_frame_num: ', txt_frame_num)
        if txt_frame_num == json_frame_num:
            return txt_file

    return None


def filter_labels_by_range(label_txt_dir, label_json_dir,
                           output_txt_dir, output_json_dir, pc_range):
    """过滤range外的标注"""
    os.makedirs(output_txt_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    stats = {'total': 0, 'filtered': 0, 'no_match': 0, 'filtered_boxes': 0}

    for json_file in sorted(os.listdir(label_json_dir)):
        if not json_file.endswith('.json'):
            continue

        stats['total'] += 1

        # 读取3D标注
        json_path = os.path.join(label_json_dir, json_file)
        with open(json_path) as f:
            boxes_3d = json.load(f)

        # 过滤range内的boxes
        valid_boxes = []
        valid_indices = []
        for i, box in enumerate(boxes_3d):
            x = box['position3d']['x']
            y = box['position3d']['y']
            z = box['position3d']['z']

            if (pc_range[0] <= x <= pc_range[3] and
                    pc_range[1] <= y <= pc_range[4] and
                    pc_range[2] <= z <= pc_range[5]):
                valid_boxes.append(box)
                valid_indices.append(i)

        # 保存过滤后的3D标注
        output_json_path = os.path.join(output_json_dir, json_file)

        with open(output_json_path, 'w') as f:
            json.dump(valid_boxes, f, indent=2)

        # 查找匹配的txt文件
        txt_file = find_matching_txt(json_file, label_txt_dir)

        if txt_file is None:
            print(f"⚠️  未找到匹配的txt: {json_file}")
            stats['no_match'] += 1
            continue

        # 读取并过滤2D标注
        txt_path = os.path.join(label_txt_dir, txt_file)
        with open(txt_path) as f:
            lines = f.readlines()

        valid_lines = [lines[i] for i in valid_indices if i < len(lines)]

        # 保存过滤后的2D标注
        output_txt_path = os.path.join(output_txt_dir, txt_file)
        with open(output_txt_path, 'w') as f:
            f.writelines(valid_lines)

        if len(boxes_3d) != len(valid_boxes):
            stats['filtered'] += 1
            stats['filtered_boxes'] += (len(boxes_3d) - len(valid_boxes))  # ← 添加这行
            print(f"✓ {json_file}: {len(boxes_3d)} -> {len(valid_boxes)} boxes")

    # 打印统计信息
    print(f"\n=== 过滤统计 ===")
    print(f"总文件数: {stats['total']}")
    print(f"有过滤的文件: {stats['filtered']}")
    print(f"过滤的box总数: {stats['filtered_boxes']}")  # ← 添加这行
    print(f"未匹配txt: {stats['no_match']}")
    print(f"过滤率: {stats['filtered'] / stats['total'] * 100:.1f}%")


# 使用示例
if __name__ == '__main__':
    # # val set
    # filter_labels_by_range(
    #     label_txt_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/620_bus/train/labels',
    #     label_json_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/620_custom/training/label',
    #     output_txt_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/bus_labels_filtered',
    #     output_json_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/custom_label_filtered',
    #     pc_range=[-51.2, -60, -3, 51.2, 41.12, 1]
    # )

    # # # # train set
    # filter_labels_by_range(
    #     label_txt_dir = '/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/bus_labels_filtered_z5',
    #     label_json_dir = '/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/custom_label_filtered_z5',
    #     output_txt_dir = '/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/bus_labels_filtered_havefilterxy_z1',
    #     output_json_dir = '/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/custom_label_filtered_havefilterxy_z1',
    #     pc_range = [-51.2, -60, -3, 51.2, 41.12, 1]
    # )

    # filter_labels_by_range(
    #     label_txt_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/original/Bus/labels',
    #     label_json_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/original/custom/label',
    #     output_txt_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/z5/one_time_filter_xyz/bus_labels_filtered',
    #     output_json_dir='/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/z5/one_time_filter_xyz/custom_label_filtered',
    #     pc_range=[-51.2, -60, -3, 51.2, 41.12, 5]
    # )
    filter_labels_by_range(
        label_txt_dir='/home/yanan/Downloads/projects/multimodal_detection/data/Bus/train/labels2',
        label_json_dir='/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/label2',
        output_txt_dir='/home/yanan/Downloads/projects/multimodal_detection/data/Bus/train/labels',
        output_json_dir='/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/label',
        pc_range=[-51.2, -60, -3, 51.2, 41.12, 5]
    )