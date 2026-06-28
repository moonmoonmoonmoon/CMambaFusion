import json
import os
import shutil


def extract_frame_num(sample_name):
    """从sample名称提取帧编号: "pcd_out_000007.bin" -> "00007" """
    # return sample_name.replace('pcd_out_0', '').replace('.bin', '')
    return sample_name.replace('bus_30_02_pcd_out_0', '').replace('.bin', '')


def extract_frame_num_from_image(image_name):
    """从图像名提取帧编号: "frame_00007_combined.png" -> "00007" """
    return image_name.split('_')[1]


# ==================== 配置 ====================
json_file = "/home/yanan/Downloads/All_Route_30_02_PointCloud-Final_30_02_Bus_All.json"
image_dir = "/home/yanan/Downloads/projects/multimodal_detection/output/images/output_combined_images/upload/30_02_upload"


# 新的清理后文件夹路径（会自动创建）
output_dir = "../output/images/output_combined_images/Final/30_02_final"  # 可以修改为其他路径
# ==================== 开始处理 ====================

# 1. 读取JSON，获取所有有效的帧编号
print("读取JSON文件...")
with open(json_file, 'r') as f:
    data = json.load(f)

valid_frames = set()
for sample in data['dataset']['samples']:
    frame_num = extract_frame_num(sample['name'])
    valid_frames.add(frame_num)

print(f"JSON中有 {len(valid_frames)} 个有效帧")

# 2. 扫描图像文件夹
print(f"\n扫描图像文件夹: {image_dir}")
all_images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
print(f"找到 {len(all_images)} 个图像文件")

# 3. 找出要删除的图像（不在JSON中的）
to_delete = []
for img in all_images:
    frame_num = extract_frame_num_from_image(img)
    if frame_num not in valid_frames:
        to_delete.append(img)

# 4. 显示结果
print(f"\n找到 {len(to_delete)} 个要删除的图像")
if to_delete:
    print("前10个示例:")
    for img in to_delete[:10]:
        print(f"  - {img}")

# 5. 确认并删除
if to_delete:
    response = input(f"\n确定要删除这 {len(to_delete)} 个图像吗? (yes/no): ")
    if response.lower() == 'yes':
        # 创建输出文件夹
        output_path = f"{output_dir}"
        os.makedirs(output_path, exist_ok=True)
        print(f"\n输出文件夹: {output_path}")

        # 复制所有图像到新文件夹
        print("\n正在复制所有图像到新文件夹...")
        for img in all_images:
            src = os.path.join(image_dir, img)
            dst = os.path.join(output_path, img)
            shutil.copy2(src, dst)
        print(f"✓ 已复制 {len(all_images)} 个图像")

        # 在新文件夹删除不需要的图像
        print("\n正在从新文件夹删除不需要的图像...")
        for img in to_delete:
            os.remove(os.path.join(output_path, img))
            print(f"已删除: {img}")
        print(f"\n完成! 已删除 {len(to_delete)} 个图像")
    else:
        print("已取消")
else:
    print("\n没有需要删除的图像!")