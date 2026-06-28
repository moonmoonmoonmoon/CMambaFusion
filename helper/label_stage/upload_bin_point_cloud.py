import os
import shutil

# 设置路径
source_folder = "/media/yanan/MA2023-2/Ouster_LiDAR/2025_bus_all/38_02/bin"  # 修改为你的源文件夹路径
target_folder = "./bin/38_02_upload"  # 修改为你的目标文件夹路径

# 创建目标文件夹（如果不存在）
os.makedirs(target_folder, exist_ok=True)

# 从5001开始，每5帧复制一个
frame = 1
while True:
    filename = f"pcd_out_{frame:06d}.bin"
    source_path = os.path.join(source_folder, filename)

    # 如果文件不存在，停止
    if not os.path.exists(source_path):
        break

    # 复制文件
    target_path = os.path.join(target_folder, filename)
    shutil.copy2(source_path, target_path)
    print(f"已复制: {filename}")

    # 跳到下一个（每5帧）
    frame += 5

print(f"完成！共复制了 {(frame - 5001) // 5} 个文件")

#
# import os
# import shutil
#
# # 设置路径
# source_folder = "/home/yanan/Downloads/projects/multimodal_detection/output_frames/images"  # 修改为你的源文件夹路径
# target_folder = "./images/38_02_upload"  # 修改为你的目标文件夹路径
#
# # 创建目标文件夹（如果不存在）
# os.makedirs(target_folder, exist_ok=True)
#
# # 从5001开始，每5帧复制一个
# frame = 1
# while True:
#     filename = f"frame_{frame:05d}_combined.png"
#     source_path = os.path.join(source_folder, filename)
#
#     # 如果文件不存在，停止
#     if not os.path.exists(source_path):
#         break
#
#     # 复制文件
#     target_path = os.path.join(target_folder, filename)
#     shutil.copy2(source_path, target_path)
#     print(f"已复制: {filename}")
#
#     # 跳到下一个（每5帧）
#     frame += 5
#
# print(f"完成！共复制了 {(frame - 5001) // 5} 个文件")