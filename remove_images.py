import os
import shutil
from pathlib import Path


def move_matching_png_files(txt_folder, png_folder, output_folder):
    """
    根据txt文件名，将对应的png文件移动到新文件夹

    参数:
        txt_folder: 包含txt文件的文件夹路径
        png_folder: 包含png文件的文件夹路径
        output_folder: 输出文件夹路径（用于存放移动后的png文件）
    """

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有txt文件的基本名称（不含扩展名）
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

    moved_count = 0
    not_found_count = 0
    not_found_files = []

    print(f"找到 {len(txt_files)} 个txt文件")
    print(f"开始处理...\n")

    for txt_file in txt_files:
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(txt_file)[0]

        # 构建对应的png文件名
        png_file = base_name + '.png'
        png_path = os.path.join(png_folder, png_file)

        # 检查png文件是否存在
        if os.path.exists(png_path):
            # 构建目标路径
            output_path = os.path.join(output_folder, png_file)

            # 移动文件
            shutil.move(png_path, output_path)
            moved_count += 1
            print(f"✓ 已移动: {png_file}")
        else:
            not_found_count += 1
            not_found_files.append(png_file)
            print(f"✗ 未找到: {png_file}")

    # 打印统计信息
    print(f"\n{'=' * 50}")
    print(f"处理完成!")
    print(f"成功移动: {moved_count} 个文件")
    print(f"未找到: {not_found_count} 个文件")

    if not_found_files:
        print(f"\n未找到的文件列表:")
        for file in not_found_files:
            print(f"  - {file}")


def copy_every_5th_frame(png_folder, output_folder, start=1):
    """从第start帧开始，每5帧复制一个png到output文件夹"""
    os.makedirs(output_folder, exist_ok=True)

    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])
    print(png_files)

    for i in range(start, len(png_files), 5):
        src = os.path.join(png_folder, png_files[i])
        dst = os.path.join(output_folder, png_files[i])
        shutil.copy(src, dst)
        print(f"已复制: {png_files[i]}")

    print(f"\n总共复制了 {(len(png_files) - start + 1) // 5 + 1} 个文件")


if __name__ == "__main__":
    # 设置文件夹路径（请根据实际情况修改）
    # txt_folder = "./data/Bus/test/labels"  # txt文件所在的文件夹
    # png_folder = "./output/output_combined_images"  # png文件所在的文件夹
    # output_folder = "./data/Bus/test/image"  # 输出文件夹

    png_folder = "./output/images/output_combined_images/30_02"  # png文件所在的文件夹
    output_folder = "./output/images/output_combined_images/30_02_upload"  # 输出文件夹

    # # 检查输入文件夹是否存在
    # if not os.path.exists(txt_folder):
    #     print(f"错误: txt文件夹不存在: {txt_folder}")
    #     exit(1)

    if not os.path.exists(png_folder):
        print(f"错误: png文件夹不存在: {png_folder}")
        exit(1)


    # 执行移动操作
    # move_matching_png_files(txt_folder, png_folder, output_folder)
    copy_every_5th_frame(png_folder, output_folder, start=1)