import os
import numpy as np

input_dir = '/home/yanan/Downloads/projects/multimodal_detection/data/dataset/Test/Boston_01/bin'
output_dir = '/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/data'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith('.bin'):
        continue

    filepath = os.path.join(input_dir, filename)
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

    points[:, 2] += 1.5  # 只改z

    output_path = os.path.join(output_dir, filename)
    points.tofile(output_path)

print("完成！")