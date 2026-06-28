import os
import json

input_dir = '/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/label2'

for filename in os.listdir(input_dir):
    if not filename.endswith('.json'):
        continue
    if not filename.startswith('boston_'):  # ← 只处理boston文件
        continue

    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'r') as f:
        boxes = json.load(f)

    for box in boxes:
        box['position3d']['z'] += 1.5

    with open(filepath, 'w') as f:
        json.dump(boxes, f, indent=4)

    print(f"处理: {filename}")

print("完成！")