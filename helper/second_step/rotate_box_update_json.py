import os
import json
import numpy as np

label_dir = '/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/label'

for filename in os.listdir(label_dir):
    if not filename.endswith('.json'):
        continue
    if not filename.startswith('boston_'):
        continue

    filepath = os.path.join(label_dir, filename)
    with open(filepath, 'r') as f:
        boxes = json.load(f)

    for box in boxes:
        x = box['position3d']['x']
        y = box['position3d']['y']
        # 逆时针90度：new_x = -y, new_y = x
        box['position3d']['x'] = -y
        box['position3d']['y'] = x
        # heading也要转
        box['heading'] = box['heading'] + np.pi / 2

    with open(filepath, 'w') as f:
        json.dump(boxes, f, indent=4)
    print(f'处理: {filename}')

print('完成！')