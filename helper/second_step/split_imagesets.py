import os

bus_root = "/home/yanan/Downloads/projects/multimodal_detection/data/Bus"
imagesets_root = "/home/yanan/Downloads/projects/multimodal_detection/data/custom/ImageSets"

splits = ["train", "val", "test"]
os.makedirs(imagesets_root, exist_ok=True)

for split in splits:
    label_dir = os.path.join(bus_root, split, "labels")
    out_txt = os.path.join(imagesets_root, f"{split}.txt")

    names = []

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".txt"):
            continue

        # 去掉后缀
        name = fname.replace(".txt", "")

        # 拆分前缀 和 frame id
        prefix, frame_part = name.split("_frame_")
        frame_id = frame_part.replace("_combined", "").zfill(6)

        # 统一生成 PointPillars 文件名
        names.append(f"{prefix}_pcd_out_{frame_id}")

    with open(out_txt, "w") as f:
        f.write("\n".join(names))

    print(f"{split}.txt written, {len(names)} samples.")
