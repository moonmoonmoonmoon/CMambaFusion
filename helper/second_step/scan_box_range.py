# 中心点范围（position3d x/y 的 min/max）
# import os, glob, json
# import numpy as np
#
# LABEL_DIR = "/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/label1"   # 改成你的label目录
# PATTERN = "*.json"                        # 或 "bus_30_01_pcd_out_*.json"
#
#
# def main():
#     files = sorted(glob.glob(os.path.join(LABEL_DIR, PATTERN)))
#     assert files, f"No label json found in {LABEL_DIR}"
#
#     gxmin, gxmax = np.inf, -np.inf
#     gymin, gymax = np.inf, -np.inf
#     fxmin = fxmax = fymin = fymax = None
#
#     for f in files:
#         boxes = json.load(open(f, "r"))
#         if not boxes:   # 空标注: []
#             continue
#         for b in boxes:
#             x = b["position3d"]["x"]
#             y = b["position3d"]["y"]
#
#             if x < gxmin: gxmin, fxmin = x, os.path.basename(f)
#             if x > gxmax: gxmax, fxmax = x, os.path.basename(f)
#             if y < gymin: gymin, fymin = y, os.path.basename(f)
#             if y > gymax: gymax, fymax = y, os.path.basename(f)
#
#     print("=== Box CENTER range from labels ===")
#     print(f"x_min(center) = {gxmin:.3f}  in {fxmin}")
#     print(f"x_max(center) = {gxmax:.3f}  in {fxmax}")
#     print(f"y_min(center) = {gymin:.3f}  in {fymin}")
#     print(f"y_max(center) = {gymax:.3f}  in {fymax}")
#
# if __name__ == "__main__":
#     main()



# # 真实盒子边界范围（center ± size/2，xmin/xmax/ymin/ymax）
# import os, glob, json
# import numpy as np
#
# LABEL_DIR = "/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/label2"   # 改成你的label目录
# PATTERN = "*.json"
# X_MIN_LIM, X_MAX_LIM = -51.2, 51.2
# Y_MIN_LIM, Y_MAX_LIM = -60.0, 41.12
# Z_MIN_LIM, Z_MAX_LIM = -3, 1
#
#
# # X_MIN_LIM, X_MAX_LIM = -51.2, 51.2
# # Y_MIN_LIM, Y_MAX_LIM = -70.0, 51.12
# def main():
#     files = sorted(glob.glob(os.path.join(LABEL_DIR, PATTERN)))
#     assert files, f"No label json found in {LABEL_DIR}"
#
#     gxmin, gxmax = np.inf, -np.inf
#     gymin, gymax = np.inf, -np.inf
#     fxmin = fxmax = fymin = fymax = None
#     total_boxes = 0
#     out_boxes = 0
#
#     frames_with_out = set()
#     non_empty_frames = 0
#
#     for f in files:
#         boxes = json.load(open(f, "r"))
#         if not boxes:
#             continue
#         non_empty_frames += 1
#         for b in boxes:
#             cx, cy = b["position3d"]["x"], b["position3d"]["y"]
#             sx, sy = b["size3d"]["x"], b["size3d"]["y"]
#             xmin, xmax = cx - sx/2, cx + sx/2
#             ymin, ymax = cy - sy/2, cy + sy/2
#
#             if xmin < gxmin: gxmin, fxmin = xmin, os.path.basename(f)
#             if xmax > gxmax: gxmax, fxmax = xmax, os.path.basename(f)
#             if ymin < gymin: gymin, fymin = ymin, os.path.basename(f)
#             if ymax > gymax: gymax, fymax = ymax, os.path.basename(f)
#             total_boxes += 1
#
#             out = (
#                     xmin < X_MIN_LIM or xmax > X_MAX_LIM or
#                     ymin < Y_MIN_LIM or ymax > Y_MAX_LIM
#             )
#             # out = not (X_MIN_LIM <= cx <= X_MAX_LIM and Y_MIN_LIM <= cy <= Y_MAX_LIM)
#
#             if out:
#                 out_boxes += 1
#                 frames_with_out.add(os.path.basename(f))
#
#     print("=== Box EXTENT range from labels (center ± size/2) ===")
#     print(f"x_min = {gxmin:.3f}  in {fxmin}")
#     print(f"x_max = {gxmax:.3f}  in {fxmax}")
#     print(f"y_min = {gymin:.3f}  in {fymin}")
#     print(f"y_max = {gymax:.3f}  in {fymax}")
#
#     # 结果
#     total_frames = len(files)
#     out_frames = len(frames_with_out)
#
#     print("限制范围:")
#     print(f"x in [{X_MIN_LIM}, {X_MAX_LIM}], y in [{Y_MIN_LIM}, {Y_MAX_LIM}]\n")
#
#     print("超出范围的帧（前50个）:")
#     print(sorted(list(frames_with_out))[:500], "\n")
#
#     print("统计结果:")
#     print(f"超出范围的box数: {out_boxes}")
#     print(f"总box数: {total_boxes}")
#     print(f"box超出占比: {out_boxes / total_boxes:.4%}")
#
#     print(f"\n含超出box的帧数: {out_frames}")
#     print(f"总帧数(含空标注帧): {total_frames}")
#     print(f"帧占比(对全部帧): {out_frames / total_frames:.4%}")
#
#     print(f"\n总帧数(仅非空标注帧): {non_empty_frames}")
#     print(f"帧占比(对非空帧): {out_frames / non_empty_frames:.4%}")
#
# if __name__ == "__main__":
#     main()


# 中心点范围（position3d x/y/z 的 min/max）
import os, glob, json
import numpy as np

LABEL_DIR = "/home/yanan/Downloads/projects/multimodal_detection/data/custom1/training/label"   # 改成你的label目录
# LABEL_DIR = "/home/yanan/Downloads/projects/multimodal_detection/data/filter_labels/5827/custom_label_filtered"
PATTERN = "*.json"
X_MIN_LIM, X_MAX_LIM = -51.2, 51.2
Y_MIN_LIM, Y_MAX_LIM = -60.0, 41.12
Z_MIN_LIM, Z_MAX_LIM = -3, 1


# X_MIN_LIM, X_MAX_LIM = -51.2, 51.2
# Y_MIN_LIM, Y_MAX_LIM = -70.0, 51.12
def main():
    files = sorted(glob.glob(os.path.join(LABEL_DIR, PATTERN)))
    assert files, f"No label json found in {LABEL_DIR}"

    gxmin, gxmax = np.inf, -np.inf
    gymin, gymax = np.inf, -np.inf
    gzmin, gzmax = np.inf, -np.inf
    fxmin = fxmax = fymin = fymax = fzmin = fzmax = None
    total_boxes = 0
    out_boxes = 0

    frames_with_out = set()
    non_empty_frames = 0

    for f in files:
        boxes = json.load(open(f, "r"))
        if not boxes:
            continue
        non_empty_frames += 1
        for b in boxes:
            cx, cy, cz = b["position3d"]["x"], b["position3d"]["y"], b["position3d"]["z"]
            sx, sy = b["size3d"]["x"], b["size3d"]["y"]


            if cx < gxmin: gxmin, fxmin = cx, os.path.basename(f)
            if cx > gxmax: gxmax, fxmax = cx, os.path.basename(f)
            if cy < gymin: gymin, fymin = cy, os.path.basename(f)
            if cy > gymax: gymax, fymax = cy, os.path.basename(f)
            if cz < gzmin: gzmin, fzmin = cz, os.path.basename(f)
            if cz > gzmax: gzmax, fzmax = cz, os.path.basename(f)
            total_boxes += 1

            out = (
                    cx < X_MIN_LIM or cx > X_MAX_LIM or
                    cy < Y_MIN_LIM or cy > Y_MAX_LIM or
                    cz < Z_MIN_LIM or cz > Z_MAX_LIM
            )
            # out = (
            #         cx < X_MIN_LIM or cx > X_MAX_LIM or
            #         cy < Y_MIN_LIM or cy > Y_MAX_LIM
            # )
            # out = (
            #             cz < Z_MIN_LIM or cz > Z_MAX_LIM
            #     )
            # out = not (X_MIN_LIM <= cx <= X_MAX_LIM and Y_MIN_LIM <= cy <= Y_MAX_LIM)

            if out:
                out_boxes += 1
                frames_with_out.add(os.path.basename(f))

    print("=== Box EXTENT range from labels (center ± size/2) ===")
    print(f"x_min = {gxmin:.3f}  in {fxmin}")
    print(f"x_max = {gxmax:.3f}  in {fxmax}")
    print(f"y_min = {gymin:.3f}  in {fymin}")
    print(f"y_max = {gymax:.3f}  in {fymax}")
    print(f"z_min = {gzmin:.3f}  in {fzmin}")
    print(f"z_max = {gzmax:.3f}  in {fzmax}")

    # 结果
    total_frames = len(files)
    out_frames = len(frames_with_out)

    print("限制范围:")
    print(f"x in [{X_MIN_LIM}, {X_MAX_LIM}], y in [{Y_MIN_LIM}, {Y_MAX_LIM}]\n")

    print("超出范围的帧（前50个）:")
    print(sorted(list(frames_with_out))[:500], "\n")

    print("统计结果:")
    print(f"超出范围的box数: {out_boxes}")
    print(f"总box数: {total_boxes}")
    print(f"box超出占比: {out_boxes / total_boxes:.4%}")

    print(f"\n含超出box的帧数: {out_frames}")
    print(f"总帧数(含空标注帧): {total_frames}")
    print(f"帧占比(对全部帧): {out_frames / total_frames:.4%}")

    print(f"\n总帧数(仅非空标注帧): {non_empty_frames}")
    print(f"帧占比(对非空帧): {out_frames / non_empty_frames:.4%}")

if __name__ == "__main__":
    main()