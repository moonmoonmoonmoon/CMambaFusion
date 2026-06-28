"""
每次跑一个模型，存 PR 数据
用法（跑完一个模型之后，改代码/yaml，再跑下一个）:

# 跑 CMambaFusion
python save_pr_data.py \
  --cfg cfgs/custom_models/pointpillar.yaml \
  --ckpt .../cmamba_fusion.pth \
  --save_name cmamba \
  --class_name Car --iou_thresh 0.7

# 跑 PointPillars（记得先把 yaml 改成 LiDAR-only）
python save_pr_data.py \
  --cfg cfgs/custom_models/pointpillar.yaml \
  --ckpt .../pointpillar.pth \
  --save_name pointpillars \
  --class_name Car --iou_thresh 0.7
"""

import argparse, sys, pickle
import numpy as np
import torch

sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


def collect(model, dataloader, class_names, target_class, iou_thresh, score_thresh=0.1):
    class_id = class_names.index(target_class) + 1
    all_scores, all_is_tp, all_frame_ids = [], [], []
    n_gt_total = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            load_data_to_gpu(batch)
            # gt_b = batch['gt_boxes'][0].cpu().numpy()
            # valid = gt_b[gt_b[:, 7] > 0]  # 过滤padding
            # print(f"batch gt_boxes 里总共有效GT数: {len(valid)}")
            # print(f"Car GT数: {(valid[:, 7] == 1).sum()}")
            # break
            pred_dicts, _ = model(batch)
            gt_boxes_batch = batch['gt_boxes']
            frame_ids = batch['frame_id']  # ← 加这行

            for b_idx, pred in enumerate(pred_dicts):
                pred_boxes  = pred['pred_boxes'].cpu().numpy()
                pred_scores = pred['pred_scores'].cpu().numpy()
                pred_labels = pred['pred_labels'].cpu().numpy()

                mask = (pred_labels == class_id) & (pred_scores >= score_thresh)
                pred_boxes, pred_scores = pred_boxes[mask], pred_scores[mask]

                gt_b = gt_boxes_batch[b_idx].cpu().numpy()
                gt_boxes = gt_b[gt_b[:, 7] == class_id, :7]
                n_gt_total += len(gt_boxes)

                frame_id = frame_ids[b_idx]  # ← 记录当前帧

                if len(pred_boxes) == 0:
                    continue

                is_tp = np.zeros(len(pred_boxes), dtype=bool)
                if len(gt_boxes) > 0:
                    iou_mat = iou3d_nms_utils.boxes_iou3d_gpu(
                        torch.tensor(pred_boxes, dtype=torch.float32).cuda(),
                        torch.tensor(gt_boxes,   dtype=torch.float32).cuda()
                    ).cpu().numpy()
                    matched_gt = set()
                    for pi in np.argsort(-pred_scores):
                        best = np.argmax(iou_mat[pi])
                        if iou_mat[pi, best] >= iou_thresh and best not in matched_gt:
                            is_tp[pi] = True
                            matched_gt.add(best)

                all_scores.append(pred_scores)
                all_is_tp.append(is_tp)
                all_frame_ids.extend([frame_id] * len(pred_scores))  # ← 每个预测框对应的帧

    scores = np.concatenate(all_scores) if all_scores else np.array([])
    is_tp  = np.concatenate(all_is_tp)  if all_is_tp  else np.array([])
    frame_ids = all_frame_ids
    return scores, is_tp, n_gt_total, frame_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg',          required=True)
    p.add_argument('--ckpt',         required=True)
    p.add_argument('--save_name',    required=True, help='存为 pr_<save_name>.pkl')
    p.add_argument('--class_name',   default='Car')
    p.add_argument('--iou_thresh',   type=float, default=0.7)
    p.add_argument('--score_thresh', type=float, default=0.1)
    args = p.parse_args()

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()
    cfg.DATA_CONFIG.DATA_SPLIT = {'train': 'train', 'test': 'val'}

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=4, dist=False, training=False, logger=logger
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()

    scores, is_tp, n_gt, frame_ids = collect(
        model, dataloader, cfg.CLASS_NAMES,
        args.class_name, args.iou_thresh, args.score_thresh
    )
    print(f"n_gt={n_gt}, n_pred={len(scores)}, n_tp={is_tp.sum()}")

    out = f'pr_{args.save_name}.pkl'
    with open(out, 'wb') as f:
        pickle.dump({'scores': scores, 'is_tp': is_tp, 'n_gt': n_gt, 'frame_ids': frame_ids,   # ← 新增
                     'class': args.class_name, 'iou': args.iou_thresh}, f)
    print(f"已保存: {out}")


if __name__ == '__main__':
    main()

#
#
# import  pickle
# # 找 CMambaFusion 比 PointPillars 多 TP 的帧
# with open('./pr_cmamba_car_01.pkl', 'rb') as f:
#     cm = pickle.load(f)
# with open('./pr_pp_car_01.pkl', 'rb') as f:
#     pp = pickle.load(f)
#
# # 按帧统计 TP 数
# from collections import defaultdict
# def tp_per_frame(d):
#     frame_tp = defaultdict(int)
#     for fid, tp in zip(d['frame_ids'], d['is_tp']):
#         if tp:
#             frame_tp[fid] += 1
#     return frame_tp
#
# cm_tp = tp_per_frame(cm)
# pp_tp = tp_per_frame(pp)
#
# # 找 CMamba 比 PP 多的帧
# diff = {fid: cm_tp[fid] - pp_tp.get(fid, 0)
#         for fid in cm_tp if cm_tp[fid] > pp_tp.get(fid, 0)}
# top_frames = sorted(diff.items(), key=lambda x: -x[1])[:10]
# print("CMambaFusion 比 PointPillars 多 TP 的帧（Top10）:")
# for fid, d in top_frames:
#     print(f"  {fid}: +{d} TP")

