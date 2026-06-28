# #!/usr/bin/env python3
# """
# 详细评估分析脚本
# 用于诊断融合模型的性能
#
# 使用方法：
# python detailed_evaluation.py \
#     --base_ckpt output/.../base/ckpt/checkpoint_epoch_75.pth \
#     --fusion_ckpt output/.../fusion/ckpt/checkpoint_epoch_45.pth \
#     --cfg_file cfgs/custom_models/pointpillar.yaml \
#     --output_dir analysis_results
# """
#
# import argparse
# import os
# import pickle
# from pathlib import Path
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict
#
# import sys
#
# sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
#
# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import build_dataloader
# from pcdet.models import build_network
# from pcdet.utils import common_utils
#
#
# class DetailedEvaluator:
#     """详细评估器"""
#
#     def __init__(self, model, dataloader, class_names, output_dir):
#         self.model = model
#         self.dataloader = dataloader
#         self.class_names = class_names
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#
#         # 存储结果
#         self.all_pred_boxes = []
#         self.all_pred_scores = []
#         self.all_pred_labels = []
#         self.all_gt_boxes = []
#         self.all_gt_labels = []
#         self.all_ious = defaultdict(list)
#         self.all_matches = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
#         self.all_point_counts = []
#
#     def evaluate(self):
#         """运行评估"""
#         print("开始详细评估...")
#         self.model.eval()
#
#         with torch.no_grad():
#             for i, batch_dict in enumerate(self.dataloader):
#                 if i % 10 == 0:
#                     print(f"Processing {i}/{len(self.dataloader)}...")
#
#                 # 加载到 GPU
#                 self._load_data_to_gpu(batch_dict)
#
#                 # 前向传播
#                 pred_dicts, _ = self.model.forward(batch_dict)
#
#                 # 收集结果
#                 self._collect_results(batch_dict, pred_dicts)
#
#         print("评估完成，开始分析...")
#
#         # 生成所有分析
#         self.analyze_iou_distribution()
#         self.analyze_tp_fp_fn()
#         self.generate_confusion_matrix()
#         self.analyze_by_point_count()
#         self.visualize_false_positives()
#         self.generate_pr_curve()
#
#         print(f"分析完成！结果保存在: {self.output_dir}")
#
#     def _load_data_to_gpu(self, batch_dict):
#         """加载数据到GPU"""
#         for key, val in batch_dict.items():
#             if not isinstance(val, np.ndarray):
#                 continue
#             if key in ['frame_id', 'metadata', 'calib']:
#                 continue
#             batch_dict[key] = torch.from_numpy(val).float().cuda()
#
#     def _collect_results(self, batch_dict, pred_dicts):
#         """收集预测和GT结果"""
#
#         # 🔥 添加调试信息
#         print(f"\n[DEBUG] batch_dict keys: {batch_dict.keys()}")
#         print(f"[DEBUG] pred_dicts type: {type(pred_dicts)}, len: {len(pred_dicts)}")
#
#         batch_size = batch_dict.get('batch_size', 1)  # ← 改这里
#         print(f"[DEBUG] batch_size: {batch_size}")
#
#         for batch_idx in range(batch_size):
#             # 预测结果
#             if batch_idx >= len(pred_dicts):
#                 print(f"[DEBUG] batch_idx {batch_idx} >= len(pred_dicts) {len(pred_dicts)}")
#                 break
#
#             pred_boxes = pred_dicts[batch_idx]['pred_boxes'].cpu().numpy()
#             pred_scores = pred_dicts[batch_idx]['pred_scores'].cpu().numpy()
#             pred_labels = pred_dicts[batch_idx]['pred_labels'].cpu().numpy()
#
#             print(f"[DEBUG] Sample {batch_idx}: pred_boxes shape: {pred_boxes.shape}")
#
#             # GT
#             gt_boxes = batch_dict['gt_boxes'][batch_idx].cpu().numpy()
#             print(f"[DEBUG] Sample {batch_idx}: gt_boxes shape before filter: {gt_boxes.shape}")
#
#             # 过滤掉 padding（全0的boxes）
#             valid_gt = np.any(gt_boxes[:, :3] != 0, axis=1)  # ← 改这里，只看xyz
#             gt_boxes = gt_boxes[valid_gt]
#
#             print(f"[DEBUG] Sample {batch_idx}: gt_boxes shape after filter: {gt_boxes.shape}")
#
#             if len(gt_boxes) == 0:
#                 print(f"[DEBUG] Sample {batch_idx}: 没有有效的 GT boxes，跳过")
#                 continue
#     #
#     #         # ... 后面的代码
#     #
#     # def _collect_results(self, batch_dict, pred_dicts):
#     #     """收集预测和GT结果"""
#     #     # ... 前面的代码 ...
#     #
#     #     # 计算点数（如果有点云数据）
#     #     if 'points' in batch_dict:
#     #         points = batch_dict['points']
#     #         # 检查 points 的维度
#     #         print(f"DEBUG: points shape = {points.shape}")
#     #
#     #         if len(points.shape) == 2 and points.shape[1] >= 4:
#     #             # 筛选属于当前 batch 的点
#     #             batch_mask = points[:, 0] == batch_idx
#     #             batch_points = points[batch_mask, 1:4].cpu().numpy()
#     #             self._count_points_in_boxes(batch_points, gt_boxes)
#     #         else:
#     #             print(f"⚠️  points 格式不符合预期: {points.shape}")
#
#     def _compute_matches(self, pred_boxes, pred_scores, pred_labels,
#                          gt_boxes, gt_labels):
#         """计算 TP/FP/FN 和 IoU"""
#         from pcdet.ops.iou3d_nms import iou3d_nms_utils
#
#         # 对每个类别分别处理
#         all_labels = np.concatenate([pred_labels, gt_labels])
#         unique_labels = np.unique(all_labels)
#
#         for cls_id in unique_labels:
#             # 检查类别ID
#             cls_id = int(cls_id)
#             if cls_id < 1 or cls_id > len(self.class_names):
#                 continue
#
#             cls_name = self.class_names[cls_id - 1]
#
#             # 筛选该类别
#             pred_mask = pred_labels == cls_id
#             gt_mask = gt_labels == cls_id
#
#             cls_pred_boxes = pred_boxes[pred_mask]
#             cls_pred_scores = pred_scores[pred_mask]  # 🔥 修复：也要筛选scores
#             cls_gt_boxes = gt_boxes[gt_mask]
#
#             if len(cls_pred_boxes) == 0 and len(cls_gt_boxes) == 0:
#                 continue
#
#             if len(cls_pred_boxes) == 0:
#                 self.all_matches[cls_name]['FN'] += len(cls_gt_boxes)
#                 continue
#
#             if len(cls_gt_boxes) == 0:
#                 self.all_matches[cls_name]['FP'] += len(cls_pred_boxes)
#                 continue
#
#             # 计算 IoU
#             ious = iou3d_nms_utils.boxes_iou3d_gpu(
#                 torch.from_numpy(cls_pred_boxes).cuda(),
#                 torch.from_numpy(cls_gt_boxes).cuda()
#             ).cpu().numpy()
#
#             # IoU 阈值
#             iou_threshold = 0.7 if 'Car' in cls_name or 'Truck' in cls_name else 0.5
#
#             # 匹配
#             matched_gt = set()
#
#             # 🔥 修复：按正确的scores排序
#             sorted_indices = np.argsort(-cls_pred_scores)
#
#             for pred_idx in sorted_indices:
#                 max_iou_idx = ious[pred_idx].argmax()
#                 max_iou = ious[pred_idx, max_iou_idx]
#
#                 if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
#                     # TP
#                     self.all_matches[cls_name]['TP'] += 1
#                     matched_gt.add(max_iou_idx)
#                     self.all_ious[cls_name].append(max_iou)
#                 else:
#                     # FP
#                     self.all_matches[cls_name]['FP'] += 1
#
#             # FN
#             self.all_matches[cls_name]['FN'] += len(cls_gt_boxes) - len(matched_gt)
#
#     # def _compute_matches(self, pred_boxes, pred_scores, pred_labels,
#     #                      gt_boxes, gt_labels):
#     #     """计算 TP/FP/FN 和 IoU"""
#     #     from pcdet.ops.iou3d_nms import iou3d_nms_utils
#     #     print(f"[DEBUG] _compute_matches called: pred={len(pred_boxes)}, gt={len(gt_boxes)}")
#     #
#     #     # 对每个类别分别处理
#     #     for cls_id in np.unique(np.concatenate([pred_labels, gt_labels])):
#     #         cls_name = self.class_names[cls_id - 1]
#     #
#     #         # 筛选该类别
#     #         pred_mask = pred_labels == cls_id
#     #         gt_mask = gt_labels == cls_id
#     #
#     #         cls_pred_boxes = pred_boxes[pred_mask]
#     #         cls_gt_boxes = gt_boxes[gt_mask]
#     #
#     #         if len(cls_pred_boxes) == 0 and len(cls_gt_boxes) == 0:
#     #             continue
#     #
#     #         if len(cls_pred_boxes) == 0:
#     #             # 全是 FN
#     #             self.all_matches[cls_name]['FN'] += len(cls_gt_boxes)
#     #             continue
#     #
#     #         if len(cls_gt_boxes) == 0:
#     #             # 全是 FP
#     #             self.all_matches[cls_name]['FP'] += len(cls_pred_boxes)
#     #             continue
#     #
#     #         # 计算 IoU
#     #         ious = iou3d_nms_utils.boxes_iou3d_gpu(
#     #             torch.from_numpy(cls_pred_boxes).cuda(),
#     #             torch.from_numpy(cls_gt_boxes).cuda()
#     #         ).cpu().numpy()
#     #
#     #         # 匹配（贪心匹配）
#     #         matched_gt = set()
#     #         iou_threshold = 0.7 if cls_name == 'Car' else 0.5
#     #
#     #         # 按 score 降序处理预测框
#     #         for pred_idx in np.argsort(-pred_scores[pred_mask]):
#     #             if pred_idx >= len(ious):
#     #                 break
#     #
#     #             max_iou_idx = ious[pred_idx].argmax()
#     #             max_iou = ious[pred_idx, max_iou_idx]
#     #
#     #             if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
#     #                 # TP
#     #                 self.all_matches[cls_name]['TP'] += 1
#     #                 matched_gt.add(max_iou_idx)
#     #                 self.all_ious[cls_name].append(max_iou)
#     #             else:
#     #                 # FP
#     #                 self.all_matches[cls_name]['FP'] += 1
#     #
#     #         # 未匹配的 GT 是 FN
#     #         self.all_matches[cls_name]['FN'] += len(cls_gt_boxes) - len(matched_gt)
#     #
#     # def _count_points_in_boxes(self, points, gt_boxes):
#     #     """统计每个box内的点数"""
#     #     for box in gt_boxes:
#     #         # 简化的点数统计（应该用旋转box，这里用AABB近似）
#     #         center = box[:3]
#     #         size = box[3:6]
#     #
#     #         in_box = np.all(np.abs(points - center) <= size / 2, axis=1)
#     #         point_count = np.sum(in_box)
#     #         self.all_point_counts.append(point_count)
#
#     def analyze_iou_distribution(self):
#         """分析 IoU 分布"""
#         print("分析 IoU 分布...")
#
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#         axes = axes.flatten()
#
#         for idx, cls_name in enumerate(self.class_names):
#             if cls_name not in self.all_ious or len(self.all_ious[cls_name]) == 0:
#                 continue
#
#             ious = self.all_ious[cls_name]
#             ax = axes[idx]
#
#             ax.hist(ious, bins=50, alpha=0.7, edgecolor='black')
#             ax.axvline(np.mean(ious), color='red', linestyle='--',
#                        label=f'Mean: {np.mean(ious):.3f}')
#             ax.axvline(np.median(ious), color='green', linestyle='--',
#                        label=f'Median: {np.median(ious):.3f}')
#             ax.set_xlabel('IoU')
#             ax.set_ylabel('Count')
#             ax.set_title(f'{cls_name} IoU Distribution')
#             ax.legend()
#             ax.grid(alpha=0.3)
#
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'iou_distribution.png', dpi=150)
#         plt.close()
#
#         # 保存统计数据
#         with open(self.output_dir / 'iou_stats.txt', 'w') as f:
#             f.write("IoU 统计\n")
#             f.write("=" * 60 + "\n")
#             for cls_name in self.class_names:
#                 if cls_name in self.all_ious and len(self.all_ious[cls_name]) > 0:
#                     ious = self.all_ious[cls_name]
#                     f.write(f"\n{cls_name}:\n")
#                     f.write(f"  Count:  {len(ious)}\n")
#                     f.write(f"  Mean:   {np.mean(ious):.4f}\n")
#                     f.write(f"  Median: {np.median(ious):.4f}\n")
#                     f.write(f"  Std:    {np.std(ious):.4f}\n")
#                     f.write(f"  Min:    {np.min(ious):.4f}\n")
#                     f.write(f"  Max:    {np.max(ious):.4f}\n")
#
#         print(f"IoU 分布已保存: {self.output_dir / 'iou_distribution.png'}")
#
#     def analyze_tp_fp_fn(self):
#         """分析 TP/FP/FN"""
#         print("分析 TP/FP/FN...")
#
#         # 计算 Precision 和 Recall
#         stats = {}
#         for cls_name in self.class_names:
#             if cls_name not in self.all_matches:
#                 continue
#
#             tp = self.all_matches[cls_name]['TP']
#             fp = self.all_matches[cls_name]['FP']
#             fn = self.all_matches[cls_name]['FN']
#
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#             f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#
#             stats[cls_name] = {
#                 'TP': tp, 'FP': fp, 'FN': fn,
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F1': f1
#             }
#
#         # 保存统计
#         with open(self.output_dir / 'tp_fp_fn_stats.txt', 'w') as f:
#             f.write("TP/FP/FN 统计\n")
#             f.write("=" * 80 + "\n")
#             f.write(f"{'Class':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
#             f.write("-" * 80 + "\n")
#
#             for cls_name, stat in stats.items():
#                 f.write(f"{cls_name:<15} "
#                         f"{stat['TP']:>6} "
#                         f"{stat['FP']:>6} "
#                         f"{stat['FN']:>6} "
#                         f"{stat['Precision']:>10.4f} "
#                         f"{stat['Recall']:>10.4f} "
#                         f"{stat['F1']:>10.4f}\n")
#
#         # 可视化
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#         classes = list(stats.keys())
#         precisions = [stats[c]['Precision'] for c in classes]
#         recalls = [stats[c]['Recall'] for c in classes]
#         f1s = [stats[c]['F1'] for c in classes]
#
#         axes[0].bar(classes, precisions)
#         axes[0].set_ylabel('Precision')
#         axes[0].set_title('Precision by Class')
#         axes[0].tick_params(axis='x', rotation=45)
#
#         axes[1].bar(classes, recalls)
#         axes[1].set_ylabel('Recall')
#         axes[1].set_title('Recall by Class')
#         axes[1].tick_params(axis='x', rotation=45)
#
#         axes[2].bar(classes, f1s)
#         axes[2].set_ylabel('F1 Score')
#         axes[2].set_title('F1 Score by Class')
#         axes[2].tick_params(axis='x', rotation=45)
#
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'precision_recall_f1.png', dpi=150)
#         plt.close()
#
#         print(f"TP/FP/FN 统计已保存: {self.output_dir / 'tp_fp_fn_stats.txt'}")
#
#     def generate_confusion_matrix(self):
#         """生成混淆矩阵"""
#         print("生成混淆矩阵...")
#
#         # 改进版：基于 TP 匹配
#         all_pred = []
#         all_gt = []
#
#         # 从 TP 收集匹配对
#         for cls_name in self.class_names:
#             if cls_name not in self.all_matches:
#                 continue
#
#             # 这里需要保存匹配的预测和GT类别
#             # 暂时跳过，因为需要修改 _compute_matches
#             pass
#
#         # 暂时用简化方案：只统计 TP 数量
#         print("⚠️  当前版本的混淆矩阵功能有限")
#         print("    建议查看 tp_fp_fn_stats.txt 了解各类别表现")
#
#         # 可视化 TP 数量
#         classes = [c for c in self.class_names if c in self.all_matches]
#         tp_counts = [self.all_matches[c]['TP'] for c in classes]
#
#         plt.figure(figsize=(10, 6))
#         plt.bar(classes, tp_counts)
#         plt.xlabel('Class')
#         plt.ylabel('True Positives')
#         plt.title('True Positives by Class')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'tp_by_class.png', dpi=150)
#         plt.close()
#
#         print(f"TP 统计已保存: {self.output_dir / 'tp_by_class.png'}")
#
#     def analyze_by_point_count(self):
#         """按点数分析性能"""
#         print("按点数分析性能...")
#
#         if len(self.all_point_counts) == 0:
#             print("没有点数信息，跳过此分析")
#             return
#
#         # 定义桶
#         buckets = {
#             'sparse (1-10 pts)': (1, 10),
#             'medium (10-50 pts)': (10, 50),
#             'dense (>50 pts)': (50, float('inf'))
#         }
#
#         # 统计每个桶的AP（简化版，完整版需要重新计算）
#         bucket_stats = defaultdict(lambda: {'count': 0, 'mean_points': 0})
#
#         for points in self.all_point_counts:
#             for bucket_name, (min_pts, max_pts) in buckets.items():
#                 if min_pts <= points < max_pts:
#                     bucket_stats[bucket_name]['count'] += 1
#                     bucket_stats[bucket_name]['mean_points'] += points
#
#         # 可视化
#         fig, ax = plt.subplots(figsize=(10, 6))
#
#         bucket_names = list(bucket_stats.keys())
#         counts = [bucket_stats[b]['count'] for b in bucket_names]
#
#         ax.bar(bucket_names, counts)
#         ax.set_ylabel('Number of Objects')
#         ax.set_title('Object Distribution by Point Count')
#         ax.tick_params(axis='x', rotation=45)
#
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'point_count_distribution.png', dpi=150)
#         plt.close()
#
#         # 保存统计
#         with open(self.output_dir / 'point_count_stats.txt', 'w') as f:
#             f.write("按点数分析\n")
#             f.write("=" * 60 + "\n")
#             for bucket_name, stats in bucket_stats.items():
#                 count = stats['count']
#                 if count > 0:
#                     mean_pts = stats['mean_points'] / count
#                     f.write(f"{bucket_name}:\n")
#                     f.write(f"  Count: {count}\n")
#                     f.write(f"  Mean points: {mean_pts:.1f}\n\n")
#
#         print(f"点数分析已保存: {self.output_dir / 'point_count_distribution.png'}")
#
#     def visualize_false_positives(self, max_vis=10):
#         """可视化误检（FP）"""
#         print(f"可视化前 {max_vis} 个 False Positives...")
#
#         fp_dir = self.output_dir / 'false_positives'
#         fp_dir.mkdir(exist_ok=True)
#
#         # TODO: 这需要访问原始点云数据
#         # 这里只是框架，完整实现需要加载点云并可视化
#
#         print(f"False Positives 可视化目录: {fp_dir}")
#         print("（完整实现需要加载点云数据进行3D可视化）")
#
#     def generate_pr_curve(self):
#         """生成 Precision-Recall 曲线"""
#         print("生成 PR 曲线...")
#
#         # 简化版：基于已有的 Precision 和 Recall
#         # 完整版需要在不同阈值下重新计算
#
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#         axes = axes.flatten()
#
#         for idx, cls_name in enumerate(self.class_names):
#             if cls_name not in self.all_matches:
#                 continue
#
#             ax = axes[idx]
#
#             # 简化：只绘制单点
#             tp = self.all_matches[cls_name]['TP']
#             fp = self.all_matches[cls_name]['FP']
#             fn = self.all_matches[cls_name]['FN']
#
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#
#             ax.plot([0, recall, 1], [1, precision, 0], 'b-o')
#             ax.set_xlabel('Recall')
#             ax.set_ylabel('Precision')
#             ax.set_title(f'{cls_name} PR Curve')
#             ax.grid(alpha=0.3)
#             ax.set_xlim([0, 1])
#             ax.set_ylim([0, 1])
#
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'pr_curves.png', dpi=150)
#         plt.close()
#
#         print(f"PR 曲线已保存: {self.output_dir / 'pr_curves.png'}")
#
#
# def main():
#     parser = argparse.ArgumentParser(description='详细评估分析')
#     parser.add_argument('--ckpt', type=str, required=True, help='checkpoint 路径')
#     parser.add_argument('--cfg_file', type=str, required=True, help='配置文件')
#     parser.add_argument('--output_dir', type=str, default='analysis_results',
#                         help='输出目录')
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--workers', type=int, default=4)
#
#     args = parser.parse_args()
#
#     # 加载配置
#     cfg_from_yaml_file(args.cfg_file, cfg)
#
#     # 创建logger
#     logger = common_utils.create_logger()
#     logger.info('开始详细评估分析')
#     cfg.DATA_CONFIG.DATA_SPLIT = {
#         'train': 'train',
#         'test': 'val'  # 强制使用 val
#     }
#
#     # 构建数据集
#     root_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'custom'
#     logger.info(f"Data root_path: {root_path}")
#     logger.info(f"Path exists: {root_path.exists()}")
#     # 🔥 检查 pkl 文件
#     pkl_file = root_path / 'custom_infos_val.pkl'
#     if not pkl_file.exists():
#         logger.error(f"❌ PKL 文件不存在: {pkl_file}")
#         logger.error("请先生成数据集信息文件！")
#         return
#
#     test_set, test_loader, _ = build_dataloader(
#         dataset_cfg=cfg.DATA_CONFIG,
#         class_names=cfg.CLASS_NAMES,
#         batch_size=args.batch_size,
#         dist=False,
#         workers=args.workers,
#         logger=logger,
#         training=False,
#         root_path=root_path
#     )
#     # 🔥 检查数据集大小
#     logger.info(f"Dataset size: {len(test_set)}")
#     if len(test_set) == 0:
#         logger.error("❌ 数据集为空！请检查配置和数据路径。")
#         return
#
#     # 构建模型
#     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
#                           dataset=test_set)
#     model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
#     model.cuda()
#
#     # 创建评估器
#     evaluator = DetailedEvaluator(
#         model=model,
#         dataloader=test_loader,
#         class_names=cfg.CLASS_NAMES,
#         output_dir=args.output_dir
#     )
#
#     # 运行评估
#     evaluator.evaluate()
#
#     logger.info(f'评估完成！结果保存在: {args.output_dir}')
#
#
# if __name__ == '__main__':
#     main()
# !/usr/bin/env python3
"""
详细评估分析脚本 - 修复版

主要修复：
1. 修复 _compute_matches 中的类别索引问题
2. 保存匹配对用于混淆矩阵
3. 正确生成混淆矩阵
4. 添加更多调试信息
"""

import argparse
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import sys

sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


class DetailedEvaluator:
    """详细评估器 - 修复版"""

    def __init__(self, model, dataloader, class_names, output_dir):
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 存储结果
        self.all_pred_boxes = []
        self.all_pred_scores = []
        self.all_pred_labels = []
        self.all_gt_boxes = []
        self.all_gt_labels = []
        self.all_ious = defaultdict(list)
        self.all_matches = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

        # 🔥 新增：保存匹配对用于混淆矩阵
        self.matched_pairs = []  # [(pred_label, gt_label), ...]

    def evaluate(self):
        """运行评估"""
        print("开始详细评估...")
        self.model.eval()

        total_samples = 0
        valid_samples = 0

        with torch.no_grad():
            for i, batch_dict in enumerate(self.dataloader):
                if i % 100 == 0:
                    print(f"Processing {i}/{len(self.dataloader)}...")

                total_samples += 1

                # 加载到 GPU
                self._load_data_to_gpu(batch_dict)

                # 前向传播
                pred_dicts, _ = self.model.forward(batch_dict)

                # 收集结果
                if self._collect_results(batch_dict, pred_dicts):
                    valid_samples += 1

        print(f"\n评估完成！")
        print(f"总样本数: {total_samples}")
        print(f"有效样本数: {valid_samples}")
        print(f"收集到的匹配数: {len(self.matched_pairs)}")
        print("\n开始分析...")

        # 生成所有分析
        self.analyze_iou_distribution()
        self.analyze_tp_fp_fn()
        self.generate_confusion_matrix()
        self.generate_pr_curve()

        print(f"\n分析完成！结果保存在: {self.output_dir}")

    def _load_data_to_gpu(self, batch_dict):
        """加载数据到GPU"""
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            if key in ['frame_id', 'metadata', 'calib']:
                continue
            batch_dict[key] = torch.from_numpy(val).float().cuda()

    def _collect_results(self, batch_dict, pred_dicts):
        """收集预测和GT结果"""
        batch_size = batch_dict.get('batch_size', 1)

        has_valid_data = False

        for batch_idx in range(batch_size):
            # 预测结果
            if batch_idx >= len(pred_dicts):
                break

            pred_boxes = pred_dicts[batch_idx]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[batch_idx]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[batch_idx]['pred_labels'].cpu().numpy()
            # 🔥 添加置信度过滤
            SCORE_THRESH = 0.3  # ← 提高到 0.3

            score_mask = pred_scores >= SCORE_THRESH
            pred_boxes = pred_boxes[score_mask]
            pred_scores = pred_scores[score_mask]
            pred_labels = pred_labels[score_mask]

            # GT
            gt_boxes = batch_dict['gt_boxes'][batch_idx].cpu().numpy()

            # 过滤 padding
            valid_gt = np.any(gt_boxes[:, :7] != 0, axis=1)
            gt_boxes = gt_boxes[valid_gt]

            if len(gt_boxes) == 0:
                continue

            gt_labels = gt_boxes[:, -1].astype(np.int32)
            gt_boxes = gt_boxes[:, :7]

            # 存储
            self.all_pred_boxes.append(pred_boxes)
            self.all_pred_scores.append(pred_scores)
            self.all_pred_labels.append(pred_labels)
            self.all_gt_boxes.append(gt_boxes)
            self.all_gt_labels.append(gt_labels)

            # 计算 IoU 和匹配
            self._compute_matches(pred_boxes, pred_scores, pred_labels,
                                  gt_boxes, gt_labels)

            has_valid_data = True

        return has_valid_data

    def _compute_matches(self, pred_boxes, pred_scores, pred_labels,
                         gt_boxes, gt_labels):
        """计算 TP/FP/FN 和 IoU - 修复版"""
        from pcdet.ops.iou3d_nms import iou3d_nms_utils

        # 对每个类别分别处理
        all_labels = np.concatenate([pred_labels, gt_labels])
        unique_labels = np.unique(all_labels)

        for cls_id in unique_labels:
            # 转换为 int
            cls_id = int(cls_id)

            # 检查类别ID是否有效
            if cls_id < 1 or cls_id > len(self.class_names):
                print(f"⚠️  跳过无效类别ID: {cls_id}")
                continue

            cls_name = self.class_names[cls_id - 1]

            # 筛选该类别
            pred_mask = pred_labels == cls_id
            gt_mask = gt_labels == cls_id

            cls_pred_boxes = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]
            cls_gt_boxes = gt_boxes[gt_mask]

            # 处理空情况
            if len(cls_pred_boxes) == 0 and len(cls_gt_boxes) == 0:
                continue

            if len(cls_pred_boxes) == 0:
                # 全是 FN
                self.all_matches[cls_name]['FN'] += len(cls_gt_boxes)
                continue

            if len(cls_gt_boxes) == 0:
                # 全是 FP
                self.all_matches[cls_name]['FP'] += len(cls_pred_boxes)
                # 🔥 记录FP（预测了但GT是空）
                for _ in range(len(cls_pred_boxes)):
                    self.matched_pairs.append((cls_id, 0))  # 0表示背景
                continue

            # 计算 IoU
            try:
                ious = iou3d_nms_utils.boxes_iou3d_gpu(
                    torch.from_numpy(cls_pred_boxes).cuda(),
                    torch.from_numpy(cls_gt_boxes).cuda()
                ).cpu().numpy()
            except Exception as e:
                print(f"⚠️  计算IoU失败 ({cls_name}): {e}")
                continue

            # IoU 阈值
            if 'Car' in cls_name or 'Truck' in cls_name:
                iou_threshold = 0.7
            elif 'Pedestrian' in cls_name or 'Cyclist' in cls_name:
                iou_threshold = 0.5
            else:
                iou_threshold = 0.5

            # 匹配（贪心匹配）
            matched_gt = set()
            matched_pred = set()

            # 按 score 降序处理预测框
            sorted_indices = np.argsort(-cls_pred_scores)

            for pred_idx in sorted_indices:
                if pred_idx >= len(ious):
                    break

                max_iou_idx = ious[pred_idx].argmax()
                max_iou = ious[pred_idx, max_iou_idx]

                if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
                    # TP
                    self.all_matches[cls_name]['TP'] += 1
                    matched_gt.add(max_iou_idx)
                    matched_pred.add(pred_idx)
                    self.all_ious[cls_name].append(max_iou)

                    # 🔥 记录匹配对（用于混淆矩阵）
                    self.matched_pairs.append((cls_id, cls_id))
                else:
                    # FP
                    self.all_matches[cls_name]['FP'] += 1
                    # 🔥 记录FP
                    self.matched_pairs.append((cls_id, 0))

            # 未匹配的 GT 是 FN
            fn_count = len(cls_gt_boxes) - len(matched_gt)
            self.all_matches[cls_name]['FN'] += fn_count

            # 🔥 记录FN（GT存在但没被预测到）
            for _ in range(fn_count):
                self.matched_pairs.append((0, cls_id))

    def analyze_iou_distribution(self):
        """分析 IoU 分布"""
        print("分析 IoU 分布...")

        if not any(self.all_ious.values()):
            print("⚠️  没有IoU数据")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, cls_name in enumerate(self.class_names):
            if idx >= len(axes):
                break

            if cls_name not in self.all_ious or len(self.all_ious[cls_name]) == 0:
                axes[idx].text(0.5, 0.5, f'{cls_name}\nNo Data',
                               ha='center', va='center')
                axes[idx].set_xlim([0, 1])
                axes[idx].set_ylim([0, 1])
                continue

            ious = self.all_ious[cls_name]
            ax = axes[idx]

            ax.hist(ious, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(ious), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ious):.3f}')
            ax.axvline(np.median(ious), color='green', linestyle='--',
                       label=f'Median: {np.median(ious):.3f}')
            ax.set_xlabel('IoU')
            ax.set_ylabel('Count')
            ax.set_title(f'{cls_name} IoU Distribution (n={len(ious)})')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'iou_distribution.png', dpi=150)
        plt.close()

        # 保存统计
        with open(self.output_dir / 'iou_stats.txt', 'w') as f:
            f.write("IoU 统计\n")
            f.write("=" * 60 + "\n\n")
            for cls_name in self.class_names:
                if cls_name in self.all_ious and len(self.all_ious[cls_name]) > 0:
                    ious = self.all_ious[cls_name]
                    f.write(f"{cls_name}:\n")
                    f.write(f"  Count:  {len(ious)}\n")
                    f.write(f"  Mean:   {np.mean(ious):.4f}\n")
                    f.write(f"  Median: {np.median(ious):.4f}\n")
                    f.write(f"  Std:    {np.std(ious):.4f}\n")
                    f.write(f"  Min:    {np.min(ious):.4f}\n")
                    f.write(f"  Max:    {np.max(ious):.4f}\n\n")

        print(f"IoU 分布已保存: {self.output_dir / 'iou_distribution.png'}")

    def analyze_tp_fp_fn(self):
        """分析 TP/FP/FN"""
        print("分析 TP/FP/FN...")

        if not self.all_matches:
            print("⚠️  没有TP/FP/FN数据")
            return

        stats = {}
        for cls_name in self.class_names:
            if cls_name not in self.all_matches:
                continue

            tp = self.all_matches[cls_name]['TP']
            fp = self.all_matches[cls_name]['FP']
            fn = self.all_matches[cls_name]['FN']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            stats[cls_name] = {
                'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }

        # 保存统计
        with open(self.output_dir / 'tp_fp_fn_stats.txt', 'w') as f:
            f.write("TP/FP/FN 统计\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Class':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
            f.write("-" * 80 + "\n")

            for cls_name, stat in stats.items():
                f.write(f"{cls_name:<15} "
                        f"{stat['TP']:>6} "
                        f"{stat['FP']:>6} "
                        f"{stat['FN']:>6} "
                        f"{stat['Precision']:>10.4f} "
                        f"{stat['Recall']:>10.4f} "
                        f"{stat['F1']:>10.4f}\n")

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        classes = list(stats.keys())
        if not classes:
            print("⚠️  没有类别数据")
            return

        precisions = [stats[c]['Precision'] for c in classes]
        recalls = [stats[c]['Recall'] for c in classes]
        f1s = [stats[c]['F1'] for c in classes]

        axes[0].bar(classes, precisions)
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision by Class')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim([0, 1])

        axes[1].bar(classes, recalls)
        axes[1].set_ylabel('Recall')
        axes[1].set_title('Recall by Class')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim([0, 1])

        axes[2].bar(classes, f1s)
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('F1 Score by Class')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_f1.png', dpi=150)
        plt.close()

        print(f"TP/FP/FN 统计已保存: {self.output_dir / 'tp_fp_fn_stats.txt'}")

    def generate_confusion_matrix(self):
        """生成混淆矩阵 - 修复版"""
        print("生成混淆矩阵...")

        if len(self.matched_pairs) == 0:
            print("⚠️  没有匹配对数据，无法生成混淆矩阵")
            return

        # 提取预测和GT标签
        all_pred = [p[0] for p in self.matched_pairs]
        all_gt = [p[1] for p in self.matched_pairs]

        print(f"匹配对数量: {len(self.matched_pairs)}")
        print(f"预测标签范围: {min(all_pred)} - {max(all_pred)}")
        print(f"GT标签范围: {min(all_gt)} - {max(all_gt)}")

        # 创建标签映射（包括背景类0）
        label_names = ['Background'] + self.class_names
        label_ids = list(range(len(label_names)))

        # 生成混淆矩阵
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(all_gt, all_pred, labels=label_ids)

        # 可视化
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names,
                    yticklabels=label_names,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()

        # 保存数值
        np.savetxt(self.output_dir / 'confusion_matrix.txt', cm, fmt='%d',
                   header=f'Confusion Matrix (rows=GT, cols=Pred)\nClasses: {", ".join(label_names)}')

        print(f"混淆矩阵已保存: {self.output_dir / 'confusion_matrix.png'}")

    def generate_pr_curve(self):
        """生成 PR 曲线"""
        print("生成 PR 曲线...")

        if not self.all_matches:
            print("⚠️  没有数据生成PR曲线")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, cls_name in enumerate(self.class_names):
            if idx >= len(axes):
                break

            if cls_name not in self.all_matches:
                axes[idx].text(0.5, 0.5, f'{cls_name}\nNo Data',
                               ha='center', va='center')
                axes[idx].set_xlim([0, 1])
                axes[idx].set_ylim([0, 1])
                continue

            ax = axes[idx]

            tp = self.all_matches[cls_name]['TP']
            fp = self.all_matches[cls_name]['FP']
            fn = self.all_matches[cls_name]['FN']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # 简化版PR曲线（单点）
            ax.plot([0, recall, 1], [1, precision, 0], 'b-o', linewidth=2)
            ax.scatter([recall], [precision], s=100, c='red', zorder=5,
                       label=f'P={precision:.3f}, R={recall:.3f}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{cls_name} PR Curve')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pr_curves.png', dpi=150)
        plt.close()

        print(f"PR 曲线已保存: {self.output_dir / 'pr_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description='详细评估分析 - 修复版')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint 路径')
    parser.add_argument('--cfg_file', type=str, required=True, help='配置文件')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()

    # 加载配置
    cfg_from_yaml_file(args.cfg_file, cfg)

    # 🔥 强制使用 val split
    cfg.DATA_CONFIG.DATA_SPLIT = {
        'train': 'train',
        'test': 'val'
    }

    # 创建logger
    logger = common_utils.create_logger()
    logger.info('开始详细评估分析')

    # 构建数据集
    root_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'custom'
    logger.info(f'Data root_path: {root_path}')

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
        root_path=root_path
    )

    logger.info(f'Dataset size: {len(test_set)}')

    if len(test_set) == 0:
        logger.error("❌ 数据集为空！")
        return

    # 构建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                          dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()

    # 创建评估器
    evaluator = DetailedEvaluator(
        model=model,
        dataloader=test_loader,
        class_names=cfg.CLASS_NAMES,
        output_dir=args.output_dir
    )

    # 运行评估
    evaluator.evaluate()

    logger.info(f'评估完成！结果保存在: {args.output_dir}')


if __name__ == '__main__':
    main()