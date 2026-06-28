# """
# python plot_pr.py \
#   --pkls pr_pointpillars.pkl pr_cross_attn.pkl pr_cross_mamba.pkl pr_cmamba.pkl \
#   --names "PointPillars" "Bi-Cross-Attn" "4-Dir Cross-Mamba" "CMambaFusion" \
#   --output pr_car_07.png
# """
#
# import argparse, pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # def compute_pr(scores, is_tp, n_gt):
# #     order     = np.argsort(-scores)
# #     is_tp     = is_tp[order]
# #     tp_cum    = np.cumsum(is_tp)
# #     precision = tp_cum / (tp_cum + np.cumsum(~is_tp) + 1e-9)
# #     recall    = tp_cum / (n_gt + 1e-9)
# #     return recall, precision
#
#
# def compute_pr(scores, is_tp, n_gt):
#     order = np.argsort(-scores)
#     is_tp = is_tp[order].astype(bool)
#     tp_cum = np.cumsum(is_tp)
#     fp_cum = np.cumsum(~is_tp)
#     precision = tp_cum / (tp_cum + fp_cum + 1e-9)
#     recall = tp_cum / (n_gt + 1e-9)
#     recall = np.concatenate([[0.0], recall, [recall[-1]]])
#     precision = np.concatenate([[1.0], precision, [0.0]])
#     precision_env = precision.copy()
#     for i in range(len(precision_env) - 2, -1, -1):
#         precision_env[i] = max(precision_env[i], precision_env[i + 1])
#
#     # ── 改这里：40-point 插值，和 KITTI eval 一致 ──
#     recall_thresholds = np.linspace(0.0, 1.0, 41)  # 40个间隔，41个点
#     ap = 0.0
#     for r in recall_thresholds:
#         # 找所有 recall >= r 的点里 precision 的最大值
#         prec_at_r = precision_env[recall >= r]
#         ap += prec_at_r.max() if len(prec_at_r) > 0 else 0.0
#     ap /= 41
#
#     return recall, precision_env, ap  # 注意现在返回三个值
#
#
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--pkls',  nargs='+', required=True)
#     p.add_argument('--names', nargs='+', required=True)
#     p.add_argument('--output', default='pr_curve.png')
#     args = p.parse_args()
#
#     COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
#     plt.figure(figsize=(7, 6))
#
#     for i, (pkl_path, name) in enumerate(zip(args.pkls, args.names)):
#         with open(pkl_path, 'rb') as f:
#             d = pickle.load(f)
#         scores, is_tp, n_gt = d['scores'], d['is_tp'], d['n_gt']
#         recall, precision = compute_pr(scores, is_tp, n_gt)
#         ap = np.trapz(precision, recall)
#         plt.plot(recall, precision, color=COLORS[i % len(COLORS)],
#                  linewidth=2, label=f'{name}  (AP={ap:.3f})')
#         print(f"{name}: AP={ap:.4f}")
#
#     plt.xlabel('Recall', fontsize=13)
#     plt.ylabel('Precision', fontsize=13)
#     plt.title(f'PR Curve — {d["class"]}  (3D IoU={d["iou"]})', fontsize=14)
#     plt.legend(fontsize=11)
#     plt.grid(True, alpha=0.3)
#     plt.xlim([0, 1]); plt.ylim([0, 1.02])
#     plt.tight_layout()
#     plt.savefig(args.output, dpi=150)
#     print(f"已保存: {args.output}")
#
#
# if __name__ == '__main__':
#     main()

"""
python plot_pr.py \
  --pkls pr_cross_attn_car.pkl pr_cross_mamba_car.pkl pr_cmamba_car.pkl \
  --names "Bi-Cross-Attn" "4-Dir Cross-Mamba" "CMambaFusion" \
  --output pr_curve.png
"""

import argparse, pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
})

COLORS = ['#E06C75', '#61AFEF', '#98C379', '#C678DD']
STYLES = [':', '--', '-.', '-']


def compute_pr(scores, is_tp, n_gt):
    order     = np.argsort(-scores)
    is_tp     = is_tp[order].astype(bool)
    tp_cum    = np.cumsum(is_tp)
    fp_cum    = np.cumsum(~is_tp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall    = tp_cum / (n_gt + 1e-9)

    # 加首尾点保证曲线完整
    recall    = np.concatenate([[0.0], recall,    [recall[-1]]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # 单调包络（标准做法）
    precision_env = precision.copy()
    for i in range(len(precision_env) - 2, -1, -1):
        precision_env[i] = max(precision_env[i], precision_env[i + 1])

    # # ── 40-point 插值，与 KITTI eval 一致 ──
    # recall_thresholds = np.linspace(0.0, 1.0, 41)
    # ap = 0.0
    # for r in recall_thresholds:
    #     prec_at_r = precision_env[recall >= r]
    #     ap += prec_at_r.max() if len(prec_at_r) > 0 else 0.0
    # ap /= 41
    # 完全对应官方 get_mAP_R40
    recall_thresholds = np.arange(0., 1.01, 0.025)  # 41个点，index 0~40
    ap = 0.0
    prec_at_thresholds = []
    for r in recall_thresholds:
        prec_at_r = precision_env[recall >= r]
        prec_at_thresholds.append(prec_at_r.max() if len(prec_at_r) > 0 else 0.0)

    # 跳过 index 0，从 index 1 开始加，共40个值
    ap = sum(prec_at_thresholds[1:]) / 40   # ×100 转成百分比

    return recall, precision_env, ap   # ← 返回三个值


def plot_one_class(ax, pkls, names, class_name, title):
    n_models = len(names)  # 4个模型
    matched = 0
    for pkl_path in pkls:
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        if d['class'] != class_name:
            continue  # 不打印warning，静默跳过

        # 按顺序分配 name 和 style
        i = matched
        name = names[i % n_models]
        matched += 1

        scores, is_tp, n_gt = d['scores'], d['is_tp'], d['n_gt']
        if len(scores) == 0 or n_gt == 0:
            continue

        recall, precision, ap = compute_pr(scores, is_tp, n_gt)
        ax.plot(recall, precision,
                color=COLORS[i % len(COLORS)],
                linestyle=STYLES[i % len(STYLES)],
                linewidth=2.0,
                label=f'{name})'
                # label=f'{name} (AP={ap * 100:.2f})'
                )

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(direction='in', length=4)
# def plot_one_class(ax, pkls, names, class_name, title):
#     for i, (pkl_path, name) in enumerate(zip(pkls, names)):
#         with open(pkl_path, 'rb') as f:
#             d = pickle.load(f)
#         if d['class'] != class_name:
#             print(f"⚠️  {pkl_path} 存的是 {d['class']}，跳过")
#             continue
#         scores, is_tp, n_gt = d['scores'], d['is_tp'], d['n_gt']
#         if len(scores) == 0 or n_gt == 0:
#             continue
#
#         # ── 改动在这里：解包三个返回值，AP×100 与表格单位一致 ──
#         recall, precision, ap = compute_pr(scores, is_tp, n_gt)
#         ax.plot(recall, precision,
#                 color=COLORS[i % len(COLORS)],
#                 linestyle=STYLES[i % len(STYLES)],
#                 linewidth=2.0,
#                 label=f'{name} (AP={ap * 100:.2f})')   # ← AP×100
#
#     ax.set_title(title, fontsize=13, fontweight='bold')
#     ax.set_xlabel('Recall', fontsize=12)
#     ax.set_ylabel('Precision', fontsize=12)
#     ax.set_xlim([0.0, 1.0])
#     ax.set_ylim([0.0, 1.05])
#     ax.legend(fontsize=10, loc='lower left')
#     ax.grid(True, linestyle='--', alpha=0.4)
#     ax.tick_params(direction='in', length=4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pkls',   nargs='+', required=True)
    p.add_argument('--names',  nargs='+', required=True)
    p.add_argument('--iou',    type=float, default=0.7)
    p.add_argument('--output', default='pr_curve.png')
    args = p.parse_args()

    # classes = [
    #     ('Car',        f'Car (IoU={args.iou})'),
    #     ('Pedestrian', 'Pedestrian (IoU=0.5)'),
    #     ('Scooter',    'Scooter (IoU=0.5)'),
    # ]
    classes = [
        ('Car', f'Car (IoU={args.iou})'),
        ('Pedestrian', 'Pedestrian (IoU=0.5)'),
        ('PickupTruck', f'PickupTruck (IoU={args.iou})'),
        ('Cyclist', 'Cyclist (IoU=0.5)'),
        ('Scooter', 'Scooter (IoU=0.5)'),
        ('Bus', f'Bus (IoU={args.iou})'),
        ('MediumTruck', f'MediumTruck (IoU={args.iou})'),
        ('Train', f'Train (IoU={args.iou})'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.subplots_adjust(wspace=0.35, hspace=0.45)

    for ax, (cls_name, title) in zip(axes.flatten(), classes):
        plot_one_class(ax, args.pkls, args.names, cls_name, title)
    # fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # fig.subplots_adjust(wspace=0.32)

    # for ax, (cls_name, title) in zip(axes, classes):
    #     plot_one_class(ax, args.pkls, args.names, cls_name, title)

    plt.suptitle('Precision–Recall Curves on PALIN Dataset',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {args.output}")


if __name__ == '__main__':
    main()

# import pickle
#
# for name in ['pr_pp_pedestrian.pkl', 'pr_bi-cross_attention_pedestrian.pkl',
#              'pr_4d_cross_mamba_pedestrian.pkl', 'pr_cmamba_pedestrian.pkl']:
#     with open(name, 'rb') as f:
#         d = pickle.load(f)
#     print(f"{name}: class={d['class']}, n_gt={d['n_gt']}, n_pred={len(d['scores'])}")
