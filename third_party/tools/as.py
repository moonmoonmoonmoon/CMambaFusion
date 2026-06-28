# """
# 用 result.pkl + custom_infos_val.pkl 走官方 kitti_eval，直接提取 PR 数据画图
# """
# import sys, pickle
# import numpy as np
#
# sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
# from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
#
# # 读 GT
# with open('/home/yanan/Downloads/projects/multimodal_detection/data/custom/custom_infos_val.pkl', 'rb') as f:
#     infos = pickle.load(f)
#
# # 读预测结果
# with open('/home/yanan/Downloads/projects/multimodal_detection/third_party/output/cfgs/custom_models/pointpillar/6264/test_base_74/eval/epoch_74/test/default/result.pkl', 'rb') as f:
#     results = pickle.load(f)
#
# # 把 results 按 frame_id 建索引
# pred_dict = {r['frame_id']: r for r in results}
#
# # 组织成 kitti_eval 需要的格式
# gt_annos, dt_annos = [], []
# for info in infos:
#     frame_id = info['point_cloud']['lidar_idx']
#
#     # GT
#     gt_annos.append(info['annos'])
#
#     # 预测
#     if frame_id in pred_dict:
#         r = pred_dict[frame_id]
#         dt_annos.append({
#             'name': r['name'],
#             'score': r['score'],
#             'boxes_lidar': r['boxes_lidar'],
#         })
#     else:
#         dt_annos.append({
#             'name': np.array([]),
#             'score': np.array([]),
#             'boxes_lidar': np.zeros((0, 7)),
#         })
#
# print(f"GT帧数: {len(gt_annos)}, 预测帧数: {len(dt_annos)}")
