# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Revised by Yanan
# All Rights Reserved 2024.


import numpy as np
import pickle
# import tensorflow as tf
# from google.protobuf import text_format
# from waymo_open_dataset.metrics.python import detection_metrics
# from waymo_open_dataset.protos import metrics_pb2
import argparse

import os
import copy
import time
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from tabulate import tabulate
import pandas as pd
from matplotlib import pyplot as plt
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.once.once_eval.evaluation import iou3d_kernel_with_heading
from pcdet.utils import common_utils
from ..dataset_tools import load_pcd

# tf.get_logger().setLevel("INFO")


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


class MyDetMetric:
    def __init__(
        self,
        iou_cfg=None,
        difficulty_cfg=None,
        range_cfg=None,
        angle_cfg=None,
        save_path=None,
        point_cloud_range=None,
        target_precisions=(0.90, 0.85, 0.8),
        score_thresh=0.0,
        class_names=None,
        obstacle_overlay_thresh=0.2,
        logger=None,
    ):

        self.logger = logger
        default_class_names =  ['Vehicle', 'Pedestrian', 'Cyclist', 'Bigcar', 'Tricycle', 'Cone', 'Barrier']
        self.class_names = default_class_names if class_names is None else class_names
        self.class_index_map = {class_name: index+1 for index, class_name in enumerate(self.class_names)}
        default_iou_cfg = {"Vehicle": [0.7, 0.5], "Pedestrian": [0.5, 0.25], "Cyclist": [0.5], "Bigcar": [0.7, 0.5], 
                           "Tricycle": [0.5], "Cone": [0.5, 0.25], "Barrier": [0.5, 0.25]}  # {type:[iou11，iou2...]}
        self.eval_iou_cfg = self.filter_cfg(default_iou_cfg, iou_cfg)

        default_difficulty_cfg = [0, 1, 2]
        self.eval_difficulty_cfg = default_difficulty_cfg if difficulty_cfg is None else difficulty_cfg

        self.vehicle_position_remap = {"CIPV": 0, "ALAV": 1, "OTHERS": 2, "RISK": 0}
        self.risk_remap = {"AT_RISK": 0, "NO_RISK": 1, "RISK_VRU": 0}

        default_range_cfg = {
            "Vehicle": [[0, 50], [50, 100], [100, 150]],
            "Pedestrian": [[0, 40], [40, 80]],
            "Cyclist": [[0, 40], [40, 80]],
            "Bigcar": [[0, 50], [50, 100], [100, 150]],
            "Tricycle": [[0, 40], [40, 80]],
            "Cone": [[0, 40], [40, 80]],
            "Barrier": [[0, 40], [40, 80]],
        }
        self.eval_range_cfg = self.filter_cfg(default_range_cfg, range_cfg)

        default_angle_cfg = {
            "Vehicle": [[-60, -30], [-30, 30], [30, 60]],
            "Pedestrian": [[-60, -30], [-30, 30], [30, 60]],
            "Cyclist": [[-60, -30], [-30, 30], [30, 60]],
            "Bigcar": [[-60, -30], [-30, 30], [30, 60]],
            "Tricycle": [[-60, -30], [-30, 30], [30, 60]],
            "Cone": [[-60, -30], [-30, 30], [30, 60]],
            "Barrier": [[-60, -30], [-30, 30], [30, 60]],
        }
        self.eval_angle_cfg = self.filter_cfg(default_angle_cfg, angle_cfg)

        default_confusion_range_config = {"Vehicle": [0, 140], "Pedestrian": [0, 50], "Cyclist": [0, 50], "Bigcar": [0, 140],
                                          "Tricycle": [0, 50], "Cone": [0, 50], "Barrier": [0, 50]}
        self.confusion_eval_range = self.filter_cfg(default_confusion_range_config, None)
        self.obstacle_confusion_eval_range = {"x": [0, 50], "y": [-20, 20]}

        self.point_cloud_range = point_cloud_range
        self.eval_type_list = list(self.eval_iou_cfg.keys())
        self.score_thresh = score_thresh
        self.confusion_eval_score = 0.4
        self.obstacle_overlay_thresh = obstacle_overlay_thresh
        self.target_precisions = target_precisions if isinstance(target_precisions, (list, tuple)) else (target_precisions)
        if save_path is not None:
            self.save_path = Path(save_path)
            self.figs_save_path = self.save_path / "eval_figs"
            self.figs_save_path.mkdir(parents=True, exist_ok=True)
            self.excel_name = f"eval_result-{time.strftime('%Y%m%d_%H%M%S',time.localtime())}.xlsx"
        self.init_log()

    def filter_cfg(self, default_cfg, input_cfg):
        ori_cfg = default_cfg if input_cfg is None else input_cfg
        new_cfg = {}
        for k, v in ori_cfg.items():
            if k in self.class_names:
                new_idx = self.class_names.index(k) + 1
                new_cfg[new_idx] = v
        return new_cfg

    def output_log(self, msg, string=None, log_type="INFO", padding=None):
        assert log_type in ["INFO", "WARN", "ERROR"], "Only support log type in [I, W, E]"
        string_time = f"[TIME] {time.strftime('%Y-%D %H:%M:%S',time.localtime())}  {log_type} : "
        string_msg = string_time + msg + "\n"
        if padding is not None:
            string_padding = "".join([padding for i in range(10)])
            string_msg = string_time + "\n" + string_padding + msg + string_padding + "\n"
        else:
            string_msg = string_time + msg + "\n"

        if self.logger is not None:
            if log_type == "INFO":
                self.logger.info(msg)
            elif log_type == "WARN":
                self.logger.warning(msg)
            elif log_type == "ERROR":
                self.logger.error(msg)
            else:
                raise NotImplementedError
        else:
            print(string_msg)

        if string is not None:
            string = string + string_msg
            return string

    def init_log(self):
        self.output_log(f"My eval - class names: {self.class_names}")
        self.output_log(f"My eval - prediction score threshold: {self.score_thresh}")
        self.output_log(f"My eval - point cloud range: {self.point_cloud_range}")
        self.output_log(f"My eval - eval iou config: {self.eval_iou_cfg}")
        self.output_log(f"My eval - eval difficulty config: {self.eval_difficulty_cfg}")
        self.output_log(f"My eval - eval range config: {self.eval_range_cfg}")
        self.output_log(f"My eval - eval angle config: {self.eval_angle_cfg}")
        self.output_log(f"My eval - return recall&score at target precisions: {self.target_precisions}")
        self.output_log(f"My eval - result save path: {self.save_path}")
        self.output_log(f"My eval - result figures save path: {self.figs_save_path}")
        self.output_log(f"My eval - result excel save path: {self.save_path / self.excel_name}")

    def _mask_boxes_by_range(self, boxes, limit_range):
        if boxes.shape[0] <= 0:
            return boxes
        mask = (boxes[:, 0] >= limit_range[0]) & (boxes[:, 0] <= limit_range[3]) & (boxes[:, 1] >= limit_range[1]) & (boxes[:, 1] <= limit_range[4])
        return boxes[mask]

    def _split_bbox_by_cls(self, all_result, all_gts):
        bboxes_by_cls = {}
        dets_gts = {"gts": [], "dets": []}
        for cls in self.eval_type_list:
            bboxes_by_cls.update({cls: {"gts": [], "dets": []}})
            frame_step = len(all_result) // 10
        for frame_idx, frame_infos in enumerate(all_result):
            # gt: xyz,lwh, yaw, cls, det_level
            gts = all_gts[frame_idx]["gt_boxes_lidar"]
            gts_name = all_gts[frame_idx]['name']
            gts_id = np.array([self.class_index_map[element] if element in self.class_index_map else -1 for element in gts_name])
            # gts_id = all_gts[frame_idx]["label_id"]
            gts_diff = all_gts[frame_idx]["difficulty"]
            if "vehicle_relation" in all_gts[frame_idx] :
                gts_vr = np.array([self.vehicle_position_remap.get(vr, 3) for vr in all_gts[frame_idx]["vehicle_relation"]])
            else:
                if "vehicle_relation_add" in all_gts[frame_idx]:
                    gts_vr = np.array([self.vehicle_position_remap.get(vr, 3) for vr in all_gts[frame_idx]["vehicle_relation_add"]])
                    assert gts_vr.shape == gts_diff.shape
                else:
                    gts_vr = np.ones_like(gts_diff) * 2
            if "object_risk" in all_gts[frame_idx]:
                gts_risk = np.array([self.risk_remap.get(vr, 2) for vr in all_gts[frame_idx]["object_risk"]])
            else:
                if "vehicle_relation_add" in all_gts[frame_idx]:
                    gts_risk = np.array([self.risk_remap.get(vr, 2) for vr in all_gts[frame_idx]["vehicle_relation_add"]])
                    assert gts_risk.shape == gts_diff.shape
                else:
                    gts_risk = np.ones_like(gts_diff) * 1

            # gts: gt(0-6)/id(7)/difficulty(8)/vehicle_relation(9)/risk(10)/status(11)
            gts = np.concatenate([gts, gts_id[..., None], gts_diff[..., None], gts_vr[..., None], gts_risk[..., None], np.zeros_like(gts_id[..., None])], axis=1)  # label_id
            # dets: xyz（0-2）, lwh（3-5）, yaw（6）, cls（7）, score（8）, frame_idx（9）
            dets_name = frame_infos['name']
            dets_id = np.array([self.class_index_map[element] if element in self.class_index_map else -1 for element in dets_name])
            dets = np.concatenate(
                [
                    frame_infos["boxes_lidar"],
                    dets_id[..., None],
                    frame_infos["score"][..., None],
                    np.ones_like(frame_infos["score"][..., None]) * frame_idx,
                ],
                axis=1,
            )

            if self.point_cloud_range is not None:
                gts = self._mask_boxes_by_range(gts, self.point_cloud_range)

            dets_gts["gts"].append(gts)
            dets_gts["dets"].append(dets)
            for cls in self.eval_type_list:
                valid_gts = gts[np.logical_or(gts[:, 7] == cls, gts[:, 7] == -1)]
                valid_dets = dets[dets[:, 7] == cls]
                bboxes_by_cls[cls]["gts"].append(valid_gts)
                bboxes_by_cls[cls]["dets"].append(valid_dets)
        return bboxes_by_cls, dets_gts

    def _filtered_pred_boxes(self, boxes, range_list, score_thresh=None, score_axis=8):
        if score_thresh is not None:
            boxes = boxes[boxes[:, score_axis] >= score_thresh]
        if range_list is not None:
            # [TODO] sjw: better way to calculate corner of the boxes instead of center_x + l/2
            range_mask = np.logical_and((boxes[:, 0] + boxes[:, 3] / 2) > range_list[0], (boxes[:, 0] - boxes[:, 3] / 2) < range_list[1])
            boxes = boxes[range_mask]
        return boxes

    def _filtered_pred_boxes_by_range_and_cls(self, boxes, range_list, cls_name):
        range_remove_mask = np.logical_or((boxes[:, 0] + boxes[:, 3] / 2) < range_list[0], (boxes[:, 0] - boxes[:, 3] / 2) > range_list[1])
        cls_mask = boxes[:, 7] == cls_name
        remove_mask = np.logical_and(range_remove_mask, cls_mask)
        keep_mask = np.logical_not(remove_mask)
        boxes = boxes[keep_mask, :]
        return boxes
    
    def get_lidar(self, lidar_file, num_features=4):
        if isinstance(lidar_file, str):
            lidar_file = Path(lidar_file)
        assert lidar_file.exists()
        points = load_pcd.get_points_from_pcd_file(lidar_file, num_features=num_features)
        return points

    def _filtered_boxes_by_angle(self, boxes, gt_boxes, limit_angle, box_tolerance=0.05):
        if limit_angle is not None:
            left_boundary_boxes_yaw = np.arctan2(boxes[:, 1], boxes[:, 0] - box_tolerance)/ np.pi * 180
            right_boundary_boxes_yaw = np.arctan2(boxes[:, 1], boxes[:, 0] + box_tolerance)/ np.pi * 180
            boxes_mask = np.logical_and(right_boundary_boxes_yaw >= limit_angle[0], left_boundary_boxes_yaw < limit_angle[1])
            boxes = boxes[boxes_mask]

            for idx in range(len(gt_boxes)):
                gts = gt_boxes[idx]
                gts_yaw = np.arctan2(gts[:, 1], gts[:, 0]) / np.pi * 180
                gts_mask = np.logical_and(gts_yaw >= limit_angle[0], gts_yaw < limit_angle[1])
                gt_boxes[idx] = gts[gts_mask]
        return boxes, gt_boxes

    def _ordered_boxes(self, boxes, ordered_axis=8):
        order = np.argsort(boxes[:, ordered_axis])[::-1]
        return boxes[order]

    def _filtered_gt_boxes(self, boxes, range_list):
        for idx in range(len(boxes)):
            gts = boxes[idx]
            range_gt_mask = np.logical_and((gts[:, 0] + gts[:, 3] / 2) > range_list[0], (gts[:, 0] - gts[:, 3] / 2) < range_list[1])
            boxes[idx] = gts[range_gt_mask]
        return boxes

    def _filter_gt_boxes_by_axis(self, boxes, range_list, axis_type):
        for idx in range(len(boxes)):
            if axis_type == "x":
                gts = boxes[idx]
                range_gt_mask = np.logical_and((gts[:, 0] + gts[:, 3] / 2) > range_list[0], (gts[:, 0] - gts[:, 3] / 2) < range_list[1])
                boxes[idx] = gts[range_gt_mask]
            elif axis_type == "y":
                gts = boxes[idx]
                range_gt_mask = np.logical_and((gts[:, 1] + gts[:, 4] / 2) > range_list[0], (gts[:, 1] - gts[:, 4] / 2) < range_list[1])
                boxes[idx] = gts[range_gt_mask]
            else:
                raise TypeError(f"axis type:{axis_type} not know!")
        return boxes

    def _filtered_gt_boxes_by_range_and_cls(self, boxes, range_list, cls_name):
        for idx in range(len(boxes)):
            gts = boxes[idx]
            range_remove_mask = np.logical_or((gts[:, 0] + gts[:, 3] / 2) < range_list[0], (gts[:, 0] - gts[:, 3] / 2) > range_list[1])
            cls_type_mask = gts[:, 7] == cls_name
            remove_mask = np.logical_and(range_remove_mask, cls_type_mask)
            keep_mask = np.logical_not(remove_mask)
            boxes[idx] = gts[keep_mask, :]
        return boxes

    def _get_part_gts(self, annotation_cls, frames_idx):
        # gt(0-6)/id(7)/difficulty(8)/vehicle_relation(9)/risk(10)/status(11)/gt_idx_inframes(12)/det_idx_in_part(13)
        frames_gts_list = []
        for i_in_part, f_idx in enumerate(frames_idx):
            if annotation_cls[f_idx].shape[0] > 0:
                gt_temp = np.concatenate(
                    [annotation_cls[f_idx], np.arange(annotation_cls[f_idx].shape[0])[..., np.newaxis], np.ones_like(annotation_cls[f_idx][:, :1]) * i_in_part], axis=1
                )
            else:
                # do not set boxes size = 0 (bug in rotate_iou_gpu_eval)
                gt_temp = np.array([[-100, -100, -100, 0.01, 0.01, 0.01, 0, 0, 10, 10, -1, 0, -1, i_in_part]])
            frames_gts_list.append(gt_temp)
        frames_gts = np.concatenate(frames_gts_list, axis=0)
        return frames_gts

    def _get_part_ious(self, frames_dets, frames_gts, iou_type):
        if iou_type == "2D":
            frames_dets_bev = np.concatenate([frames_dets[:, 0:2], frames_dets[:, 3:5], frames_dets[:, 6][..., np.newaxis]], axis=1)
            frames_gts_bev = np.concatenate([frames_gts[:, 0:2], frames_gts[:, 3:5], frames_gts[:, 6][..., np.newaxis]], axis=1)
            ious = rotate_iou_gpu_eval(frames_dets_bev, frames_gts_bev).astype(np.float64)
        elif iou_type == "3D":
            ious = iou3d_kernel_with_heading(frames_dets, frames_gts)
        else:
            NotImplementedError

        _, frames_gts_inv = np.unique(frames_gts[:, 13], return_inverse=True)
        ious_mask = np.eye(frames_dets.shape[0], dtype=bool)[frames_gts_inv]
        ious_mask = np.transpose(ious_mask)
        ious_masked = ious * ious_mask
        max_iou = ious_masked.max(axis=1)
        max_iou_idx = ious_masked.argmax(axis=1)
        return max_iou, max_iou_idx

    def _get_part_overlay(self, frames_dets, frames_gts, iou_type):
        if iou_type == "2D":
            frames_dets_bev = np.concatenate([frames_dets[:, 0:2], frames_dets[:, 3:5], frames_dets[:, 6][..., np.newaxis]], axis=1)
            frames_gts_bev = np.concatenate([frames_gts[:, 0:2], frames_gts[:, 3:5], frames_gts[:, 6][..., np.newaxis]], axis=1)
            ious = rotate_iou_gpu_eval(frames_dets_bev, frames_gts_bev).astype(np.float64)
        elif iou_type == "3D":
            ious = iou3d_kernel_with_heading(frames_dets, frames_gts)
        else:
            NotImplementedError

        det_sizes = frames_dets[:, 3] * frames_dets[:, 4]
        gt_sizes = frames_gts[:, 3] * frames_gts[:, 4]

        _, frames_gts_inv = np.unique(frames_gts[:, 13], return_inverse=True)
        ious_mask = np.eye(frames_dets.shape[0], dtype=bool)[frames_gts_inv]
        ious_mask = np.transpose(ious_mask)
        ious_masked = ious * ious_mask
        max_iou = ious_masked.max(axis=1)
        max_iou_idx = ious_masked.argmax(axis=1)
        union = (det_sizes + gt_sizes[max_iou_idx]) * max_iou / (1 + max_iou)
        max_overlay = union / (gt_sizes[max_iou_idx] + 1e-6)
        return max_overlay, max_iou_idx

    def _calculate_ap(self, ap_recalls, recalls_per_cls, precisions_per_cls, score_per_cls):
        ap_precisions = []
        ap_scores_thresholds = []

        for recall_level in ap_recalls:
            try:
                args = np.argwhere(recalls_per_cls >= recall_level).flatten()
                candidate_p = precisions_per_cls[args]
                prec = max(candidate_p)
                pr_thres = score_per_cls[args[0]]
            except ValueError:
                prec = 0.0
                pr_thres = 0.0
            ap_precisions.append(prec)
            ap_scores_thresholds.append(pr_thres)

        ap_scores_thresholds = np.array(ap_scores_thresholds)
        ap_precisions = np.array(ap_precisions)
        return ap_precisions, ap_scores_thresholds

    def _calculate_tp_attr(self, angle_diffs, boxes_diffs, overlaps):
        if len(angle_diffs) > 0:
            angle_diffs = np.array(angle_diffs)
            mean_angle_diff = np.mean(np.minimum(np.array(angle_diffs), np.pi - np.array(angle_diffs))) / np.pi * 180
            flip_percentage = np.mean(angle_diffs > np.pi / 2)
            mean_iou = float(np.mean(overlaps))
            mean_boxes_diff = np.round(np.mean(np.array(boxes_diffs), axis=0), 3)
            return mean_iou, mean_angle_diff, flip_percentage, mean_boxes_diff
        else:
            return np.nan, np.nan, np.nan, np.array([np.nan] * 7)

    def _save_pr_fig(self, recalls_per_cls, precisions_per_cls, score_per_cls, plt_suffix):
        filename = "pr_{}.png".format(plt_suffix)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.axis("square")
        ax1.set_xlabel("recall")
        ax1.set_ylabel("")
        ax1.axis([0, 1.0, 0, 1.0])
        ax1.plot(recalls_per_cls, precisions_per_cls, label="dense_precision", color="b")
        ax1.plot(recalls_per_cls, score_per_cls, label="dense_score_threshold", color="g")
        ax1.locator_params("x", nbins=10)
        ax1.locator_params("y", nbins=10)
        ax1.legend()
        plt.savefig(os.path.join(self.figs_save_path, filename))
        plt.cla()
        plt.close(fig)

    def _eval_for_cls_at_iou_part(self, bboxes_by_cls, iou_threshold, plt_suffix, difficulty=0, difficulty_index=8, part_num=50, range_list=None, angle_list=None, iou_type="2D", risk_eval=False):
        # filtered and sorted detection bboxes
        dets_by_cls_list = np.concatenate(bboxes_by_cls["dets"], axis=0)
        dets_by_cls_list = self._filtered_pred_boxes(dets_by_cls_list, range_list, self.score_thresh)

        # filtered and sorted gt bboxes
        annotation_cls = copy.deepcopy(bboxes_by_cls["gts"])
        if range_list is not None:
            annotation_cls = self._filtered_gt_boxes(annotation_cls, range_list)

        if angle_list is not None:
            dets_by_cls_list, annotation_cls = self._filtered_boxes_by_angle(dets_by_cls_list, annotation_cls, angle_list)

        dets_all_sorted = self._ordered_boxes(dets_by_cls_list, 8)

        num_gt_boxes = sum([np.logical_and(gts[:, difficulty_index] <= difficulty, gts[:, 7] != -1).sum() for gts in annotation_cls])
        tp, fp, fn, class_remain, repeat_tp, dontcare = 0, 0, num_gt_boxes, num_gt_boxes, 0, 0
        num_det_boxes, num_matched = 0, 0
        angle_diffs, boxes_diffs = [], []
        precisions_per_cls, recalls_per_cls, score_per_cls, overlaps = [], [], [], []

        part_split_num = np.ceil(dets_all_sorted.shape[0] / part_num).astype(np.int32)
        for i in range(part_split_num):
            det_start = i * part_num
            det_end = (i + 1) * part_num
            if det_end > dets_all_sorted.shape[0]:
                det_end = dets_all_sorted.shape[0]
            frames_dets = dets_all_sorted[det_start:det_end]
            frames_idx = frames_dets[:, 9].astype(np.int32)
            frames_gts = self._get_part_gts(annotation_cls, frames_idx)

            max_iou_partlist, max_iou_idx_partlist = self._get_part_ious(frames_dets, frames_gts, iou_type=iou_type)
            for i_part_num in range(part_num):
                dontcare_flag = False
                if i_part_num >= frames_dets.shape[0]:
                    break
                det = frames_dets[i_part_num]
                max_iou = max_iou_partlist[i_part_num]
                max_iou_idx = max_iou_idx_partlist[i_part_num]
                if risk_eval == True:
                    ignore_iou_threshold = 0.01
                else:
                    ignore_iou_threshold = iou_threshold / 2.0
                if max_iou >= ignore_iou_threshold and max_iou <= 1.0:
                    matched_gt_idx = int(frames_gts[max_iou_idx][12])
                    anno_idx = int(frames_dets[i_part_num, 9])
                    matched_gt = annotation_cls[anno_idx][matched_gt_idx]
                    if matched_gt[difficulty_index] > difficulty or matched_gt[7] == -1:
                        dontcare_flag = True
                        dontcare += 1

                    if max_iou >= iou_threshold:
                        if dontcare_flag == False:
                            if matched_gt[11] == 1:
                                dontcare_flag = True
                                repeat_tp += 1
                            else:
                                matched_gt[11] = 1

                        if dontcare_flag == False:
                            num_matched += 1
                            tp += 1
                            fn -= 1
                            class_remain -= 1

                            gt_angle = matched_gt[6]
                            gt_angle = np.arctan2(np.sin(gt_angle), np.cos(gt_angle))
                            pred_angle = np.arctan2(np.sin(det[6]), np.cos(det[6]))
                            angle_diff = np.abs(pred_angle - gt_angle)
                            if angle_diff > np.pi:
                                angle_diff = np.abs(np.pi * 2 - angle_diff)
                            angle_diffs.append(angle_diff)
                            boxes_diffs.append(np.abs(matched_gt[0:6] - det[0:6]))
                            overlaps.append(max_iou)
                    else:
                        if dontcare_flag == False:
                            fp += 1

                else:
                    if risk_eval == True:
                        dontcare_flag = True
                    if dontcare_flag == False:
                        fp += 1

                if dontcare_flag == False:
                    num_det_boxes += 1
                    prec = 1.0 * tp / np.clip((tp + fp), 1, None)
                    rec = 1.0 * tp / np.clip((tp + fn), 1, None)
                    precisions_per_cls.append(prec)
                    recalls_per_cls.append(rec)
                    score_per_cls.append(det[8])

        if len(precisions_per_cls) == 0:
            precisions_per_cls = [0.0]
            recalls_per_cls = [0.0]
            score_per_cls = [0.0]
        precisions_per_cls = np.array(precisions_per_cls)
        recalls_per_cls = np.array(recalls_per_cls)
        score_per_cls = np.array(score_per_cls)
        ap_recalls = np.linspace(0.0, 1.0, 101)
        ap_precisions, ap_scores_thresholds = self._calculate_ap(ap_recalls, recalls_per_cls, precisions_per_cls, score_per_cls)
        mean_iou, mean_angle_diff, flip_percentage, mean_boxes_diff = self._calculate_tp_attr(angle_diffs, boxes_diffs, overlaps)

        eval_results = {
            "ap_scalar": np.mean(ap_precisions),
            "ap_recalls": ap_recalls,
            "ap_precisions": ap_precisions,
            "ap_score_thresholds": ap_scores_thresholds,
            "detailed_ap_precisions": precisions_per_cls,
            "detailed_ap_recalls": recalls_per_cls,
            "mean_iou": mean_iou,
            "angle_diff": mean_angle_diff,
            "size_diff": mean_boxes_diff[3:6],
            "center_diff": mean_boxes_diff[0:3],
            "flip_percentage": flip_percentage,
            "det_count": num_det_boxes,
            "gt_count": num_gt_boxes,
            "detail_fp_count": fp,
            "detail_tp_count": tp,
            "detail_fn_count": fn,
            "detail_repeat_tp_count": repeat_tp,
            "detail_dontcare_count": dontcare,
        }

        if self.save_path is not None:
            self._save_pr_fig(recalls_per_cls, precisions_per_cls, score_per_cls, plt_suffix)
        return eval_results

    def _eval_for_confusion_matrix(self, all_cls_bboxes, difficulty=0, difficulty_index=8, part_num=50, iou_type="2D"):
        all_cls_dets = all_cls_bboxes["dets"]
        annotation_cls = all_cls_bboxes["gts"]
        class_num = len(self.class_names)

        all_cls_dets = np.concatenate(all_cls_dets, axis=0)
        all_cls_dets = self._filtered_pred_boxes(all_cls_dets, None, self.confusion_eval_score)
        dets_all_sorted = self._ordered_boxes(all_cls_dets, 8)

        for cls_name in self.confusion_eval_range:
            annotation_cls = self._filtered_gt_boxes_by_range_and_cls(annotation_cls, self.confusion_eval_range[cls_name], cls_name)
            dets_all_sorted = self._filtered_pred_boxes_by_range_and_cls(dets_all_sorted, self.confusion_eval_range[cls_name], cls_name)

        int_class_labels = [i + 1 for i in range(class_num)]
        confusion_matrix = {i: [0 for j in range(class_num + 2)] for i in int_class_labels}
        confusion_matrix["fp"] = [0 for i in range(class_num + 2)]
        confusion_matrix["precision"] = [0 for i in range(class_num + 2)]

        part_split_num = np.ceil(dets_all_sorted.shape[0] / part_num).astype(np.int32)
        for i in range(part_split_num):

            det_start = i * part_num
            det_end = (i + 1) * part_num
            if det_end > dets_all_sorted.shape[0]:
                det_end = dets_all_sorted.shape[0]
            frames_dets = dets_all_sorted[det_start:det_end]
            frames_idx = frames_dets[:, 9].astype(np.int32)
            frames_gts = self._get_part_gts(annotation_cls, frames_idx)
            max_iou_partlist, max_iou_idx_partlist = self._get_part_ious(frames_dets, frames_gts, iou_type=iou_type)
            for i_part_num in range(part_num):
                dontcare_flag = False
                if i_part_num >= frames_dets.shape[0]:
                    break
                det = frames_dets[i_part_num]
                det_pred_label = int(det[7])
                max_iou = max_iou_partlist[i_part_num]
                if max_iou < 0.01:
                    confusion_matrix["fp"][det_pred_label - 1] += 1
                    continue
                max_iou_idx = max_iou_idx_partlist[i_part_num]
                frame_matched_gt_idx = int(frames_gts[max_iou_idx][12])
                anno_idx = int(frames_dets[i_part_num, 9])
                matched_gt = annotation_cls[anno_idx][frame_matched_gt_idx]
                matched_gt_label = int(matched_gt[7])

                if matched_gt[difficulty_index] > difficulty or matched_gt_label == -1:
                    dontcare_flag = True

                if matched_gt_label == -1:
                    continue
                this_label_iou_thresh = self.eval_iou_cfg[matched_gt_label][0]
                if max_iou < this_label_iou_thresh:
                    if dontcare_flag == False:
                        confusion_matrix["fp"][det_pred_label - 1] += 1
                    continue

                if int(annotation_cls[anno_idx][frame_matched_gt_idx, 11]) == 0:
                    annotation_cls[anno_idx][frame_matched_gt_idx, 11] = det_pred_label

        annotation_cls = [gts[np.logical_and(gts[:, difficulty_index] <= difficulty, gts[:, 7] != -1), :] for gts in annotation_cls]
        ann_num = len(annotation_cls)
        for i in range(ann_num):
            object_num = annotation_cls[i].shape[0]
            for j in range(object_num):
                gt_label = int(annotation_cls[i][j][7])
                pred_label = int(annotation_cls[i][j][11])
                if pred_label == 0:
                    confusion_matrix[gt_label][-2] += 1
                else:
                    confusion_matrix[gt_label][pred_label - 1] += 1

        # calculate recall for each class
        for c_label in int_class_labels:
            class_gt_num = 0
            for i in range(class_num + 1):
                class_gt_num += confusion_matrix[c_label][i]
            confusion_matrix[c_label][-1] = confusion_matrix[c_label][c_label - 1] / (class_gt_num + 1e-6)

        # calculate precison for each class
        for i in range(class_num):
            class_pred_all = 0
            row_keys = int_class_labels + ["fp"]
            for r_k in row_keys:
                class_pred_all += confusion_matrix[r_k][i]
            confusion_matrix["precision"][i] = confusion_matrix[i + 1][i] / (class_pred_all + 1e-6)

        return confusion_matrix

    def _eval_for_general_obstacle_confusion_matrix(self, all_cls_bboxes, difficulty=0, difficulty_index=8, iou_type="2D"):
        all_cls_dets = all_cls_bboxes["dets"]
        annotation_cls = all_cls_bboxes["gts"]
        class_num = len(self.class_names)

        # annotation_cls = self._filtered_gt_boxes(annotation_cls, range_list)
        annotation_cls = self._filter_gt_boxes_by_axis(annotation_cls, range_list=self.obstacle_confusion_eval_range["x"], axis_type="x")
        annotation_cls = self._filter_gt_boxes_by_axis(annotation_cls, range_list=self.obstacle_confusion_eval_range["y"], axis_type="y")

        # delete all do not care class
        annotation_cls = [gts[gts[:, 7] != -1, :] for gts in annotation_cls]
        annotation_cls = [gts[gts[:, difficulty_index] <= difficulty, :] for gts in annotation_cls]

        confusion_matrix = {1: [0 for j in range(class_num + 2)]}

        frame_idx = 0
        for frame_dets, frame_gts in zip(all_cls_dets, annotation_cls):
            frames_dets_bev = np.concatenate([frame_dets[:, 0:2], frame_dets[:, 3:5], frame_dets[:, 6][..., np.newaxis]], axis=1)
            frames_gts_bev = np.concatenate([frame_gts[:, 0:2], frame_gts[:, 3:5], frame_gts[:, 6][..., np.newaxis]], axis=1)

            frame_det_sizes = frame_dets[:, 3] * frame_dets[:, 4]
            frame_gt_sizes = frame_gts[:, 3] * frame_gts[:, 4]

            ious = rotate_iou_gpu_eval(frames_dets_bev, frames_gts_bev).astype(np.float64)

            frame_det_gt_sizes = frame_det_sizes[:, None] + frame_gt_sizes[None, :]
            union = (ious * frame_det_gt_sizes) / (1 + ious)
            overlay_matrix = union / frame_gt_sizes[None, :]
            overlay_accumulate_gt = overlay_matrix.sum(axis=0)

            gt_num = overlay_accumulate_gt.shape[0]
            for gt_idx in range(gt_num):
                if overlay_accumulate_gt[gt_idx] > self.obstacle_overlay_thresh:
                    annotation_cls[frame_idx][gt_idx, 11] = 1
            frame_idx += 1

        ann_num = len(annotation_cls)
        for i in range(ann_num):
            object_num = annotation_cls[i].shape[0]
            for j in range(object_num):
                gt_label = int(annotation_cls[i][j][7])
                pred_label = int(annotation_cls[i][j][11])
                if pred_label == 0:
                    confusion_matrix[1][-2] += 1
                else:
                    confusion_matrix[1][gt_label - 1] += 1
        # calculate recall for each class
        gt_matched_num = 0
        for i in range(class_num):
            gt_matched_num += confusion_matrix[1][i]

        all_gt = gt_matched_num + confusion_matrix[1][-2]
        confusion_matrix[1][-1] = gt_matched_num / (all_gt + 1e-6)

        return confusion_matrix

    def update_dataframe(self, df, df_heads, metrics_list, head_name):
        if df is None:
            df_heads = [head_name] + df_heads
            res_df = pd.DataFrame(metrics_list, columns=df_heads)
            df = res_df
        else:
            metrics_list_new = [metrics_list[i][1:] for i in range(len(metrics_list))]
            res_df = pd.DataFrame(metrics_list_new, columns=df_heads)
            df = pd.concat([df, res_df], axis=1)
        return df

    def generate_confusion_vis_table(self, confusion_matrix_diff):
        class_to_int = {"unknown": -1}
        for class_idx, class_name in enumerate(self.class_names):
            class_to_int[class_name] = class_idx + 1

        heads = []
        metrics_list = []
        for diff in self.eval_difficulty_cfg:
            head_name = f"score_{self.confusion_eval_score:.2f} @ diff_{diff}"
            heads.append(head_name)
            confusion_matrix = confusion_matrix_diff[diff]
            for class_name in self.class_names:
                class_name_key = self.class_names.index(class_name) + 1
                class_range = [str(i) for i in self.confusion_eval_range[class_name_key]]
                class_range = "-".join(class_range)
                class_name = f"{class_name} @ range_{class_range}"
                heads.append(class_name)
            heads.append("fn")
            heads.append("recall")
            row_names = [i for i in self.class_names]
            row_names.append("fp")
            row_names.append("precision")
            for row_name in row_names:
                datas = []
                if row_name in self.class_names:
                    matrix_key = class_to_int[row_name]
                else:
                    matrix_key = row_name
                for p in confusion_matrix[matrix_key]:
                    if isinstance(p, float):
                        datas.append(f"{p:.4f}")
                    elif isinstance(p, int):
                        datas.append(f"{p:d}")
                    elif isinstance(p, str):
                        datas.append(p)
                    else:
                        datas.append(p)
                metrics_list.append([row_name, *datas])

        diff_num = len(self.eval_difficulty_cfg)
        metrics_rows = len(self.class_names) + 2
        rejust_metrics_list = [[] for i in range(metrics_rows)]
        for i in range(metrics_rows):
            all_diff_row_metrics = []
            for j in range(diff_num):
                all_diff_row_metrics = all_diff_row_metrics + metrics_list[j * metrics_rows + i]
            rejust_metrics_list[i] = all_diff_row_metrics

        df = pd.DataFrame(rejust_metrics_list, columns=heads)
        return df

    def generate_general_obstacle_confusion_table(self, confusion_matrix):
        heads = []
        heads.append(f"Iou_Thresh @ {self.obstacle_overlay_thresh:.2f}")
        for c in self.class_names:
            heads.append(c)
        heads.append("fn")
        heads.append("recall")
        metrics_list = []
        datas = []
        for p in confusion_matrix[1]:
            if isinstance(p, float):
                datas.append(f"{p:.4f}")
            elif isinstance(p, int):
                datas.append(f"{p:d}")
            elif isinstance(p, str):
                datas.append(p)
            else:
                datas.append(p)
        metrics_list.append(["obstacle", *datas])
        df = pd.DataFrame(metrics_list, columns=heads)
        return df

    def generate_difficulty_vis_table(self, summaries, class_name, iou_list, iou_type, dataframe=None, all_dataframe=None):
        heads = [f"Difficulty @ {class_name} @ {iou_type}"]
        metrics = OrderedDict()
        res_dict = {}
        for diff in self.eval_difficulty_cfg:
            for iou_thres in iou_list:
                heads.append(f"difficulty_{diff} iou@{iou_thres}")
                results = summaries[diff][iou_thres]

                name_str = f"{class_name}_diff@{diff}_iou-{iou_type}@{iou_thres}"
                res_dict[name_str] = results["ap"]

                for k, v in results.items():
                    if "_forTF" in k:
                        continue
                    elif k not in metrics.keys():
                        metrics[k] = []
                    metrics[k].append(v)

        metrics_list = []
        for k, v in metrics.items():
            if "percentage" in k:
                v_percentage = [f"{p*100:.2f}%" for p in v]
                metrics_list.append([k, *v_percentage])
            elif "angle_diff" == k:
                datas = [f"{p:.2f}" for p in v]
                metrics_list.append([k, *datas])
            else:
                datas = []
                for p in v:
                    if isinstance(p, float):
                        datas.append(f"{p:.4f}")
                    elif isinstance(p, int):
                        datas.append(f"{p:d}")
                    elif isinstance(p, str):
                        datas.append(p)
                    else:
                        datas.append(p)
                metrics_list.append([k, *datas])

        tables = tabulate(metrics_list, heads, tablefmt="grid", stralign="right", numalign="right")
        df_heads = [f"{class_name[:3]}. {heads[i].replace('difficulty', 'diff')}" for i in range(1, len(heads))]

        dataframe = self.update_dataframe(dataframe, df_heads, metrics_list, f"Difficulty @ {iou_type}")
        all_dataframe = self.update_dataframe(all_dataframe, df_heads, metrics_list, f"Eval @ {iou_type}")
        return tables, res_dict, dataframe, all_dataframe

    def generate_range_vis_table(
        self, summaries, class_name="Vehicle", ranges_list=[[0, 50], [50, 100], [100, 150]], iou_list=[0.7], iou_type="2D", dataframe=None, all_dataframe=None
    ):
        heads = [f"Range @ {class_name} @ {iou_type}"]
        metrics = OrderedDict()
        res_dict = {}

        for range_list in ranges_list:
            for iou_thres in iou_list:
                heads.append(f"{range_list[0]}m - {range_list[1]}m iou@{iou_thres}")
                range_str = str(range_list[0]) + "-" + str(range_list[1])
                results = summaries[range_str][iou_thres]

                name_str = f"{class_name}_range{range_str}_iou-{iou_type}@{iou_thres}"
                res_dict[name_str] = results["ap"]

                for k, v in results.items():
                    if "_forTF" in k:
                        continue
                    elif k not in metrics.keys():
                        metrics[k] = []
                    metrics[k].append(v)

        metrics_list = []
        for k, v in metrics.items():
            if "percentage" in k:
                v_percentage = [f"{p*100:.2f}%" for p in v]
                metrics_list.append([k, *v_percentage])
            elif "angle_diff" == k:
                datas = [f"{p:.2f}" for p in v]
                metrics_list.append([k, *datas])
            else:
                datas = []
                for p in v:
                    if isinstance(p, float):
                        datas.append(f"{p:.4f}")
                    elif isinstance(p, int):
                        datas.append(f"{p:d}")
                    elif isinstance(p, str):
                        datas.append(p)
                    else:
                        datas.append(p)
                metrics_list.append([k, *datas])

        tables = tabulate(metrics_list, heads, tablefmt="grid", stralign="right", numalign="right")

        df_heads = [f"{class_name[:3]}. {heads[i]}" for i in range(1, len(heads))]
        dataframe = self.update_dataframe(dataframe, df_heads, metrics_list, f"Range @ {iou_type}")
        all_dataframe = self.update_dataframe(all_dataframe, df_heads, metrics_list, f"Eval @ {iou_type}")
        return tables, res_dict, dataframe, all_dataframe

    def generate_angle_vis_table(
        self, summaries, class_name="Vehicle", angles_list=[[-60, -30], [-30, 30], [30, 60]], iou_list=[0.7], iou_type="2D", dataframe=None, all_dataframe=None
    ):
        heads = [f"Angle @ {class_name} @ {iou_type}"]
        metrics = OrderedDict()
        res_dict = {}

        for angle_list in angles_list:
            for iou_thres in iou_list:
                heads.append(f"{angle_list[0]}° ~ {angle_list[1]}° iou@{iou_thres}")
                angle_str = str(angle_list[0]) + "-" + str(angle_list[1])
                results = summaries[angle_str][iou_thres]

                name_str = f"{class_name}_angle{angle_str}_iou-{iou_type}@{iou_thres}"
                res_dict[name_str] = results["ap"]

                for k, v in results.items():
                    if "_forTF" in k:
                        continue
                    elif k not in metrics.keys():
                        metrics[k] = []
                    metrics[k].append(v)

        metrics_list = []
        for k, v in metrics.items():
            if "percentage" in k:
                v_percentage = [f"{p*100:.2f}%" for p in v]
                metrics_list.append([k, *v_percentage])
            elif "angle_diff" == k:
                datas = [f"{p:.2f}" for p in v]
                metrics_list.append([k, *datas])
            else:
                datas = []
                for p in v:
                    if isinstance(p, float):
                        datas.append(f"{p:.4f}")
                    elif isinstance(p, int):
                        datas.append(f"{p:d}")
                    elif isinstance(p, str):
                        datas.append(p)
                    else:
                        datas.append(p)
                metrics_list.append([k, *datas])

        tables = tabulate(metrics_list, heads, tablefmt="grid", stralign="right", numalign="right")

        df_heads = [f"{class_name[:3]}. {heads[i]}" for i in range(1, len(heads))]
        dataframe = self.update_dataframe(dataframe, df_heads, metrics_list, f"Angle @ {iou_type}")
        all_dataframe = self.update_dataframe(all_dataframe, df_heads, metrics_list, f"Eval @ {iou_type}")
        return tables, res_dict, dataframe, all_dataframe

    def generate_risk_vis_table(self, summaries, iou_type, all_dataframe=None):
        heads = [f"Risk Object @ {iou_type}"]
        metrics = OrderedDict()
        res_dict = {}

        for category, iou_list in self.eval_iou_cfg.items():
            if self.class_names[category - 1] in ["Vehicle", "Bigcar"]:
                diff_list = ["CIPV", "ALAV"]
            elif self.class_names[category - 1] in ["Pedestrian", "Cyclist"]:
                diff_list = ["RISK"]
            else:
                NotImplementedError
            for diff, diff_type in enumerate(diff_list):

                for iou_thres in iou_list:
                    heads.append(f"{self.class_names[category-1]} {diff_type} iou@{iou_thres}")
                    results = summaries[category][diff_type][iou_thres]

                    name_str = f"{self.class_names[category-1]}_{diff_list}_iou-{iou_type}@{iou_thres}"
                    res_dict[name_str] = results["ap"]

                    for k, v in results.items():
                        if "_forTF" in k:
                            continue
                        elif k not in metrics.keys():
                            metrics[k] = []
                        metrics[k].append(v)

        metrics_list = []
        for k, v in metrics.items():
            if "percentage" in k:
                v_percentage = [f"{p*100:.2f}%" for p in v]
                metrics_list.append([k, *v_percentage])
            elif "angle_diff" == k:
                datas = [f"{p:.2f}" for p in v]
                metrics_list.append([k, *datas])
            else:
                datas = []
                for p in v:
                    if isinstance(p, float):
                        datas.append(f"{p:.4f}")
                    elif isinstance(p, int):
                        datas.append(f"{p:d}")
                    elif isinstance(p, str):
                        datas.append(p)
                    else:
                        datas.append(p)
                metrics_list.append([k, *datas])

        tables = tabulate(metrics_list, heads, tablefmt="grid", stralign="right", numalign="right")
        df_heads = heads[1:]
        dataframe = self.update_dataframe(None, df_heads, metrics_list, f"Risk Object @ {iou_type}")
        all_dataframe = self.update_dataframe(all_dataframe, df_heads, metrics_list, f"Eval @ {iou_type}")
        self.export_excel(heads[0], dataframe)
        return tables, res_dict, all_dataframe

    def get_summary(self, eval_results):
        summary = OrderedDict()
        summary.update({"ap": eval_results["ap_scalar"]})
        for target_p in self.target_precisions:
            recalls = eval_results["ap_recalls"]
            precisions = eval_results["ap_precisions"]
            score_thresholds = eval_results["ap_score_thresholds"]
            index = np.argmin(np.abs(precisions - target_p))
            s = score_thresholds[index]
            r = recalls[index]
            summary.update({f"prec@{target_p:.2f} r@s": f"{r:.3f}@{s:.2f}"})
            summary.update({f"prec@{target_p:.2f}_forTF": r})

        summary.update({"tp_mean_iou": eval_results["mean_iou"]})
        summary.update({"yaw_symm_err(deg)": eval_results["angle_diff"]})
        summary.update({"yaw_flip_percentage": eval_results["flip_percentage"]})
        if eval_results["detail_tp_count"] > 0:
            summary.update({"center_err(x/y/z)": f"{eval_results['center_diff'][0]}/{eval_results['center_diff'][1]}/{eval_results['center_diff'][2]}"})
            summary.update({"size_err(l/w/h)": f"{eval_results['size_diff'][0]}/{eval_results['size_diff'][1]}/{eval_results['size_diff'][2]}"})
        else:
            summary.update({"center_err(x/y/z)": f"nan"})
            summary.update({"size_err(l/w/h)": f"nan"})
        summary.update(
            {"det/gt/retp/dc": f"{eval_results['det_count']}/{eval_results['gt_count']}/{eval_results['detail_repeat_tp_count']}/{eval_results['detail_dontcare_count']}"}
        )
        summary.update({"fn/fp/tp": f"{eval_results['detail_fn_count']}/{eval_results['detail_fp_count']}/{eval_results['detail_tp_count']}"})

        return summary

    def preprocess_all_infos(self, all_infos, all_gt_infos):
        if isinstance(all_infos, str):
            with open(all_infos, "rb") as f:
                all_result = pickle.load(f)
        elif isinstance(all_infos, list):
            all_result = all_infos
        else:
            raise TypeError(f"Evaluation data type:{type(all_infos)} not in [str, list]!")

        if isinstance(all_gt_infos, list):
            all_gts = all_gt_infos
        else:
            raise TypeError(f"Evaluation gt data type:{type(all_gt_infos)} not in [str, list]!")

        dets_gts_by_cls, dets_gts = self._split_bbox_by_cls(all_result, all_gts)
        return dets_gts_by_cls, dets_gts

    def eval_by_difficulty(self, iou_type="2D"):
        all_summary_diff = OrderedDict()
        for category, iou_list in tqdm(self.eval_iou_cfg.items()):
            all_bboxes = self.bboxes_by_cls[category]
            all_summary_diff[category] = OrderedDict()
            for diff in self.eval_difficulty_cfg:
                all_summary_diff[category][diff] = OrderedDict()
                for iou_thre in iou_list:
                    all_summary_diff[category][diff][iou_thre] = OrderedDict({})
                    plt_suffix_str = f"{self.class_names[category-1]}_{iou_type}_diff@{diff}_iou@{iou_thre}"
                    eval_results = self._eval_for_cls_at_iou_part(all_bboxes, iou_thre, plt_suffix=plt_suffix_str, difficulty=diff, part_num=50, iou_type=iou_type)
                    all_summary_diff[category][diff][iou_thre].update(self.get_summary(eval_results))
        return all_summary_diff

    def get_confusion_eval_score(self, all_summary_diff):
        iou = self.eval_iou_cfg[1][0]
        s1 = float(all_summary_diff[1][0][iou]["prec@0.90 r@s"].split("@")[1])
        s2 = float(all_summary_diff[1][0][iou]["prec@0.80 r@s"].split("@")[1])
        self.confusion_eval_score = (s1 + s2) / 2

    def confusion_matrix_for_general_obstacle(self, iou_type="2D"):
        all_cls_bboxes = copy.deepcopy(self.dets_gts)
        confusion_matrix = self._eval_for_general_obstacle_confusion_matrix(all_cls_bboxes, difficulty=0, iou_type=iou_type)
        dataframe = self.generate_general_obstacle_confusion_table(confusion_matrix)
        self.export_excel("Confusion Matrix @ Obstacle", dataframe)

    def confusion_matrix_eval_by_difficulty(self, iou_type="2D"):
        all_confusion_matrix_diff = OrderedDict()
        for diff in self.eval_difficulty_cfg:
            all_confusion_matrix_diff[diff] = OrderedDict()
            all_cls_bboxes = copy.deepcopy(self.dets_gts)
            confusion_matrix = self._eval_for_confusion_matrix(all_cls_bboxes, difficulty=diff, iou_type=iou_type)
            all_confusion_matrix_diff[diff].update(confusion_matrix)
        return all_confusion_matrix_diff

    def eval_by_range(self, difficulty=1, iou_type="2D"):
        all_summary_range = OrderedDict()
        for category, iou_list in tqdm(self.eval_iou_cfg.items()):
            all_bboxes = self.bboxes_by_cls[category]
            all_summary_range[category] = OrderedDict()
            for range_list in self.eval_range_cfg[category]:
                range_str = str(range_list[0]) + "-" + str(range_list[1])
                all_summary_range[category][range_str] = OrderedDict()
                for iou_thre in iou_list:
                    all_summary_range[category][range_str][iou_thre] = OrderedDict({})
                    plt_suffix_str = f"{self.class_names[category-1]}_{iou_type}_diff@{difficulty}_range@{range_str}_iou@{iou_thre}"
                    eval_results = self._eval_for_cls_at_iou_part(
                        all_bboxes, iou_thre, plt_suffix=plt_suffix_str, difficulty=difficulty, range_list=range_list, part_num=50, iou_type=iou_type
                    )
                    all_summary_range[category][range_str][iou_thre].update(self.get_summary(eval_results))
        return all_summary_range

    def eval_by_angle(self, difficulty=1, iou_type="2D"):
        all_summary_angle = OrderedDict()
        for category, iou_list in tqdm(self.eval_iou_cfg.items()):
            all_bboxes = self.bboxes_by_cls[category]
            all_summary_angle[category] = OrderedDict()
            for angle_list in self.eval_angle_cfg[category]:
                angle_str = str(angle_list[0]) + "-" + str(angle_list[1])
                all_summary_angle[category][angle_str] = OrderedDict()
                for iou_thre in iou_list:
                    all_summary_angle[category][angle_str][iou_thre] = OrderedDict({})
                    plt_suffix_str = f"{self.class_names[category-1]}_{iou_type}_diff@{difficulty}_angle@{angle_str}_iou@{iou_thre}"
                    eval_results = self._eval_for_cls_at_iou_part(
                        all_bboxes, iou_thre, plt_suffix=plt_suffix_str, difficulty=difficulty, angle_list=angle_list, part_num=50, iou_type=iou_type
                    )
                    all_summary_angle[category][angle_str][iou_thre].update(self.get_summary(eval_results))
        return all_summary_angle

    def eval_by_risk(self, iou_type="2D"):
        all_summary_risk = OrderedDict()
        for category, iou_list in tqdm(self.eval_iou_cfg.items()):
            all_bboxes = self.bboxes_by_cls[category]
            all_summary_risk[category] = OrderedDict()

            if self.class_names[category - 1] in ["Vehicle", "Bigcar"]:
                difficulty_index = 9
                diff_list = ["CIPV", "ALAV"]
            elif self.class_names[category - 1] in ["Pedestrian", "Cyclist"]:
                difficulty_index = 10
                diff_list = ["RISK"]
            else:
                NotImplementedError

            for diff, diff_type in enumerate(diff_list):
                if diff_type not in all_summary_risk[category]:
                    all_summary_risk[category][diff_type] = OrderedDict()

                for iou_thre in iou_list:
                    all_summary_risk[category][diff_type][iou_thre] = OrderedDict({})
                    plt_suffix_str = f"{self.class_names[category-1]}_{iou_type}_{diff_type}_iou@{iou_thre}"
                    eval_results = self._eval_for_cls_at_iou_part(
                        all_bboxes, iou_thre, plt_suffix=plt_suffix_str, difficulty=diff, difficulty_index=difficulty_index, part_num=50, iou_type=iou_type, risk_eval=True
                    )
                    all_summary_risk[category][diff_type][iou_thre].update(self.get_summary(eval_results))
        return all_summary_risk

    def export_excel(self, sheet_name, table, mode="a", col_width=25):
        if self.save_path is None:
            return
        self.excel_save_path = self.save_path / self.excel_name

        if self.excel_save_path.exists() is False:
            mode = "w"
        with pd.ExcelWriter(self.save_path / self.excel_name, mode=mode, engine="openpyxl") as writer:
            table.to_excel(writer, sheet_name=sheet_name)
            writer.sheets[sheet_name].column_dimensions["A"].width = 5
            for col_i in ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD"]:
                writer.sheets[sheet_name].column_dimensions[col_i].width = col_width

    def my_general_obstacle_evaluation(self, prediction_infos, gt_infos, iou_type="2D"):
        assert iou_type in ["2D", "3D"], "Only support iou type in [2D, 3D]!"
        self.bboxes_by_cls, self.dets_gts = self.preprocess_all_infos(prediction_infos, gt_infos)
        self.confusion_matrix_for_general_obstacle()

    def my_evaluation(self, prediction_infos, gt_infos, iou_type="2D"):
        assert iou_type in ["2D", "3D"], "Only support iou type in [2D, 3D]!"
        all_summary = OrderedDict()
        summary_string = self.output_log("My eval - Starting preprocessing infos.", "")
        self.bboxes_by_cls, self.dets_gts = self.preprocess_all_infos(prediction_infos, gt_infos)

        summary_string = self.output_log("My eval - Starting eval by difficulty.", summary_string)
        all_summary_diff = self.eval_by_difficulty(iou_type=iou_type)

        summary_string = self.output_log("My eval - Starting eval by range.", summary_string)
        range_difficulty = 1
        all_summary_range = self.eval_by_range(difficulty=range_difficulty, iou_type=iou_type)

        summary_string = self.output_log("My eval - Starting eval by angle.", summary_string)
        angle_difficulty = 1
        all_summary_angle = self.eval_by_angle(difficulty=angle_difficulty, iou_type=iou_type)

        summary_string = self.output_log("My eval - Starting eval by risk.", summary_string)
        all_summary_risk = self.eval_by_risk(iou_type=iou_type)

        summary_string = self.output_log("My eval - Starting eval confusion matrix by difficulty.", summary_string)
        self.get_confusion_eval_score(all_summary_diff)
        all_confusion_matrix_diff = self.confusion_matrix_eval_by_difficulty()

        res_dict_all = {}
        summary_string = self.output_log("Results by different difficulty level", summary_string, padding="=")
        all_dataframe = None
        diff_dataframe = None
        for category, iou_list in self.eval_iou_cfg.items():
            tables_vis, res_diff_dict, diff_dataframe, all_dataframe = self.generate_difficulty_vis_table(
                all_summary_diff[category], class_name=self.class_names[category - 1], iou_list=iou_list, iou_type=iou_type, dataframe=diff_dataframe, all_dataframe=all_dataframe
            )
            summary_string = self.output_log(f"\n[Eval by Difficulty] - {self.class_names[category-1]} @ Difficulty\n" + tables_vis, summary_string)
            res_dict_all.update(res_diff_dict)
        self.export_excel(f"Difficulty @ {iou_type}", diff_dataframe)

        summary_string = self.output_log("Results by different difficulty ranges", summary_string, padding="=")
        range_dataframe = None
        for category, iou_list in self.eval_iou_cfg.items():
            tables_vis, res_range_dict, range_dataframe, all_dataframe = self.generate_range_vis_table(
                all_summary_range[category],
                class_name=self.class_names[category - 1],
                ranges_list=self.eval_range_cfg[category],
                iou_list=iou_list,
                iou_type=iou_type,
                dataframe=range_dataframe,
                all_dataframe=all_dataframe,
            )
            summary_string = self.output_log(f"\n[Eval by Range] - {self.class_names[category-1]} @ Difficulty {range_difficulty}\n" + tables_vis, summary_string)
            res_dict_all.update(res_range_dict)
        self.export_excel(f"Range @ {iou_type}", range_dataframe)

        summary_string = self.output_log("Results by different difficulty angles", summary_string, padding="=")
        angle_dataframe = None
        for category, iou_list in self.eval_iou_cfg.items():
            tables_vis, res_angle_dict, angle_dataframe, all_dataframe = self.generate_angle_vis_table(
                all_summary_angle[category],
                class_name=self.class_names[category - 1],
                angles_list=self.eval_angle_cfg[category],
                iou_list=iou_list,
                iou_type=iou_type,
                dataframe=angle_dataframe,
                all_dataframe=all_dataframe,
            )
            summary_string = self.output_log(f"\n[Eval by Angles] - {self.class_names[category-1]} @ Difficulty {angle_difficulty}\n" + tables_vis, summary_string)
            res_dict_all.update(res_angle_dict)
        self.export_excel(f"Angles @ {iou_type}", angle_dataframe)



        summary_string = self.output_log("Results by risk objects", summary_string, padding="=")
        tables_vis, res_risk_dict, all_dataframe = self.generate_risk_vis_table(all_summary_risk, iou_type=iou_type, all_dataframe=all_dataframe)
        res_dict_all.update(res_risk_dict)
        summary_string = self.output_log(f"\n[Eval by Risk] -\n" + tables_vis, summary_string)
        self.export_excel(f"Eval @ {iou_type}", all_dataframe)

        confusion_dataframe = self.generate_confusion_vis_table(all_confusion_matrix_diff)
        self.export_excel(f"Confusion Matrix @ {iou_type}", confusion_dataframe)

        all_summary["difficulty"] = all_summary_diff
        all_summary["range"] = all_summary_range
        all_summary["angle"] = all_summary_angle
        all_summary["risk"] = all_summary_risk

        if self.save_path is not None:
            with open(self.save_path / "eval_res_dict.pkl", "wb") as f:
                pickle.dump(all_summary, f)

        return res_dict_all, summary_string


def get_dummy_info(cur_info):
    dummy_gt = [-100, -100, -100, 0.01, 0.01, 0.01, 0]
    cur_info["gt_boxes_lidar"] = np.array([dummy_gt])
    cur_info["mini_frame"] = np.array([dummy_gt])
    cur_info["location"] = cur_info["gt_boxes_lidar"][:, 0:3]
    cur_info["dimensions"] = cur_info["gt_boxes_lidar"][:, 0:3]

    cur_info["name"] = np.array(["unknown"])
    cur_info["difficulty"] = np.array([4])
    cur_info["num_points_in_gt"] = np.array([0])
    cur_info["label_id"] = np.array([-1])
    for k in ["vehicle_relation", "vehicle_status", "object_risk", "ped_status", "animal_status", "riding_status", "object_point_type", "vehicle_relation_add"]:
        if k in cur_info:
            cur_info[k] = np.array([None])
    return cur_info


def change_gt_name_remap_for_general_obstacle(gt_name_map):
    gt_name_map["CAR"] = "unknown"
    gt_name_map["VAN"] = "unknown"
    gt_name_map["MINI_TRUCK"] = "unknown"
    gt_name_map["TINY_CAR"] = "unknown"
    gt_name_map["BUS"] = "unknown"
    gt_name_map["TRUCK"] = "unknown"
    gt_name_map["FLATBED_TRUCK"] = "unknown"
    gt_name_map["DUMP_TRUCK"] = "unknown"
    gt_name_map["SEMI_TRUCK"] = "unknown"
    gt_name_map["CONSTRUCTION_TRUCK"] = "unknown"
    gt_name_map["SPECIAL_VEHICLE"] = "unknown"
    gt_name_map["BICYCLE"] = "unknown"
    gt_name_map["MOTORCYCLE"] = "unknown"
    gt_name_map["TRICYCLE"] = "unknown"
    gt_name_map["PED_ADULT"] = "unknown"
    gt_name_map["PED_CHILD"] = "unknown"
    gt_name_map["BABY_CARRIAGE"] = "unknown"
    gt_name_map["ANIMAL"] = "unknown"
    gt_name_map["CONE"] = "Cone"
    gt_name_map["POLE"] = "unknown"
    gt_name_map["BARRIER"] = "Barrier"
    return gt_name_map


def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--pred_infos", type=str, default=None, help="pickle file")
    parser.add_argument("--gt_infos", type=str, default=None, help="pickle file")
    parser.add_argument("--class_names", type=str, nargs="+", default=["Vehicle", "Pedestrian", "Cyclist", "Bigcar"], help="")
    parser.add_argument("--dataset_cfg_file", type=str, default="tools/cfgs/dataset_configs/my_dataset.yaml", help="config yaml file")
    parser.add_argument("--eval_type", type=str, default="my_data_eval_metric", help="eval type - [my_data_eval_metric, waymo]")
    parser.add_argument("--iou_type", type=str, default="2D", help="eval type - [2D, 3D]")
    parser.add_argument("--sampled_interval", type=int, default=1, help="sampled interval for GT sequences")
    parser.add_argument("--general_obstacle_detection", action="store_true", help="eval general obstacle detection")
    parser.add_argument("--obstacle_overlay_thresh", type=float, default=0.1, help="obstacle iou threshold")
    args = parser.parse_args()

    assert args.eval_type in ["my_data_eval_metric", "waymo"], "eval_type only support ['my_data_eval_metric', 'waymo']"
    assert args.iou_type in ["2D", "3D"], "iou_type only support ['2D', '3D']"

    import yaml

    with open(args.dataset_cfg_file, "r") as f:
        try:
            dataset_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            dataset_config = yaml.safe_load(f)

    gt_name_remap = None
    if "GT_NAME_REMAP" in dataset_config:
        gt_name_remap = dataset_config["GT_NAME_REMAP"]

    if args.general_obstacle_detection:
        gt_name_remap = change_gt_name_remap_for_general_obstacle(gt_name_remap)
        args.class_names = ["Cone", "Barrier"]

    if "GT_CLASS_SPLIT" in dataset_config:
        gt_class_split = dataset_config["GT_CLASS_SPLIT"]
    else:
        gt_class_split = None

    # For the fp of the 'unknown' object area, the fp count will not be included
    class_map = {"unknown": -1}
    for class_idx, class_name in enumerate(args.class_names):
        class_map[class_name] = class_idx + 1
    pred_infos = pickle.load(open(args.pred_infos, "rb"))
    gt_infos = pickle.load(open(args.gt_infos, "rb"))

    if args.eval_type == "my_data_eval_metric":
        save_dir = Path(args.pred_infos).parent
        eval = MyDetMetric(
            save_path=str(save_dir), class_names=args.class_names, point_cloud_range=dataset_config["POINT_CLOUD_RANGE"], obstacle_overlay_thresh=args.obstacle_overlay_thresh
        )
        eval.output_log("Start to evaluate the my_data_eval_metric format results.")
    else:
        print(f"[TIME] {time.strftime('%Y-%D %H:%M:%S',time.localtime())} :Error! eval_type only support ['my_data_eval_metric']")

    gt_infos_dst = []
    for idx in range(0, len(gt_infos), args.sampled_interval):
        cur_info = gt_infos[idx]["annos"]
        cur_info["frame_id"] = gt_infos[idx]["common_info"]["frame_id"]
        num_obj = cur_info["gt_boxes_lidar"].shape[0]
        if num_obj <= 0:
            cur_info = get_dummy_info(cur_info)
        else:
            cur_info["name"] = np.array([gt_name_remap[name] for name in cur_info["name"]])
            cur_info = common_utils.drop_info_with_name(cur_info, name="others")
            if gt_class_split is not None:
                if num_obj > 0:
                    for k, v in gt_class_split.items():
                        class_mask = cur_info["name"] == k
                        remap_mask = np.logical_or(cur_info["gt_boxes_lidar"][:, 3] > v["l"], cur_info["gt_boxes_lidar"][:, 4] > v["w"])
                        remap_mask = np.logical_or(remap_mask, cur_info["gt_boxes_lidar"][:, 5] > v["h"])
                        remap_mask = np.logical_and(remap_mask, class_mask)
                        cur_info["name"][remap_mask] = v["remap"]

            cur_info["label_id"] = np.array([class_map[name] for name in cur_info["name"]])

        gt_infos_dst.append(cur_info)

    if args.eval_type == "my_data_eval_metric":
        if args.general_obstacle_detection:
            eval.my_general_obstacle_evaluation(pred_infos, gt_infos_dst, iou_type=args.iou_type)
        else:
            my_AP, ap_result_str = eval.my_evaluation(pred_infos, gt_infos_dst, iou_type=args.iou_type)
    elif args.eval_type == "waymo":
        my_AP, my_AP_str = eval.waymo_evaluation(pred_infos, gt_infos_dst, class_name=args.class_names, distance_thresh=1000, fake_gt_infos=False, iou_type=args.iou_type)

        ap_result_str = "\n"
        for key in my_AP:
            my_AP[key] = my_AP[key][0]
            ap_result_str += "%s %s: %.4f \n" % (args.iou_type, key, my_AP[key])

        ap_result_str += "\n"
        ap_result_str += my_AP_str

        print(ap_result_str)


if __name__ == "__main__":
    main()