import copy
import pickle
import json
import numpy as np
from skimage import io

from . import my_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..augmentor.data_augmentor import DataAugmentor
from ..dataset import DatasetTemplate
from .dataset_tools import load_pcd

from .my_utils import SceneElements


class MyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):

        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        if self.dataset_cfg.get("GT_NAME_REMAP", None) is not None:
            self.gt_name_remap = self.dataset_cfg["GT_NAME_REMAP"]
        else:
            self.gt_name_remap = None

        if self.dataset_cfg.get("GT_CLASS_SPLIT", None) is not None:
            self.gt_class_split = self.dataset_cfg["GT_CLASS_SPLIT"]
        else:
            self.gt_class_split = None

        if self.dataset_cfg.get("POINT_FEATURE_ENCODING", None) is not None:
            self.point_num_features = len(self.dataset_cfg["POINT_FEATURE_ENCODING"]["src_feature_list"])
        else:
            self.point_num_features = 4

        if self.dataset_cfg.get("SCENE_RULES", None) is not None:
            self.scene_elementer = SceneElements(self.dataset_cfg["SCENE_RULES"])
        else:
            self.scene_elementer = None
        self.data_type = 'my_data'
        self.train_for_debug_mode = self.dataset_cfg.get('TRAIN_FOR_DEBUG', False)

        if self.dataset_cfg.get('DATA_AUGMENTOR', None) is not None:
            self.data_augmentor = (
                DataAugmentor(self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger,
                              gt_name_remap=self.gt_name_remap, data_type=self.data_type) if self.training else None
            )
        else:
            self.data_augmentor = None

        self.data_composition = self.dataset_cfg.get('DATA_COMPOSITION', ['pcd'])
        self.my_infos = []
        self.include_my_data(self.mode)

    def include_my_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        my_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                my_infos.extend(infos)

        self.my_infos.extend(my_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(my_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        print("split_dir", split_dir)
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        print("sample_id_list:", self.sample_id_list)

    def get_lidar(self, idx, num_features=4):
        # lidar_path = self.root_split_path / 'data' / ('%s.pcd' % idx)
        # points = load_pcd.get_points_from_pcd_file(lidar_path, num_features=num_features)
        lidar_path = self.root_split_path / 'data' / ('%s.npy' % idx)
        points = np.load(lidar_path)
        return points

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        raise NotImplementedError("image data not supported")

    def get_label(self, idx):
        label_path = self.root_split_path / 'label' / ('%s.json' % idx)
        assert label_path.exists()
        with open(label_path, "r") as f:
            label_file = json.load(f)
        return label_file

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        raise NotImplementedError("depth_map not supported")

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': self.point_num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj["type"].replace("TYPE_", "") for obj in obj_list])
                annotations["location"] = np.array(
                    [[obj["position3d"]["x"], obj["position3d"]["y"], obj["position3d"]["z"]] for obj in obj_list])
                annotations["dimensions"] = np.array(
                    [[obj["size3d"]["x"], obj["size3d"]["y"], obj["size3d"]["z"]] for obj in obj_list])
                annotations["rotation_y"] = np.array([float(obj_list["heading"]) for obj_list in obj_list])
                annotations["difficulty"] = np.array(
                    [3 - int(obj["tag"]["confidence"]) if "confidence" in obj["tag"] else -1 for obj in obj_list])
                annotations["gt_boxes_lidar"] = np.hstack(
                    (annotations["location"], annotations["dimensions"], annotations["rotation_y"].reshape(-1, 1)))

                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            print(sample_id_list)
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('my_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx, num_features=self.point_num_features)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            # bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                gt_points = gt_points.astype(np.float32)
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {
                        'name': names[i],
                        'path': db_path,
                        'sample_idx': sample_idx,
                        'gt_idx': i,
                        'box3d_lidar': gt_boxes[i],
                        'num_points_in_gt': gt_points.shape[0],
                        'difficulty': difficulty[i],
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]

            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.my_infos[0].keys():
            return None, {}
        if kwargs["eval_iou_type"] is not None:
            eval_iou_type = kwargs["eval_iou_type"]
        else:
            eval_iou_type = ["2D"]
        assert isinstance(eval_iou_type, list), "eval_iou_type only support instance of list."

        def get_dummy_info(cur_info):
            dummy_gt = [-100, -100, -100, 0.01, 0.01, 0.01, 0]
            cur_info["gt_boxes_lidar"] = np.array([dummy_gt])
            cur_info["location"] = cur_info["gt_boxes_lidar"][:, 0:3]
            cur_info["dimensions"] = cur_info["gt_boxes_lidar"][:, 0:3]

            cur_info["name"] = np.array(["unknown"])
            cur_info["difficulty"] = np.array([4])
            cur_info["num_points_in_gt"] = np.array([0])
            cur_info["label_id"] = np.array([-1])
            for k in ["vehicle_relation", "vehicle_status", "object_risk", "ped_status", "animal_status",
                      "riding_status", "object_point_type", "vehicle_relation_add"]:
                if k in cur_info:
                    cur_info[k] = np.array([None])
            return cur_info

        def my_eval(eval_det_annos, eval_gt_annos, eval_iou_type, logger=None):
            from .my_eval.my_eval import MyDetMetric

            figs_save_path = kwargs["output_path"]
            figs_save_path.mkdir(parents=True, exist_ok=True)
            eval = MyDetMetric(save_path=str(figs_save_path.parent), class_names=self.class_names,
                                 point_cloud_range=self.point_cloud_range, logger=logger)
            ap_result_str_all = ""
            ap_dict_all = {}
            for iou_type in eval_iou_type:
                ap_dict, ap_result_str = eval.my_evaluation(eval_det_annos, eval_gt_annos, iou_type=iou_type)
                ap_result_str_all += ap_result_str
                ap_dict_all.update(ap_dict)
            return ap_result_str_all, ap_dict_all

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info["annos"]) for info in self.my_infos]

        def update_result_withgt(eval_det_annos, eval_gt_annos, res_save_path):
            res_save_path.mkdir(parents=True, exist_ok=True)
            assert len(eval_det_annos) == len(
                eval_gt_annos), "Lists eval_det_annos and eval_gt_annos have inconsistent lengths!"
            for idx in range(len(eval_det_annos)):
                eval_det_annos[idx]["annos"] = eval_gt_annos[idx]
                eval_det_annos[idx]["infos"] = {
                    "point_cloud": self.my_infos[idx]["point_cloud"],
                    "image": self.my_infos[idx]["image"],
                    "common_info": self.my_infos[idx]["common_info"],
                    "metadata": self.my_infos[idx]["metadata"],
                }
            with open(res_save_path / "result_withgt.pkl", "wb") as f:
                pickle.dump(eval_det_annos, f)

        if self.gt_name_remap is not None:
            for idx, annos in enumerate(eval_gt_annos):
                annos["name"] = np.array([self.gt_name_remap[name] for name in annos["name"]])
                annos = common_utils.drop_info_with_name(annos, name="others")
                num_obj = annos["gt_boxes_lidar"].shape[0]

                if self.gt_class_split is not None:
                    if num_obj > 0:
                        for k, v in self.gt_class_split.items():
                            class_mask = annos["name"] == k
                            remap_mask = np.logical_or(annos["gt_boxes_lidar"][:, 3] > v["l"],
                                                       annos["gt_boxes_lidar"][:, 4] > v["w"])
                            remap_mask = np.logical_or(remap_mask, annos["gt_boxes_lidar"][:, 5] > v["h"])
                            remap_mask = np.logical_and(remap_mask, class_mask)
                            annos["name"][remap_mask] = v["remap"]
                    annos["label_id"] = np.array([self.class_map[name] for name in annos["name"]])

                if num_obj <= 0:
                    annos = get_dummy_info(annos)

                eval_gt_annos[idx] = annos

        if kwargs["eval_metric"] == "my_data_eval_metric":
            ap_result_str_all, ap_dict_all = my_eval(eval_det_annos, eval_gt_annos, eval_iou_type=eval_iou_type,
                                                       logger=self.logger)
        else:
            raise NotImplementedError

        if self.dataset_cfg.get("SAVE_RESULT_WITH_GT", False):
            update_result_withgt(eval_det_annos, eval_gt_annos, kwargs["output_path"])

        return ap_result_str_all, ap_dict_all

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.my_infos) * self.total_epochs

        return len(self.my_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.my_infos)
        info = copy.deepcopy(self.my_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        input_dict = {
            'frame_id': sample_idx,
        }
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            annos = common_utils.drop_info_with_name(annos, name="unknown")
            annos = common_utils.drop_info_with_name(annos, name="others")
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']

            loc_lidar = loc
            l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
            gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx, num_features=self.point_num_features)
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        # if "calib_matricies" in get_item_list:
        #     input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = my_utils.calib_to_matricies(calib)

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_my_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = MyDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('my_infos_%s.pkl' % train_split)
    val_filename = save_path / ('my_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'my_infos_trainval.pkl'
    test_filename = save_path / 'my_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    my_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(my_infos_train, f)
    print('My info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    my_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(my_infos_val, f)
    print('My info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(my_infos_train + my_infos_val, f)
    print('My info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    my_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(my_infos_test, f)
    print('My info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_my_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_my_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Bicycle', 'Truck'],
            data_path=ROOT_DIR / 'data' / 'my_data',
            save_path=ROOT_DIR / 'data' / 'my_data'
        )