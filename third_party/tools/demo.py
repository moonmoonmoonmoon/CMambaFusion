import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

import sys
sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')

import fusion
print("fusion path:", fusion.__file__)
print("sys.path first 5:", sys.path[:5])
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import torchvision.transforms as transforms
from PIL import Image

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

        # ── 图像相关 ──────────────────────────────────────────────
        self.img_size = (128, 1024)  # Ouster Near-IR 尺寸
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def _find_image(self, bin_path):
        """
        根据点云路径推断对应的 Near-IR 图像路径。
        bin 文件示例:  .../testing/data/bus_30_03_pcd_out_004306.bin
        图像示例:      .../testing/images/bus_30_03_frame_004306_combined.png
        """
        import re
        stem = Path(bin_path).stem  # bus_30_03_pcd_out_004306
        m = re.match(r'(.+?)_pcd_out_(\d+)', stem)
        if not m:
            return None

        prefix = m.group(1)  # bus_30_03
        frame_num = m.group(2)  # 004306

        # 尝试短帧号（5位）
        short_num = str(int(frame_num)).zfill(5)  # 04306

        data_dir = Path(bin_path).parent  # .../testing/data
        # 向上一级找 images 目录
        # img_dir  = data_dir.parent / 'images'
        img_dir = Path('/home/yanan/Downloads/projects/multimodal_detection/data/Bus/val/images')

        candidates = [
            img_dir / f'{prefix}_frame_{frame_num}_combined.png',
            img_dir / f'{prefix}_frame_{short_num}_combined.png',
            img_dir / f'{prefix}_frame_{frame_num}.png',
            img_dir / f'{prefix}_frame_{short_num}.png',
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        bin_path = self.sample_file_list[index]

        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)

            # print('bin')
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        # ── Near-IR 图像 ──────────────────────────────────────────
        img_path = self._find_image(bin_path)
        if img_path is not None:
            image = Image.open(img_path).convert('RGB')
            data_dict['image'] = self.img_transform(image)  # (3, 128, 1024)
            print(f'[Demo] 加载图像: {img_path.name}')
        else:
            # 找不到图像时用零张量，防止 camera 分支崩溃
            data_dict['image'] = torch.zeros(3, *self.img_size)
            print(f'[Demo] 警告: 未找到图像，使用零张量代替')

        # ── dataset_flag（0=Bus, 1=Boston）────────────────────────
        stem = Path(bin_path).stem
        data_dict['dataset_flag'] = 1 if stem.startswith('boston') else 0

        return data_dict

    def collate_batch(self, batch_list):
        """在父类基础上额外处理 image 和 dataset_flag"""
        data_dict = super().collate_batch(batch_list)

        # 拼图像
        if 'image' in batch_list[0]:
            data_dict['images'] = torch.stack(
                [b['image'] for b in batch_list], dim=0
            )  # (B, 3, 128, 1024)

        # 拼 dataset_flags
        if 'dataset_flag' in batch_list[0]:
            data_dict['dataset_flags'] = torch.tensor(
                [b['dataset_flag'] for b in batch_list], dtype=torch.long
            )

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            if OPEN3D_FLAG:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()



# with GT boxes demo
# import argparse
# import glob
# from pathlib import Path
#
# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False
#
# import numpy as np
# import torch
# import json  # ← 新增：用于解析GT的json标注
#
# import sys
# sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
#
# import fusion
# print("fusion path:", fusion.__file__)
# print("sys.path first 5:", sys.path[:5])
# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu
# from pcdet.utils import common_utils
# import torchvision.transforms as transforms
# from PIL import Image
#
#
# class DemoDataset(DatasetTemplate):
#     def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
#         super().__init__(
#             dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
#         )
#         self.root_path = root_path
#         self.ext = ext
#         data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
#
#         data_file_list.sort()
#         self.sample_file_list = data_file_list
#
#         self.img_size = (128, 1024)
#         self.img_transform = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#         ])
#
#     def _find_image(self, bin_path):
#         import re
#         stem = Path(bin_path).stem
#         m = re.match(r'(.+?)_pcd_out_(\d+)', stem)
#         if not m:
#             return None
#         prefix = m.group(1)
#         frame_num = m.group(2)
#         short_num = str(int(frame_num)).zfill(5)
#         img_dir = Path('/home/yanan/Downloads/projects/multimodal_detection/data/Bus/val/images')
#         candidates = [
#             img_dir / f'{prefix}_frame_{frame_num}_combined.png',
#             img_dir / f'{prefix}_frame_{short_num}_combined.png',
#             img_dir / f'{prefix}_frame_{frame_num}.png',
#             img_dir / f'{prefix}_frame_{short_num}.png',
#         ]
#         for p in candidates:
#             if p.exists():
#                 return p
#         return None
#
#     # ── 新增：查找GT标注json文件 ──────────────────────────────────────
#     def _find_label(self, bin_path):
#         stem = Path(bin_path).stem  # bus_30_01_pcd_out_005181
#         label_dir = Path('/home/yanan/Downloads/projects/multimodal_detection/data/custom/training/label')
#         label_path = label_dir / f'{stem}.json'
#         return label_path if label_path.exists() else None
#
#     # ── 新增：解析GT json -> (N, 7) numpy array [x,y,z,l,w,h,yaw] ───
#     def _load_gt_boxes(self, label_path):
#         with open(label_path) as f:
#             boxes = json.load(f)
#         result = []
#         for box in boxes:
#             x   = box['position3d']['x']
#             y   = box['position3d']['y']
#             z   = box['position3d']['z']
#             l   = box['size3d']['x']
#             w   = box['size3d']['y']
#             h   = box['size3d']['z']
#             yaw = box['heading']
#             result.append([x, y, z, l, w, h, yaw])
#         return np.array(result, dtype=np.float32) if result else np.zeros((0, 7), dtype=np.float32)
#     # ─────────────────────────────────────────────────────────────────
#
#     def __len__(self):
#         return len(self.sample_file_list)
#
#     def __getitem__(self, index):
#         bin_path = self.sample_file_list[index]
#
#         if self.ext == '.bin':
#             points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
#         elif self.ext == '.npy':
#             points = np.load(self.sample_file_list[index])
#         else:
#             raise NotImplementedError
#
#         input_dict = {
#             'points': points,
#             'frame_id': index,
#         }
#
#         data_dict = self.prepare_data(data_dict=input_dict)
#
#         img_path = self._find_image(bin_path)
#         if img_path is not None:
#             image = Image.open(img_path).convert('RGB')
#             data_dict['image'] = self.img_transform(image)
#             print(f'[Demo] 加载图像: {img_path.name}')
#         else:
#             data_dict['image'] = torch.zeros(3, *self.img_size)
#             print(f'[Demo] 警告: 未找到图像，使用零张量代替')
#
#         stem = Path(bin_path).stem
#         data_dict['dataset_flag'] = 1 if stem.startswith('boston') else 0
#
#         # ── 新增：加载GT boxes ────────────────────────────────────────
#         gt_path = self._find_label(bin_path)
#         if gt_path is not None:
#             data_dict['gt_boxes_vis'] = self._load_gt_boxes(gt_path)
#             print(f'[Demo] 加载GT: {gt_path.name}，共 {len(data_dict["gt_boxes_vis"])} 个box')
#         else:
#             data_dict['gt_boxes_vis'] = np.zeros((0, 7), dtype=np.float32)
#             print(f'[Demo] 警告: 未找到GT标注')
#         # ─────────────────────────────────────────────────────────────
#
#         return data_dict
#
#     def collate_batch(self, batch_list):
#         data_dict = super().collate_batch(batch_list)
#
#         if 'image' in batch_list[0]:
#             data_dict['images'] = torch.stack(
#                 [b['image'] for b in batch_list], dim=0
#             )
#
#         if 'dataset_flag' in batch_list[0]:
#             data_dict['dataset_flags'] = torch.tensor(
#                 [b['dataset_flag'] for b in batch_list], dtype=torch.long
#             )
#
#         # ── 新增：collate gt_boxes_vis（不做batch合并，直接取第一个）──
#         if 'gt_boxes_vis' in batch_list[0]:
#             data_dict['gt_boxes_vis'] = batch_list[0]['gt_boxes_vis']
#         # ─────────────────────────────────────────────────────────────
#
#         return data_dict
#
#
# def parse_config():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml')
#     parser.add_argument('--data_path', type=str, default='demo_data')
#     parser.add_argument('--ckpt', type=str, default=None)
#     parser.add_argument('--ext', type=str, default='.npy')
#
#     args = parser.parse_args()
#     cfg_from_yaml_file(args.cfg_file, cfg)
#     return args, cfg
#
#
# def main():
#     args, cfg = parse_config()
#     logger = common_utils.create_logger()
#     logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
#     demo_dataset = DemoDataset(
#         dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
#         root_path=Path(args.data_path), ext=args.ext, logger=logger
#     )
#     logger.info(f'Total number of samples: \t{len(demo_dataset)}')
#
#     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
#     model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
#     model.cuda()
#     model.eval()
#     with torch.no_grad():
#         for idx, data_dict in enumerate(demo_dataset):
#             logger.info(f'Visualized sample index: \t{idx + 1}')
#
#             data_dict = demo_dataset.collate_batch([data_dict])
#             load_data_to_gpu(data_dict)
#             pred_dicts, _ = model.forward(data_dict)
#
#             # ── 新增：取出GT boxes ────────────────────────────────────
#             gt_boxes = data_dict.get('gt_boxes_vis', None)
#             # ─────────────────────────────────────────────────────────
#
#             if OPEN3D_FLAG:
#                 V.draw_scenes(
#                     points=data_dict['points'][:, 1:],
#                     ref_boxes=pred_dicts[0]['pred_boxes'],
#                     ref_scores=pred_dicts[0]['pred_scores'],
#                     ref_labels=pred_dicts[0]['pred_labels'],
#                     gt_boxes=gt_boxes  # ← 新增：传入GT
#                 )
#
#             if not OPEN3D_FLAG:
#                 mlab.show(stop=True)
#
#     logger.info('Demo done.')
#
#
# if __name__ == '__main__':
#     main()