
"""
修改后的自定义数据集
位置: third_party/pcdet/datasets/custom/custom_dataset.py

在原有数据集基础上添加图像数据加载
"""
import copy
import pickle
import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from ..augmentor.data_augmentor import DataAugmentor

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        # print('root_path', root_path)
        # print('da',dataset_cfg)
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # print('n',root_path)

        # 🔥 获取消融实验配置
        self.ablation_config = dataset_cfg.get('ABLATION_CONFIG', {})
        self.use_point_cloud_aug = self.ablation_config.get('USE_POINT_CLOUD_AUG', True)
        self.use_image_aug = self.ablation_config.get('USE_IMAGE_AUG', True)
        # self.use_different_seeds = self.ablation_config.get('USE_DIFFERENT_SEEDS', True)

        # print(f"🔬 数据集消融配置:")
        # print(self.ablation_config)
        # print(f"   点云数据增强: {self.use_point_cloud_aug}")
        # print(f"   图像数据增强: {self.use_image_aug}")
        # print(f"   使用不同种子: {self.use_different_seeds}")

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.data_type = 'custom'
        # self.train_for_debug_mode = self.dataset_cfg.get('TRAIN_FOR_DEBUG', False)
        #
        # if self.dataset_cfg.get('DATA_AUGMENTOR', None) is not None:
        #     self.data_augmentor = (
        #         DataAugmentor(self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger,
        #                       gt_name_remap=None, data_type=self.data_type) if self.training else None
        #     )
        # else:
        #     self.data_augmentor = None
        self.custom_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

        # 初始化图像相关配置
        self._init_image_config()

    def _init_image_config(self):
        """初始化图像配置"""
        # 检查是否启用多模态
        self.enable_multimodal = self.dataset_cfg.get('ENABLE_MULTIMODAL', False)

        if self.enable_multimodal:
            print("启用多模态数据加载")

            # 图像配置
            image_config = self.dataset_cfg.get('IMAGE_CONFIG', {})
            self.image_root = Path(image_config.get('IMAGE_ROOT', self.root_path.parent / 'Bus'))
            self.image_subdir = image_config.get('IMAGE_SUBDIR', 'train/images' if self.training else 'val/images')
            self.image_dir = self.image_root / self.image_subdir
            self.img_size = tuple(image_config.get('IMG_SIZE', [128, 1024]))

            # ── 新增：加载 Ouster beam_altitude_angles ──────────
            # OS1-128 所有录制传感器型号相同，只需加载一次
            metadata_path = image_config.get('METADATA_PATH', None)
            if metadata_path and os.path.exists(metadata_path):
                try:
                    from ouster.sdk import client
                    with open(metadata_path, 'r') as f:
                        metadata = client.SensorInfo(f.read())
                    self.beam_altitude_angles = list(metadata.beam_altitude_angles)
                    print(f"加载 beam_altitude_angles: {len(self.beam_altitude_angles)} beams")
                except Exception as e:
                    print(f"警告: 加载 metadata 失败 ({e})，使用默认 beam 角度")
                    self.beam_altitude_angles = None
            else:
                print("警告: 未配置 METADATA_PATH，beam_altitude_angles=None")
                self.beam_altitude_angles = None
            # ────────────────────────────────────────────────────

            # 设置图像预处理
            self._setup_image_transforms()

            # 建立frame_id到图像的映射
            self._build_image_mapping()

            print(f"图像目录: {self.image_dir}")
            print(f"图像尺寸: {self.img_size}")
            print(f"找到 {len(self.frame_id_to_image)} 张图像")
        else:
            print("未启用多模态，仅加载点云数据")
            self.beam_altitude_angles = None  # LiDAR-only 时也要有这个属性

    def _setup_image_transforms(self):
        """设置图像预处理（YOLO风格）"""
        """设置图像预处理（基础预处理，不含增强）"""
        # if self.training:
        #     print('setup_image_transforms: ',self.img_size)
        self.base_image_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(), # [0,255] → [0,1], HWC → CHW
        ])

    def no_apply_360_image_augmentation(self, image, data_dict):
        """
        针对360度扫描图像的一致性增强

        Args:
            image: PIL图像
            data_dict: 包含点云增强参数的数据字典
        Returns:
            augmented_image: 增强后的图像tensor
        """
        return self.base_image_transforms(image) if not isinstance(image, torch.Tensor) else image

    def apply_360_image_augmentation(self, image, data_dict):
        """
        针对360度扫描图像的一致性增强

        Args:
            image: PIL图像
            data_dict: 包含点云增强参数的数据字典

        Returns:
            augmented_image: 增强后的图像tensor
        """
        if not self.training or not self.enable_multimodal:
            return self.base_image_transforms(image) if not isinstance(image, torch.Tensor) else image

        # 转换为PIL图像（如果还不是）
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # 1. 基础预处理
        image = transforms.Resize(self.img_size)(image)
        image_array = np.array(image)  # 转为numpy便于操作
        h, w = image_array.shape[:2]

        # 2. 处理旋转（重点修正部分）
        if 'noise_rot' in data_dict:
            rotation_angle = data_dict['noise_rot']  # 弧度
            image_array = self._apply_360_rotation(image_array, rotation_angle)
            print('*******rotation_angle: ',rotation_angle)

        # 3. 处理翻转
        if 'flip_x' in data_dict and data_dict['flip_x']:
            # X轴翻转：上下象限交换
            image_array = self._apply_360_flip_x(image_array)
            # print('flip_x: ',data_dict['flip_x'])

        if 'flip_y' in data_dict and data_dict['flip_y']:
            # Y轴翻转：左右象限交换
            image_array = self._apply_360_flip_y(image_array)
            print('flip_y: ', data_dict['flip_y'])

        # 4. 处理缩放
        if 'noise_scale' in data_dict:
            scale_factor = data_dict['noise_scale']
            image_array = self._apply_360_scaling(image_array, scale_factor)
            # print('noise_scale: ', data_dict['noise_scale'])

        # 5. 转换回tensor
        image = Image.fromarray(image_array)
        image = transforms.ToTensor()(image)

        return image

    def _apply_360_rotation(self, image_array, rotation_angle):
        """
        应用360度图像旋转：根据角度计算像素偏移

        点云绕Z轴旋转angle，相当于在360度图像上水平循环移位
        """
        h, w, c = image_array.shape

        # 计算像素偏移：角度 -> 像素位置
        # 2π对应整个图像宽度w
        pixel_shift = -int((rotation_angle / (2 * np.pi)) * w)

        # 水平循环移位
        if pixel_shift != 0:
            image_array = np.roll(image_array, pixel_shift, axis=1)

        return image_array

    def _apply_360_flip_x(self, image_array):
        """
        应用X轴翻转：关于图像水平中心线的镜面对称

        X轴翻转意味着 y → -y，在360度图像中表现为：
        关于w/2位置的水平镜面对称变换
        """
        # 关于w/2的镜面对称变换
        flipped_array = np.flip(image_array, axis=1)  # 水平翻转

        return flipped_array

    def _apply_360_flip_y(self, image_array):
        """
        应用Y轴翻转：左右半部分分别关于各自中心线镜面对称

        Y轴翻转意味着 x → -x，在360度图像中表现为：
        - 左半部分（0到w/2）：两段分别关于w/4镜面对称
        - 右半部分（w/2到w）：两段分别关于3w/4镜面对称
        """
        h, w, c = image_array.shape
        flipped_array = image_array.copy()

        # 左半部分：[0, w/4) 和 [w/4, w/2) 关于 w/4 镜面对称
        quarter_w = w // 4
        half_w = w // 2
        three_quarter_w = 3 * w // 4

        # 左半部分翻转
        left_half = image_array[:, :half_w, :]  # [0, w/2)
        flipped_left = np.flip(left_half, axis=1)  # 关于w/4对称
        flipped_array[:, :half_w, :] = flipped_left

        # 右半部分翻转
        right_half = image_array[:, half_w:, :]  # [w/2, w)
        flipped_right = np.flip(right_half, axis=1)  # 关于3w/4对称
        flipped_array[:, half_w:, :] = flipped_right

        return flipped_array

    def _apply_360_scaling(self, image_array, scale_factor):
        """
        应用缩放：在360度图像中，缩放影响径向距离

        点云整体缩放意味着所有物体距离传感器更远/更近
        在360度图像中，这通常表现为：
        1. 高度方向的变化（如果高度编码了距离信息）
        2. 或者强度的变化（距离越远，反射强度可能越低）

        这里采用保守策略：仅做轻微的高度调整
        """
        h, w, c = image_array.shape

        if abs(scale_factor - 1.0) < 0.01:  # 缩放太小则跳过
            return image_array

        # 保守策略：对360度图像只做轻微高度调整
        # 因为360度图像的几何关系比较复杂，过度变换可能破坏空间对应关系

        if scale_factor > 1.0:
            # 放大：物体更近，可能在图像中更"突出"
            # 轻微增加图像高度然后裁剪
            scale_ratio = min(scale_factor, 1.1)  # 限制最大缩放
            new_h = int(h * scale_ratio)
            scaled = cv2.resize(image_array, (w, new_h))
            if new_h > h:
                start_h = (new_h - h) // 2
                scaled = scaled[start_h:start_h + h, :, :]

        else:
            # 缩小：物体更远，在图像中更"平坦"
            scale_ratio = max(scale_factor, 0.9)  # 限制最小缩放
            new_h = int(h * scale_ratio)
            scaled = cv2.resize(image_array, (w, new_h))
            # 居中填充
            pad_h = (h - new_h) // 2
            scaled = np.pad(scaled, ((pad_h, h - new_h - pad_h), (0, 0), (0, 0)), mode='constant')

        return scaled

    def _build_image_mapping(self):
        """建立frame_id到图像文件的映射"""
        self.frame_id_to_image = {}
        # self.image_dir = self.image_root / self.image_subdir
        # self.image_dir = self.image_root / ('train/images' if self.training else 'val/images')
        self.image_dir = self.image_root / f'{self.split}/images'
        print('exist',self.image_dir)

        if self.image_dir.exists():
            for img_file in self.image_dir.glob('*.png'):
                frame_id = self._extract_frame_id(img_file.name)
                if frame_id:
                    self.frame_id_to_image[frame_id] = img_file

        print(f"建立图像映射: {len(self.frame_id_to_image)} 个文件")

    def _extract_frame_id(self, filename):
        """从文件名提取frame_id"""
        import re
        name = Path(filename).stem

        # 🔥 新增：匹配新格式 prefix_frame_xxxxx
        match = re.match(r'(.+?)_frame_(\d+)', name)
        if match:
            prefix = match.group(1)  # bus_30_01
            frame_num = match.group(2).zfill(6)  # 00251 -> 000251
            # return frame_num
            return f"{prefix}_pcd_out_{frame_num}"  # bus_30_01_pcd_out_000251

        # 尝试不同的命名模式
        patterns = [
            r'frame_(\d+)',  # frame_00001_combined
            r'^(\d+)$',  # 000001
            r'(\d+)_',  # 000001_anything
        ]

        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                # return match.group(1).lstrip('0') or '0'
                return match.group(1).zfill(6) or '0'

        return name

    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

        self.custom_infos.extend(custom_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))

    def get_label(self, idx):
        label_file = self.root_split_path / 'label' / ('%s.json' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            label_file = json.load(f)
        return label_file

    def get_lidar(self, idx):
        """
        读取点云数据，支持.bin和.npy格式，自动检测列数
        """
        # 首先尝试.bin格式
        lidar_bin_file = self.root_split_path / 'data' / ('%s.bin' % idx)
        lidar_npy_file = self.root_split_path / 'data' / ('%s.npy' % idx)

        if lidar_bin_file.exists():
            # 读取.bin文件
            point_features = np.fromfile(str(lidar_bin_file), dtype=np.float32)
            # no near_ir
            point_features = point_features.reshape(-1, 4)
            # with near_ir
            # point_features = point_features.reshape(-1, 5)

        elif lidar_npy_file.exists():
            # 兼容原来的.npy格式
            point_features = np.load(lidar_npy_file)
            self.logger.info(f'Loaded {idx}.npy with {point_features.shape[1]} columns')
        else:
            raise FileNotFoundError(f'Neither {idx}.bin nor {idx}.npy exists in data folder')

        return point_features

    def _load_image(self, image_path):
        """加载图像"""

        # 使用PIL加载图像
        image = Image.open(image_path)

        # 只使用reflectivity通道（reflectivity是第2个通道）
        # 将图像转为numpy进行通道操作
        import numpy as np
        img_array = np.array(image)

        # # 只保留第一个通道，复制3次形成RGB格式
        # near_ir_channel = img_array[:, :, 0]  # near_ir是第一个通道
        # img_array = np.stack([near_ir_channel, near_ir_channel, near_ir_channel], axis=2)

        # 转回PIL图像
        image = Image.fromarray(img_array)
        # print(f"🔥 只使用near_ir_channel通道加载图像")
        return image

    def get_raw_image(self, idx):
        """获取对应的图像数据"""
        if not self.enable_multimodal:
            return None

        # 处理frame_id（移除pcd_out_前缀）
        frame_id = idx
        if frame_id.startswith('pcd_out_'):
            frame_id = frame_id[8:]

        # 查找对应的图像
        if frame_id in self.frame_id_to_image:
            image_path = self.frame_id_to_image[frame_id]
            return self._load_image(image_path)
        else:
            # 如果找不到对应图像，返回黑色图像
            print(f"警告: 未找到frame_id {frame_id} 对应的图像")
            return torch.zeros(3, *self.img_size)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']

        # 为数据增强设置确定性种子
        if self.training:
            # 从pcd_out_003871中提取003871
            import re
            numbers = re.findall(r'\d+', sample_idx)
            if numbers:
                # 直接使用提取的数字
                seed_base = int(numbers[0])  # 003871 -> 3871
            else:
                seed_base = 0

            # 设置种子
            aug_seed = (seed_base + 666)
            np.random.seed(aug_seed)
            torch.manual_seed(aug_seed)
            # test_random = np.random.random()
            # # 验证打印
            # test_random = np.random.random()
            # print(f"Sample {sample_idx} (index {index}): random={test_random:.6f}, seed={aug_seed}")

        # info = copy.deepcopy(self.custom_infos[index])
        # sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        # print('point.shape',points.shape)
        input_dict = {
            'frame_id': sample_idx,  # 使用sample_idx作为frame_id
            'points': points
        }

        # 先获取原始图像
        raw_image = None
        if self.enable_multimodal:
            raw_image = self.get_raw_image(sample_idx)
        # # 获取图像数据（如果启用多模态）
        # if self.enable_multimodal:
        #     image = self.get_image(sample_idx)
        #     input_dict['image'] = image

        # 🔥 新增：加载图像标签
        if self.enable_multimodal and self.training:
            image_labels = self.get_image_labels(sample_idx)
            # print('sample_idx1: ', sample_idx)
            # sample_idx1:  bus_30_03_pcd_out_006471
            # print('image_labels: ', image_labels)
            # image_labels:  [{'class_id': 0, 'bbox': [0.570801, 0.515625, 0.016602, 0.078125]}, {'class_id': 2, 'bbox': [0.271973, 0.4375, 0.004883, 0.140625]}, {'class_id': 0, 'bbox': [0.757324, 0.480469, 0.014648, 0.070312]}, {'class_id': 4, 'bbox': [0.735352, 0.480469, 0.017578, 0.070312]}, {'class_id': 0, 'bbox': [0.90918, 0.457031, 0.056641, 0.164062]}, {'class_id': 0, 'bbox': [0.984375, 0.449219, 0.017578, 0.085938]}, {'class_id': 0, 'bbox': [0.958984, 0.449219, 0.021484, 0.085938]}]
            if image_labels is not None:
                input_dict['image_labels'] = image_labels
                # print(' input_dict[image_labels]:', image_labels)

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        # 🔥 应用点云数据增强（可选）
        if self.use_point_cloud_aug:
            # print("raw gt_names:", input_dict["gt_names"])
            # print("raw gt_boxes shape:", np.array(input_dict["gt_boxes"]).shape)

            data_dict = self.prepare_data(data_dict=input_dict)
            # print("✓ 应用点云数据增强")
            # print('1', data_dict)
        else:
            # 跳过点云数据增强，只做基本处理
            data_dict = self.only_prepare_data(data_dict=input_dict)
            print("⊗ 跳过点云数据增强")
            # print('1',data_dict)

        # 应用360度图像一致性增强
        if self.enable_multimodal and raw_image is not None:
            if self.use_image_aug:
                augmented_image = self.apply_360_image_augmentation(raw_image, data_dict)
                # print("✓ 应用图像数据增强")
            else:
                augmented_image = self.no_apply_360_image_augmentation(raw_image, data_dict)
                # print("⊗ 跳过图像数据增强")
            data_dict['image'] = augmented_image
        # ── 数据集标识（0=Bus，1=Boston），供OusterLSSTransformDual使用 ──
        if self.enable_multimodal:
            is_boston = 1 if sample_idx.startswith('boston') else 0
            data_dict['dataset_flag'] = is_boston

        return data_dict

        # 新增图像标签加载方法：

    def get_image_labels(self, idx):
        """加载对应的图像标签（YOLO格式）"""
        if not self.enable_multimodal:
            return None

        # 处理frame_id
        frame_id = idx  # idx = "bus_30_01_pcd_out_000251"

        # 🔥 新增：提取前缀和帧号
        import re
        # 提取前缀（bus_30_01）
        prefix_match = re.match(r'(.+?)_pcd_out_(\d+)', frame_id)
        if prefix_match:
            prefix = prefix_match.group(1)  # bus_30_01
            frame_num = prefix_match.group(2)  # 000251
        else:
            # 兼容旧格式
            if frame_id.startswith('pcd_out_'):
                prefix = ''
                frame_num = frame_id[9:]
            else:
                prefix = ''
                frame_num = frame_id

        # 构建标签文件路径
        # labels_subdir = self.image_subdir.replace('images', 'labels')
        # labels_subdir = 'train/labels' if self.training else 'val/labels'
        labels_subdir = f'{self.split}/labels'
        labels_dir = self.image_root / labels_subdir

        # 🔥 更新：尝试不同的标签文件命名模式
        possible_names = [
            f"{prefix}_frame_{frame_num}_combined.txt" if prefix else f"frame_{frame_num}_combined.txt",
            f"{prefix}_frame_{frame_num}.txt" if prefix else f"frame_{frame_num}.txt",
        ]

        # 🔥 新增：处理帧号位数不同的情况（000251 vs 00251）
        if len(frame_num) > 5:
            short_frame_num = str(int(frame_num)).zfill(5)  # 去掉前导0，重新填充到5位
            if prefix:
                possible_names.extend([
                    f"{prefix}_frame_{short_frame_num}_combined.txt",
                    f"{prefix}_frame_{short_frame_num}.txt",
                ])

        label_file = None
        for name in possible_names:
            potential_path = labels_dir / name
            if potential_path.exists():
                label_file = potential_path
                break

        if label_file is None:
            print(f"警告: 未找到frame_id {frame_id} 对应的图像标签")
            return None

        return self._load_yolo_labels(label_file)

    def _load_yolo_labels(self, label_file):
        """
        加载YOLO格式的标签文件

        Args:
            label_file: 标签文件路径

        Returns:
            labels: 解析后的标签数据
        """
        try:
            labels = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # YOLO格式: class_id center_x center_y width height
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            center_x = float(parts[1])
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            labels.append({
                                'class_id': class_id,
                                'bbox': [center_x, center_y, width, height]  # YOLO格式 (归一化坐标)
                            })

            return labels if labels else None
        except Exception as e:
            print(f"图像标签加载失败 {label_file}: {e}")
            return None

    def collate_batch(self, batch_list):
        """批次整理函数（修改版，处理图像数据）"""
        data_dict = {}
        # print('run collate_batch')

        # 处理点云相关数据（保持原有逻辑）
        for key in batch_list[0]:
            # if key == 'image':
            if key in ['image', 'image_labels', 'dataset_flag']:  # 跳过图像相关数据
                continue  # 图像数据单独处理

            data_dict[key] = [batch[key] for batch in batch_list]

        # 原有的点云数据整理逻辑
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        # ── 收集dataset_flag，拼成(B,)的tensor ──
        if 'dataset_flag' in batch_list[0]:
            flags = [batch.get('dataset_flag', 0) for batch in batch_list]
            ret['dataset_flags'] = torch.tensor(flags, dtype=torch.long)

        # 处理图像数据
        if self.enable_multimodal and 'image' in batch_list[0]:
            print('image is loaded in the batch')
            images = []
            for batch in batch_list:
                if 'image' in batch and batch['image'] is not None:
                    images.append(batch['image'])

            if images:
                ret['images'] = torch.stack(images, dim=0)  # [B, 3, H, W]
                print('ret[images]', ret['images'].shape)

            # 🔥 新增：处理图像标签
        if self.enable_multimodal and any('image_labels' in batch for batch in batch_list):
            image_labels = []
            for batch in batch_list:
                if 'image_labels' in batch and batch['image_labels'] is not None:
                    image_labels.append(batch['image_labels'])
                    # print('image_labels: ',image_labels)
                else:
                    image_labels.append(None)  # 没有标签的样本

            ret['image_labels'] = image_labels
            # print('ret', ret['image_labels'])

        return ret

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        if kwargs['eval_metric'] == 'kitti':
            # for i, (det, gt) in enumerate(zip(det_annos, eval_gt_annos)):
            #     print(f"Sample {i}: GT框={len(gt['name'])}, 预测框={len(det['name'])}")
            #     print(f"预测置信度: {det['score'] if 'score' in det else 'N/A'}")
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                # annotations = {}
                # gt_boxes_lidar, name = self.get_label(sample_idx)
                # annotations['name'] = name
                # # annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                obj_list = self.get_label(sample_idx)
                # 检查是否有标注对象
                if len(obj_list) == 0:
                    annotations = {}
                    annotations['name'] = np.array([])
                    annotations["location"] = np.zeros((0, 3), dtype=np.float32)
                    annotations["dimensions"] = np.zeros((0, 3), dtype=np.float32)
                    annotations["rotation_y"] = np.array([])
                    annotations["difficulty"] = np.array([])
                    annotations["gt_boxes_lidar"] = np.zeros((0, 7), dtype=np.float32)
                else:
                    annotations = {}
                    annotations['name'] = np.array([obj["type"].replace("TYPE_", "") for obj in obj_list])
                    annotations["location"] = np.array(
                        [[obj["position3d"]["x"], obj["position3d"]["y"], obj["position3d"]["z"]] for obj in obj_list])
                    annotations["dimensions"] = np.array(
                        [[obj["size3d"]["x"], obj["size3d"]["y"], obj["size3d"]["z"]] for obj in obj_list])
                    annotations["rotation_y"] = np.array([float(obj["heading"]) for obj in obj_list])
                    annotations["difficulty"] = np.array(
                        [3 - int(obj["tag"]["confidence"]) if "confidence" in obj["tag"] else -1 for obj in obj_list])
                    annotations["gt_boxes_lidar"] = np.hstack(
                        (annotations["location"], annotations["dimensions"], annotations["rotation_y"].reshape(-1, 1)))

                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        total_objects = 0
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            # print(points.shape)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            # 如果没有标注框，跳过
            if len(names) == 0 or gt_boxes.shape[0] == 0:
                print(f'Empty annotations for sample {sample_idx}, skipping...')
                continue

            num_obj = gt_boxes.shape[0]
            total_objects += num_obj

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                # 检查是否有足够的点
                num_points_in_box = np.sum(point_indices[i] > 0)

                if num_points_in_box < 2:  # 至少需要5个点
                # if num_points_in_box < 5:
                    print(
                        f'Not enough points ({num_points_in_box}) for object {i} ({names[i]}) in sample {sample_idx}, skipping...')
                    continue
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                # 确保有点可以保存
                if gt_points.shape[0] > 0:
                # if gt_points.shape[0] >= 0:
                    with open(filepath, 'wb') as f:
                        gt_points.tofile(f)

                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'path': db_path, 'gt_idx': i, 'sample_idx,': sample_idx,
                                   'box3d_lidar': gt_boxes[i], 'difficulty': difficulty[i],
                                   'num_points_in_gt': gt_points.shape[0]}
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
        # 统计信息
        print(f'\n=== Database Generation Summary ===')
        print(f'Total objects processed: {total_objects}')

        if all_db_infos:
            for k, v in all_db_infos.items():
                print(f'Database {k}: {len(v)} samples')

            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)
            print(f'Database info saved to {db_info_save_path}')
        else:
            print('Warning: No valid database entries found!')
            print('This could be due to:')
            print('1. All objects have insufficient points (< 5 points)')
            print('2. No annotations in the dataset')
            print('3. Point cloud and annotation alignment issues')

            # 创建空数据库文件，避免后续读取错误
            empty_db = {}
            with open(db_info_save_path, 'wb') as f:
                pickle.dump(empty_db, f)
            print(f'Empty database file created at {db_info_save_path}')

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)

    def debug_360_augmentation(self, index=0):
        """调试360度图像增强效果"""
        import matplotlib.pyplot as plt

        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']

        # 原始数据
        original_points = self.get_lidar(sample_idx)
        original_image = self.get_raw_image(sample_idx)

        # 模拟不同的增强参数
        test_cases = [
            {'name': 'Original', 'params': {}},
            {'name': 'Rotate 90°', 'params': {'noise_rot': np.pi / 2}},
            {'name': 'Rotate 180°', 'params': {'noise_rot': np.pi}},
            {'name': 'Flip X', 'params': {'flip_x': True}},
            {'name': 'Scale 1.2', 'params': {'noise_scale': 1.2}},
        ]

        fig, axes = plt.subplots(2, len(test_cases), figsize=(20, 8))

        for i, case in enumerate(test_cases):
            # 模拟增强参数
            mock_data_dict = case['params'].copy()
            mock_data_dict['points'] = original_points  # 添加必要字段

            # 应用图像增强
            if case['name'] == 'Original':
                aug_image = self.base_image_transforms(original_image)
            else:
                aug_image = self.apply_360_image_augmentation(original_image, mock_data_dict)

            # 显示图像
            axes[0, i].imshow(aug_image.permute(1, 2, 0) if isinstance(aug_image, torch.Tensor) else aug_image)
            axes[0, i].set_title(f'{case["name"]}\nImage')
            axes[0, i].axis('off')

            # 显示对应的点云变化（模拟）
            points = original_points.copy()
            if 'noise_rot' in case['params']:
                angle = case['params']['noise_rot']
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                # 旋转点云
                x_new = points[:, 0] * cos_a - points[:, 1] * sin_a
                y_new = points[:, 0] * sin_a + points[:, 1] * cos_a
                points[:, 0], points[:, 1] = x_new, y_new
            elif 'flip_x' in case['params']:
                points[:, 1] = -points[:, 1]  # Y坐标翻转
            elif 'noise_scale' in case['params']:
                scale = case['params']['noise_scale']
                points[:, :3] *= scale

            axes[1, i].scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
            axes[1, i].set_title(f'{case["name"]}\nPoint Cloud')
            axes[1, i].set_aspect('equal')
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'360_augmentation_debug_{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 打印象限映射验证
        print("\n360度图像象限映射验证:")
        print("第一象限 (x>0, y>0) → 图像 (w/4, w/2)")
        print("第二象限 (x<0, y>0) → 图像 (0, w/4)")
        print("第三象限 (x<0, y<0) → 图像 (3w/4, w)")
        print("第四象限 (x>0, y<0) → 图像 (w/2, 3w/4)")


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'custom_infos_trainval.pkl'
    test_filename = save_path / 'custom_infos_test.pkl'

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(custom_infos_train + custom_infos_val, f)
    print('Custom info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    custom_infos_test = dataset.get_infos(class_names, num_workers=workers, has_label=True, num_features=num_features)
    # custom_infos_test = dataset.get_infos(class_names, num_workers=workers, has_label=False, num_features=num_features)
    with open(test_filename, 'wb') as f:
        pickle.dump(custom_infos_test, f)
    print('Custom info test file is saved to %s' % test_filename)


    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            #class_names=['Car', 'Pedestrian', 'PickupTruck'],
            class_names=['Car', 'Pedestrian', 'Cyclist', 'MediumTruck', 'PickupTruck', 'SemiTruck', 'Bus', 'Train',
                         'Scooter', 'TowedObject', 'Motorcycle'],
            data_path=ROOT_DIR / 'data' / 'custom',
            save_path=ROOT_DIR / 'data' / 'custom',
        )