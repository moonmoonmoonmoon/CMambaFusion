import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import random
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
#当虚拟环境没有进行 setup develop时
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# # 查看当前工作目录
# print("当前工作目录:", os.getcwd())
#
# # 查看当前文件路径
# print("当前文件:", __file__)
# # 验证pcdet是否能被找到
# try:
#     import pcdet
#     print("✓ pcdet模块导入成功")
#     print("pcdet位置:", pcdet.__file__)
# except ImportError as e:
#     print("✗ pcdet模块导入失败:", e)

import sys

sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='config file')
    parser.add_argument('--batch_size', type=int, default=None, required=False)
    parser.add_argument('--epochs', type=int, default=None, required=False)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extra_tag', type=str, default='default')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to resume')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model')

    # 预训练权重路径
    parser.add_argument('--pretrained_yolo', type=str, default=None, help='预训练YOLOv8权重路径(.pt)')
    parser.add_argument('--pretrained_pointpillar', type=str, default=None, help='预训练PointPillar权重路径(.pth)')
    parser.add_argument('--freeze_yolo', action='store_true', help='冻结YOLO权重')
    parser.add_argument('--freeze_pointpillar', action='store_true', help='冻结PointPillar权重')
    parser.add_argument('--pretrained_bevfusion', type=str, default=None,
                        help='BevFusion† + AnchorHead预训练checkpoint路径')
    parser.add_argument('--freeze_lidar_backbone', action='store_true',
                        help='冻结BEVFusion LiDAR backbone')

    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)
    parser.add_argument('--sync_bn', action='store_true', default=False)
    parser.add_argument('--fix_random_seed', action='store_true', default=False)
    parser.add_argument('--ckpt_save_interval', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_ckpt_save_num', type=int, default=30)
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--max_waiting_mins', type=int, default=0)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs_to_eval', type=int, default=0)
    parser.add_argument('--save_to_file', action='store_true', default=False)
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False)
    parser.add_argument('--logger_iter_interval', type=int, default=50)
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300)
    parser.add_argument('--wo_gpu_stat', action='store_true')
    parser.add_argument('--use_amp', action='store_true')

    # 消融实验参数
    parser.add_argument('--run_id', type=int, default=0, help='实验运行ID')
    parser.add_argument('--ablation_mode', type=str, default='full',
                        choices=['full', 'no_cross_attn', 'no_self_attn', 'baseline_only'],
                        help='消融实验模式')
    # 随机噪声对比实验
    parser.add_argument('--use_random_noise', action='store_true',
                        help='使用随机噪声替代图像特征（对比实验）')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    # 消融模式配置
    if args.ablation_mode == 'baseline_only':
        cfg.ENABLE_MULTIMODAL_FUSION = False
        cfg.MODEL.ENABLE_MULTIMODAL_FUSION = False
        cfg.DATA_CONFIG.ENABLE_MULTIMODAL = False
    elif args.ablation_mode == 'no_cross_attn':
        cfg.MODEL.ABLATION_CONFIG.USE_CROSS_ATTENTION = False
    elif args.ablation_mode == 'no_self_attn':
        cfg.MODEL.ABLATION_CONFIG.USE_SELF_ATTENTION = False

    # 传递随机噪声配置到模型
    if args.use_random_noise:
        cfg.MODEL.USE_RANDOM_NOISE = True
        print("⚠️  将使用随机噪声替代图像特征进行对比实验")
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_params_to_txt(state_dict, filename):
    """保存参数到txt文件"""
    with open(filename, 'w') as f:
        for key in sorted(state_dict.keys()):
            param = state_dict[key]
            f.write(f"{key}\n")
            f.write(f"  Shape: {param.shape}\n")

            # 🔥 只处理浮点型
            if param.is_floating_point():
                f.write(f"  Mean: {param.mean().item():.6f}\n")
                f.write(f"  Std: {param.std().item():.6f}\n")
                f.write(f"  First 5: {param.flatten()[:5].tolist()}\n")
            else:
                f.write(f"  Dtype: {param.dtype} (non-float, skipped)\n")

            f.write("-" * 80 + "\n")


def load_pretrained_weights(model, args, logger):
    """加载预训练权重并冻结（如果需要）"""
    # 加载PointPillar预训练权重
    if args.pretrained_pointpillar and os.path.exists(args.pretrained_pointpillar):
        logger.info(f"加载PointPillar预训练权重: {args.pretrained_pointpillar}")
        checkpoint = torch.load(args.pretrained_pointpillar, map_location='cpu')

        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        # 🔥 保存预训练模型参数
        save_params_to_txt(state_dict, 'pretrained_params.txt')

        # 只加载PointPillar相关权重（排除YOLO和融合模块）
        pp_state_dict = {k: v for k, v in state_dict.items()
                         if not any(x in k for x in ['yolo', 'fusion'])}
                         # if not any(x in k for x in ['yolo', 'fusion', 'detection_head'])}

        model.load_state_dict(pp_state_dict, strict=False)
        # 🔥 保存加载后的模型参数
        save_params_to_txt(model.state_dict(), 'loaded_params.txt')
        logger.info(f"成功加载 {len(pp_state_dict)} 个PointPillar参数")

        if args.freeze_pointpillar:
            logger.info("冻结PointPillar权重...")
            for name, param in model.named_parameters():
                # if not any(x in name for x in ['yolo', 'fusion', 'alpha']):
                if not any(x in name for x in ['yolo', 'fusion', 'alpha', 'dense_head']):
                    param.requires_grad = False
            logger.info("PointPillar权重已冻结")

    # YOLO权重在YOLOv8FeatureExtractor初始化时加载
    if args.pretrained_yolo:
        logger.info(f"YOLO预训练权重路径已设置: {args.pretrained_yolo}")
        logger.info(f"YOLO权重冻结: {args.freeze_yolo}")

    if hasattr(args, 'pretrained_bevfusion') and args.pretrained_bevfusion and \
            os.path.exists(args.pretrained_bevfusion):

        logger.info(f"加载BevFusion†预训练权重: {args.pretrained_bevfusion}")
        checkpoint = torch.load(args.pretrained_bevfusion, map_location='cpu')
        state_dict = checkpoint.get('model_state', checkpoint)

        # 只加载LiDAR backbone相关权重，跳过AnchorHead（检测头结构不同）
        # backbone_keys = [k for k in state_dict.keys()
        #                  if not any(x in k for x in [
        #         'dense_head',  # AnchorHead，不加载
                # 'yolo_extractor',  # 单独处理
                # 'lss_transform',  # 单独处理
                # 'conv_fuser',  # 单独处理
            # ])]

        backbone_keys = [k for k in state_dict.keys()
                         if 'dense_head' not in k]  # 只排除检测头

        backbone_state = {k: v for k, v in state_dict.items() if k in backbone_keys}
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)

        logger.info(f"加载成功: {len(backbone_state)} 个参数")
        logger.info(f"未加载(新增): {len(missing)} 个参数（TransFusionHead部分）")

        # 冻结BEVFusion LiDAR backbone
        if hasattr(args, 'freeze_lidar_backbone') and args.freeze_lidar_backbone:
            frozen_modules = ['vfe', 'backbone_3d', 'map_to_bev_module', 'backbone_2d']
            frozen_count = 0
            for name, param in model.named_parameters():
                if any(m in name for m in frozen_modules):
                    param.requires_grad = False
                    frozen_count += 1
            logger.info(f"冻结BEVFusion LiDAR backbone参数: {frozen_count} 个")
            logger.info("TransFusionHead参数将从零训练")


def main():
    args, cfg = parse_config()

    seed = 42 + args.run_id * 1000 if args.fix_random_seed else 42
    set_seed(seed)

    # set_seed(42)
    #
    # if args.fix_random_seed:
    #     seed = 42 + args.run_id * 1000
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    output_dir = Path(
        __file__).resolve().parent.parent.parent / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')
    logger.info(f'消融实验模式: {args.ablation_mode}')
    logger.info(f'实验运行ID: {args.run_id}')

    if args.pretrained_yolo:
        logger.info(f'YOLOv8预训练权重: {args.pretrained_yolo}')
        logger.info(f'YOLO权重冻结: {args.freeze_yolo}')
    if args.pretrained_pointpillar:
        logger.info(f'PointPillar预训练权重: {args.pretrained_pointpillar}')
        print('a', {args.pretrained_pointpillar})
        logger.info(f'PointPillar权重冻结: {args.freeze_pointpillar}')

    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")

    root_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'custom'

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        root_path=root_path,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        # seed=42 if args.fix_random_seed else None
        # seed=42 + args.run_id * 1000 if args.fix_random_seed else 42
        seed=seed
    )

    # 将预训练权重路径传递给cfg，供YOLOv8FeatureExtractor使用
    if args.pretrained_yolo:
        cfg.MODEL.YOLO_CONFIG.PRETRAINED_WEIGHTS = args.pretrained_yolo
        cfg.MODEL.YOLO_CONFIG.FREEZE_WEIGHTS = args.freeze_yolo
    # print('777')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # print('7878')
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # 加载预训练权重
    load_pretrained_weights(model, args, logger)

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    start_epoch = it = 0
    last_epoch = -1

    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer,
                                                           logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    print('1')
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    logger.info(f'Model created, total params: {sum([m.numel() for m in model.parameters()]):,}')
    logger.info(f'Trainable params: {sum([m.numel() for m in model.parameters() if m.requires_grad]):,}')

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    logger.info('**********************Start training**********************')

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training**********************')

    logger.info('**********************Start evaluation**********************')

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        root_path=root_path,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=False
    )

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )

    logger.info('**********************End evaluation**********************')


if __name__ == '__main__':
    main()

# def load_pretrained_weights(model, args, logger):
#     """加载预训练权重并冻结（如果需要）"""
#
#     print('www', os.getcwd())
#     if args.pretrained_pointpillar:
#         path = args.pretrained_pointpillar
#
#         logger.info("=" * 80)
#         logger.info("🔍 调试路径信息")
#         logger.info(f"路径字符串: '{path}'")
#         logger.info(f"路径长度: {len(path)}")
#         logger.info(f"路径repr: {repr(path)}")  # ← 重要：显示隐藏字符
#         logger.info(f"路径类型: {type(path)}")
#         logger.info(f"路径存在: {os.path.exists(path)}")
#
#         # 逐级检查路径
#         parts = path.split('/')
#         current = ''
#         for i, part in enumerate(parts):
#             if part:  # 跳过空字符串
#                 current += '/' + part
#                 exists = os.path.exists(current)
#                 logger.info(f"  [{i}] {current}: {exists}")
#
#         # 检查文件名
#         filename = os.path.basename(path)
#         parent = os.path.dirname(path)
#         logger.info(f"文件名: '{filename}'")
#         logger.info(f"父目录: '{parent}'")
#         logger.info(f"父目录存在: {os.path.exists(parent)}")
#
#         # 列出父目录内容
#         if os.path.exists(parent):
#             files = os.listdir(parent)
#             logger.info(f"父目录中的文件数: {len(files)}")
#             pth_files = [f for f in files if f.endswith('.pth')]
#             logger.info(f".pth文件: {pth_files}")
#
#             # 检查文件名是否匹配
#             if filename in files:
#                 logger.info(f"✅ 文件名匹配!")
#             else:
#                 logger.info(f"❌ 文件名不匹配!")
#                 logger.info(f"相似文件: {[f for f in files if 'checkpoint' in f]}")
#
#         logger.info("=" * 80)
#     # 加载PointPillar预训练权重
#     if args.pretrained_pointpillar and os.path.exists(args.pretrained_pointpillar):
#         logger.info(f"加载PointPillar预训练权重: {args.pretrained_pointpillar}")
#         checkpoint = torch.load(args.pretrained_pointpillar, map_location='cpu')
#
#         if 'model_state' in checkpoint:
#             state_dict = checkpoint['model_state']
#         else:
#             state_dict = checkpoint
#
#         # 🔥 保存预训练模型参数
#         save_params_to_txt(state_dict, 'pretrained_params.txt')
#
#         # 只加载PointPillar相关权重（排除YOLO和融合模块）
#         pp_state_dict = {k: v for k, v in state_dict.items()
#                          if not any(x in k for x in ['yolo', 'fusion'])}
#                          # if not any(x in k for x in ['yolo', 'fusion', 'detection_head'])}
#
#         model.load_state_dict(pp_state_dict, strict=False)
#         # 🔥 保存加载后的模型参数
#         save_params_to_txt(model.state_dict(), 'loaded_params.txt')
#         logger.info(f"成功加载 {len(pp_state_dict)} 个PointPillar参数")
#
#         if args.freeze_pointpillar:
#             logger.info("冻结PointPillar权重...")
#             for name, param in model.named_parameters():
#                 # if not any(x in name for x in ['yolo', 'fusion', 'alpha']):
#                 if not any(x in name for x in ['yolo', 'fusion', 'alpha', 'dense_head']):
#                     param.requires_grad = False
#             logger.info("PointPillar权重已冻结")
#
#     # YOLO权重在YOLOv8FeatureExtractor初始化时加载
#     if args.pretrained_yolo:
#         logger.info(f"YOLO预训练权重路径已设置: {args.pretrained_yolo}")
#         logger.info(f"YOLO权重冻结: {args.freeze_yolo}")

# import _init_path
# import argparse
# import datetime
# import glob
# import os
# from pathlib import Path
# from test import repeat_eval_ckpt
#
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from tensorboardX import SummaryWriter
#
# #当虚拟环境没有进行 setup develop时
# # import sys
# # import os
# # sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# # # 查看当前工作目录
# # print("当前工作目录:", os.getcwd())
# #
# # # 查看当前文件路径
# # print("当前文件:", __file__)
# # # 验证pcdet是否能被找到
# # try:
# #     import pcdet
# #     print("✓ pcdet模块导入成功")
# #     print("pcdet位置:", pcdet.__file__)
# # except ImportError as e:
# #     print("✗ pcdet模块导入失败:", e)
# import sys
# sys.path.insert(0, '/home/yanan/Downloads/projects/multimodal_detection/third_party')
# from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
# from pcdet.datasets import build_dataloader
# from pcdet.models import build_network, model_fn_decorator
# from pcdet.utils import common_utils
# from train_utils.optimization import build_optimizer, build_scheduler
# from train_utils.train_utils import train_model
#
#
# def parse_config():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
#
#     parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
#     parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
#     parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
#     parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
#     parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
#     parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
#
#     # 预训练权重路径
#     parser.add_argument('--pretrained_yolo', type=str, default=None, help='预训练YOLOv8权重路径(.pt)')
#     parser.add_argument('--pretrained_pointpillar', type=str, default=None, help='预训练PointPillar权重路径(.pth)')
#     parser.add_argument('--freeze_yolo', action='store_true', help='冻结YOLO权重')
#     parser.add_argument('--freeze_pointpillar', action='store_true', help='冻结PointPillar权重')
#
#     parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
#     parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
#     parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
#     parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
#     parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
#     parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
#     parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
#     parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
#     parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
#                         help='set extra config keys if needed')
#
#     parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
#     parser.add_argument('--start_epoch', type=int, default=0, help='')
#     parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
#     parser.add_argument('--save_to_file', action='store_true', default=False, help='')
#
#     parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
#     parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
#     parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
#     parser.add_argument('--wo_gpu_stat', action='store_true', help='')
#     parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
#     # 🔥 添加实验运行相关参数
#     parser.add_argument('--run_id', type=int, default=0, help='experiment run ID for different seeds')
#     parser.add_argument('--ablation_mode', type=str, default='full',
#                         choices=['full', 'no_cross_attn', 'no_self_attn', 'no_point_aug', 'no_image_aug',
#                                  'baseline_only', 'base_no_point_aug', 'no_self_cross_attn'],
#                         help='ablation experiment mode')
#
#     args = parser.parse_args()
#
#     cfg_from_yaml_file(args.cfg_file, cfg)
#     cfg.TAG = Path(args.cfg_file).stem
#     cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
#
#     args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)
#
#     # 🔥 根据消融模式修改配置
#     if args.ablation_mode == 'base_no_point_aug':
#         cfg.ENABLE_MULTIMODAL_FUSION = False
#         cfg.MODEL.ENABLE_MULTIMODAL_FUSION = False
#         cfg.DATA_CONFIG.ENABLE_MULTIMODAL = False
#         cfg.DATA_CONFIG.ABLATION_CONFIG.USE_POINT_CLOUD_AUG = False
#         # cfg.TAG += '_base_no_point_aug'
#     elif args.ablation_mode == 'no_cross_attn':
#         cfg.MODEL.ABLATION_CONFIG.USE_CROSS_ATTENTION = False
#         # cfg.TAG += '_no_cross_attn'
#     elif args.ablation_mode == 'no_self_attn':
#         # cfg.ABLATION_CONFIG.USE_SELF_ATTENTION = False
#         cfg.MODEL.ABLATION_CONFIG.USE_SELF_ATTENTION = False
#         # cfg.TAG += '_no_self_attn'
#     elif args.ablation_mode == 'no_point_aug':
#         # cfg.ABLATION_CONFIG.USE_POINT_CLOUD_AUG = False
#         cfg.DATA_CONFIG.ABLATION_CONFIG.USE_POINT_CLOUD_AUG = False
#         # cfg.TAG += '_no_point_aug'
#     elif args.ablation_mode == 'no_image_aug':
#         # cfg.ABLATION_CONFIG.USE_IMAGE_AUG = False
#         cfg.DATA_CONFIG.ABLATION_CONFIG.USE_IMAGE_AUG = False
#         # cfg.TAG += '_no_image_aug'
#     elif args.ablation_mode == 'baseline_only':
#         # 完全禁用多模态融合
#         cfg.ENABLE_MULTIMODAL_FUSION = False
#         cfg.MODEL.ENABLE_MULTIMODAL_FUSION = False
#         cfg.DATA_CONFIG.ENABLE_MULTIMODAL = False
#         # cfg.TAG += '_baseline_only'
#     elif args.ablation_mode == 'no_self_cross_attn':
#         cfg.MODEL.ABLATION_CONFIG.USE_CROSS_ATTENTION = False
#         cfg.MODEL.ABLATION_CONFIG.USE_SELF_ATTENTION = False
#
#     if args.set_cfgs is not None:
#         cfg_from_list(args.set_cfgs, cfg)
#
#     return args, cfg
#
#
# def load_pretrained_weights(model, args, logger):
#     """加载预训练权重并冻结（如果需要）"""
#
#     # 加载PointPillar预训练权重
#     if args.pretrained_pointpillar and os.path.exists(args.pretrained_pointpillar):
#         logger.info(f"加载PointPillar预训练权重: {args.pretrained_pointpillar}")
#         checkpoint = torch.load(args.pretrained_pointpillar, map_location='cpu')
#
#         if 'model_state' in checkpoint:
#             state_dict = checkpoint['model_state']
#         else:
#             state_dict = checkpoint
#
#         # 只加载PointPillar相关权重（排除YOLO和融合模块）
#         pp_state_dict = {k: v for k, v in state_dict.items()
#                          if not any(x in k for x in ['yolo', 'fusion', 'detection_head'])}
#
#         model.load_state_dict(pp_state_dict, strict=False)
#         logger.info(f"成功加载 {len(pp_state_dict)} 个PointPillar参数")
#
#         if args.freeze_pointpillar:
#             logger.info("冻结PointPillar权重...")
#             for name, param in model.named_parameters():
#                 if not any(x in name for x in ['yolo', 'fusion', 'alpha']):
#                     param.requires_grad = False
#             logger.info("PointPillar权重已冻结")
#
#     # YOLO权重在YOLOv8FeatureExtractor初始化时加载
#     if args.pretrained_yolo:
#         logger.info(f"YOLO预训练权重路径已设置: {args.pretrained_yolo}")
#         logger.info(f"YOLO权重冻结: {args.freeze_yolo}")
#
#
# def main():
#     args, cfg = parse_config()
#     # 添加这里 - 更完整的随机种子设置
#     if args.fix_random_seed:
#
#         seed = 666 + args.run_id * 1000
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         # torch.backends.cudnn.deterministic = True
#         # torch.backends.cudnn.benchmark = False
#     if args.launcher == 'none':
#         dist_train = False
#         total_gpus = 1
#     else:
#         total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
#             args.tcp_port, args.local_rank, backend='nccl'
#         )
#         dist_train = True
#
#     if args.batch_size is None:
#         args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
#     else:
#         assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
#         args.batch_size = args.batch_size // total_gpus
#
#     args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
#
#     # if args.fix_random_seed:
#     #     common_utils.set_random_seed(666 + cfg.LOCAL_RANK)
#
#     output_dir = Path(__file__).resolve().parent.parent.parent / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
#     # output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
#     ckpt_dir = output_dir / 'ckpt'
#     output_dir.mkdir(parents=True, exist_ok=True)
#     ckpt_dir.mkdir(parents=True, exist_ok=True)
#
#     log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
#     logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
#
#     # log to file
#     logger.info('**********************Start logging**********************')
#     logger.info(f'🔬 消融实验模式: {args.ablation_mode}')
#     logger.info(f'🎯 实验运行ID: {args.run_id}')
#     gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
#     logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
#
#     if args.pretrained_yolo:
#         logger.info(f'YOLOv8预训练权重: {args.pretrained_yolo}')
#         logger.info(f'YOLO权重冻结: {args.freeze_yolo}')
#     if args.pretrained_pointpillar:
#         logger.info(f'PointPillar预训练权重: {args.pretrained_pointpillar}')
#         logger.info(f'PointPillar权重冻结: {args.freeze_pointpillar}')
#
#     if dist_train:
#         logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
#     else:
#         logger.info('Training with a single process')
#
#     for key, val in vars(args).items():
#         logger.info('{:16} {}'.format(key, val))
#     log_config_to_file(cfg, logger=logger)
#     if cfg.LOCAL_RANK == 0:
#         os.system('cp %s %s' % (args.cfg_file, output_dir))
#
#     tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
#
#     logger.info("----------- Create dataloader & network & optimizer -----------")
#
#     # new part
#     root_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'custom'
#
#     train_set, train_loader, train_sampler = build_dataloader(
#         dataset_cfg=cfg.DATA_CONFIG,
#         class_names=cfg.CLASS_NAMES,
#         batch_size=args.batch_size,
#         dist=dist_train, workers=args.workers,
#         logger=logger,
#         training=True,
#         root_path=root_path,  # 显式指定正确路径
#         merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
#         total_epochs=args.epochs,
#         seed=666 if args.fix_random_seed else None
#     )
#
#     # 将预训练权重路径传递给cfg，供YOLOv8FeatureExtractor使用
#     if args.pretrained_yolo:
#         cfg.MODEL.YOLO_CONFIG.PRETRAINED_WEIGHTS = args.pretrained_yolo
#         cfg.MODEL.YOLO_CONFIG.FREEZE_WEIGHTS = args.freeze_yolo
#
#     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
#     if args.sync_bn:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     model.cuda()
#
#     # 加载预训练权重
#     load_pretrained_weights(model, args, logger)
#
#     optimizer = build_optimizer(model, cfg.OPTIMIZATION)
#
#     # load checkpoint if it is possible
#     start_epoch = it = 0
#     last_epoch = -1
#     if args.pretrained_model is not None:
#         model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
#
#     if args.ckpt is not None:
#         it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
#         last_epoch = start_epoch + 1
#     else:
#         ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
#
#         if len(ckpt_list) > 0:
#             ckpt_list.sort(key=os.path.getmtime)
#             while len(ckpt_list) > 0:
#                 try:
#                     it, start_epoch = model.load_params_with_optimizer(
#                         ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
#                     )
#                     last_epoch = start_epoch + 1
#                     break
#                 except:
#                     ckpt_list = ckpt_list[:-1]
#
#     model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
#     if dist_train:
#         model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
#     logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
#     logger.info(model)
#
#     lr_scheduler, lr_warmup_scheduler = build_scheduler(
#         optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
#         last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
#     )
#
#     # -----------------------start training---------------------------
#     logger.info('**********************Start training %s/%s(%s)**********************'
#                 % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
#
#     train_model(
#         model,
#         optimizer,
#         train_loader,
#         model_func=model_fn_decorator(),
#         lr_scheduler=lr_scheduler,
#         optim_cfg=cfg.OPTIMIZATION,
#         start_epoch=start_epoch,
#         total_epochs=args.epochs,
#         start_iter=it,
#         rank=cfg.LOCAL_RANK,
#         tb_log=tb_log,
#         ckpt_save_dir=ckpt_dir,
#         train_sampler=train_sampler,
#         lr_warmup_scheduler=lr_warmup_scheduler,
#         ckpt_save_interval=args.ckpt_save_interval,
#         max_ckpt_save_num=args.max_ckpt_save_num,
#         merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
#         logger=logger,
#         logger_iter_interval=args.logger_iter_interval,
#         ckpt_save_time_interval=args.ckpt_save_time_interval,
#         use_logger_to_record=not args.use_tqdm_to_record,
#         show_gpu_stat=not args.wo_gpu_stat,
#         use_amp=args.use_amp,
#         cfg=cfg
#     )
#
#     if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
#         train_set.clean_shared_memory()
#
#     logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
#                 % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
#
#     logger.info('**********************Start evaluation %s/%s(%s)**********************' %
#                 (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
#     # new part
#     test_set, test_loader, sampler = build_dataloader(
#         dataset_cfg=cfg.DATA_CONFIG,
#         class_names=cfg.CLASS_NAMES,
#         batch_size=args.batch_size,
#         root_path=root_path,  # 显式指定正确路径
#         dist=dist_train, workers=args.workers, logger=logger, training=False
#     )
#     eval_output_dir = output_dir / 'eval' / 'eval_with_train'
#     eval_output_dir.mkdir(parents=True, exist_ok=True)
#     args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs
#
#     repeat_eval_ckpt(
#         model.module if dist_train else model,
#         test_loader, args, eval_output_dir, logger, ckpt_dir,
#         dist_test=dist_train
#     )
#     logger.info('**********************End evaluation %s/%s(%s)**********************' %
#                 (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
#
#
# if __name__ == '__main__':
#     main()