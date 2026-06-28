import subprocess
import json
import numpy as np
import re
import os
from pathlib import Path


def extract_map_from_log(log_file_path):
    """从日志文件中提取mAP结果"""
    results = {}

    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取不同的mAP指标
    patterns = {
        'car_3d_ap_070_070_070': r'Car AP@0\.70, 0\.70, 0\.70:.*?3d\s+AP:([\d.]+),',
        'car_3d_ap_r40_070_070_070': r'Car AP_R40@0\.70, 0\.70, 0\.70:.*?3d\s+AP:([\d.]+),',
        'car_3d_ap_050_050_050': r'Car AP@0\.70, 0\.50, 0\.50:.*?3d\s+AP:([\d.]+),',
        'car_3d_ap_r40_050_050_050': r'Car AP_R40@0\.70, 0\.50, 0\.50:.*?3d\s+AP:([\d.]+),'
    }

    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            # 取最后一个匹配结果（训练结束后的评估结果）
            results[metric_name] = float(matches[-1])

    return results


# def run_simple_experiments(config_file, run_id=5, base_tag= 'base',experiment_type="baseline", epochs=80):
#     """运行多次实验并统计结果"""
#     all_results = []
#
#     print(f"开始运行 {run_id} 次实验...")
#     print("=" * 60)
#
#     if experiment_type == "baseline":
#         run_tag = f"baseline_{base_tag}_run_{run_id}"
#         ablation_mode = "baseline_only"
#     elif experiment_type == "fusion":
#         run_tag = f"fusion_{base_tag}_run_{run_id}"
#         ablation_mode = "full"
#     else:
#         run_tag = f"{base_tag}_{experiment_type}_run_{run_id}"
#         ablation_mode = experiment_type
#         # run_tag = f"{experiment_type}_run_{i + 1}"
#         # print(f"\n🚀 运行第 {i + 1}/{num_runs} 次实验 (Tag: {run_tag})")
#
#     # 构建训练命令
#     cmd = [
#         "python", "train.py",
#         "--cfg_file", config_file,
#         "--fix_random_seed",  # 固定随机种子
#         "--run_id", str(run_id),  # 🔥 关键：为不同运行设置不同ID
#         "--ablation_mode", ablation_mode,  # 🔥 设置消融模式
#         "--extra_tag", run_tag,
#         "--epochs", str(epochs)  # 可以根据需要调整
#     ]
#
#     try:
#
#         # # 查找日志文件
#         # log_pattern = f"**/train_*{run_tag}*.log"
#         # log_files = list(Path(".").glob(log_pattern))
#         #
#         # if not log_files:
#         #     # 尝试在output目录下查找
#         #     output_dir = Path(
#         #         "/home/yanan/Downloads/projects/cug_multimodal_3d_detection/output") / "cfgs" / "custom_models" / "pointpillar" / run_tag
#         #     # print(output_dir)
#         #     log_files = list(output_dir.glob("train_*.log"))
#         #
#         # if log_files:
#         #     log_file = log_files[0]  # 取最新的日志文件
#         #     print(f"📊 分析日志文件: {log_file}")
#         #
#         #     # 提取mAP结果
#         #     map_results = extract_map_from_log(log_file)
#         #
#         #     if map_results:
#         #         all_results.append(map_results)
#         #         print("📈 提取的mAP结果:")
#         #         for metric, value in map_results.items():
#         #             print(f"  {metric}: {value:.2f}")
#         #     else:
#         #         print("⚠️  未能从日志中提取到mAP结果")
#         # else:
#         #     print("⚠️  未找到日志文件")
#         # 运行训练
#         print("开始训练...")
#         # result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
#         result = subprocess.run(cmd)
#
#         if result.returncode == 0:
#             print("✅ 训练完成")
#             del result  # 删除subprocess对象
#
#             # 查找日志文件
#             log_pattern = f"**/train_*{run_tag}*.log"
#             log_files = list(Path(".").glob(log_pattern))
#
#             if not log_files:
#                 # 尝试在output目录下查找
#                 output_dir = Path("/home/yanan/Downloads/projects/cug_multimodal_3d_detection/output") / "cfgs" / "custom_models" / "pointpillar" / run_tag
#                 # print(output_dir)
#                 log_files = list(output_dir.glob("train_*.log"))
#
#             if log_files:
#                 log_file = log_files[0]  # 取最新的日志文件
#                 print(f"📊 分析日志文件: {log_file}")
#
#                 # 提取mAP结果
#                 map_results = extract_map_from_log(log_file)
#
#                 if map_results:
#                     all_results.append(map_results)
#                     print("📈 提取的mAP结果:")
#                     for metric, value in map_results.items():
#                         print(f"  {metric}: {value:.2f}")
#                 else:
#                     print("⚠️  未能从日志中提取到mAP结果")
#             else:
#                 print("⚠️  未找到日志文件")
#         else:
#             print(f"❌ 训练失败，返回码: {result.returncode}")
#             print(f"错误信息: {result.stderr}")
#
#     except subprocess.TimeoutExpired:
#         print("⏰ 训练超时")
#     except Exception as e:
#         print(f"❌ 运行出错: {e}")
#
#     # 在这里添加，每次循环最后
#     try:
#         import torch
#         torch.cuda.empty_cache()
#         print("🧹 显存清理完成")
#     except:
#         pass
#
#     # 计算统计结果
#     if all_results:
#         print("\n" + "=" * 60)
#         print("📊 统计结果")
#         print("=" * 60)
#
#         # 转换为方便计算的格式
#         metrics = {}
#         for result in all_results:
#             for metric_name, value in result.items():
#                 if metric_name not in metrics:
#                     metrics[metric_name] = []
#                 metrics[metric_name].append(value)
#
#         # 计算每个指标的统计信息
#         for metric_name, values in metrics.items():
#             if values:  # 确保有数据
#                 mean_val = np.mean(values)
#                 std_val = np.std(values)
#                 min_val = np.min(values)
#                 max_val = np.max(values)
#
#                 print(f"\n📈 {metric_name}:")
#                 print(f"  均值 ± 标准差: {mean_val:.2f} ± {std_val:.2f}")
#                 print(f"  范围: [{min_val:.2f}, {max_val:.2f}]")
#                 print(f"  所有值: {[f'{v:.2f}' for v in values]}")
#
#         # 保存详细结果到文件
#         # 指定固定路径
#         results_dir = "/home/yanan/Downloads/projects/cug_multimodal_3d_detection/output/cfgs/custom_models/pointpillar/results"  # 替换为您想要的路径
#         results_file = os.path.join(results_dir, f"experiment_results_{experiment_type}.json")
#
#         # 确保目录存在
#         os.makedirs(results_dir, exist_ok=True)
#         # results_file = f"experiment_results_{base_tag}.json"
#         with open(results_file, 'w') as f:
#             json.dump({
#                 'raw_results': all_results,
#                 'statistics': {
#                     metric_name: {
#                         'mean': float(np.mean(values)),
#                         'std': float(np.std(values)),
#                         'min': float(np.min(values)),
#                         'max': float(np.max(values)),
#                         'values': values
#                     } for metric_name, values in metrics.items()
#                 }
#             }, f, indent=2)
#
#         print(f"\n💾 详细结果已保存到: {results_file}")
#
#         # 重点关注的指标
#         key_metrics = ['car_3d_ap_r40_070_070_070', 'car_3d_ap_r40_050_050_050']
#         print(f"\n🎯 关键指标总结:")
#         for metric in key_metrics:
#             if metric in metrics:
#                 values = metrics[metric]
#                 mean_val = np.mean(values)
#                 std_val = np.std(values)
#                 print(f"  {metric}: {mean_val:.2f} ± {std_val:.2f}")
#
#     else:
#         print("\n❌ 没有成功提取到任何结果")
#
#     return all_results

def run_multiple_experiments(config_file, run_id=5, base_tag= 'base',experiment_type="baseline_only", epochs=80):
    """运行多次实验并统计结果"""
    all_results = []

    print(f"开始运行 {run_id} 次实验...")
    print("=" * 60)

    for i in range(run_id):
        if experiment_type == "baseline":
            run_tag = f"baseline_{base_tag}_run_{i + 1}"
            ablation_mode = "baseline_only"
        elif experiment_type == "fusion":
            run_tag = f"fusion_{base_tag}_run_{i + 1}"
            ablation_mode = "full"
        else:
            run_tag = f"{base_tag}_{experiment_type}_run_{i + 1}"
            ablation_mode = experiment_type
        # run_tag = f"{experiment_type}_run_{i + 1}"
        print(f"\n🚀 运行第 {i + 1}/{run_id} 次实验 (Tag: {run_tag})")

        # 构建训练命令
        cmd = [
            "python", "train.py",
            "--cfg_file", config_file,
            "--fix_random_seed",  # 固定随机种子
            "--run_id", str(i),  # 🔥 关键：为不同运行设置不同ID
            "--ablation_mode", ablation_mode,  # 🔥 设置消融模式
            "--extra_tag", run_tag,
            "--epochs", str(epochs)  # 可以根据需要调整
        ]

        try:
            # 运行训练
            print("开始训练...")
            # result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
            result = subprocess.run(cmd)

            if result.returncode == 0:
                print("✅ 训练完成")
                del result  # 删除subprocess对象

                # 查找日志文件
                log_pattern = f"**/train_*{run_tag}*.log"
                log_files = list(Path(".").glob(log_pattern))

                if not log_files:
                    # 尝试在output目录下查找
                    output_dir = Path("/home/yanan/Downloads/projects/cug_multimodal_3d_detection/output") / "cfgs" / "custom_models" / "pointpillar" / run_tag
                    # print(output_dir)
                    log_files = list(output_dir.glob("train_*.log"))

                if log_files:
                    log_file = log_files[0]  # 取最新的日志文件
                    print(f"📊 分析日志文件: {log_file}")

                    # 提取mAP结果
                    map_results = extract_map_from_log(log_file)

                    if map_results:
                        all_results.append(map_results)
                        print("📈 提取的mAP结果:")
                        for metric, value in map_results.items():
                            print(f"  {metric}: {value:.2f}")
                    else:
                        print("⚠️  未能从日志中提取到mAP结果")
                else:
                    print("⚠️  未找到日志文件")
            else:
                print(f"❌ 训练失败，返回码: {result.returncode}")
                print(f"错误信息: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("⏰ 训练超时")
        except Exception as e:
            print(f"❌ 运行出错: {e}")

        # 在这里添加，每次循环最后
        try:
            import torch
            torch.cuda.empty_cache()
            print("🧹 显存清理完成")
        except:
            pass

    # 计算统计结果
    if all_results:
        print("\n" + "=" * 60)
        print("📊 统计结果")
        print("=" * 60)

        # 转换为方便计算的格式
        metrics = {}
        for result in all_results:
            for metric_name, value in result.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)

        # 计算每个指标的统计信息
        for metric_name, values in metrics.items():
            if values:  # 确保有数据
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)

                print(f"\n📈 {metric_name}:")
                print(f"  均值 ± 标准差: {mean_val:.2f} ± {std_val:.2f}")
                print(f"  范围: [{min_val:.2f}, {max_val:.2f}]")
                print(f"  所有值: {[f'{v:.2f}' for v in values]}")

        # 保存详细结果到文件
        # 指定固定路径
        results_dir = "/home/yanan/Downloads/projects/cug_multimodal_3d_detection/output/cfgs/custom_models/pointpillar/results"  # 替换为您想要的路径
        results_file = os.path.join(results_dir, f"experiment_results_{base_tag}.json")

        # 确保目录存在
        os.makedirs(results_dir, exist_ok=True)
        # results_file = f"experiment_results_{base_tag}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'raw_results': all_results,
                'statistics': {
                    metric_name: {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'values': values
                    } for metric_name, values in metrics.items()
                }
            }, f, indent=2)

        print(f"\n💾 详细结果已保存到: {results_file}")

        # 重点关注的指标
        key_metrics = ['car_3d_ap_r40_070_070_070', 'car_3d_ap_r40_050_050_050']
        print(f"\n🎯 关键指标总结:")
        for metric in key_metrics:
            if metric in metrics:
                values = metrics[metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric}: {mean_val:.2f} ± {std_val:.2f}")

    else:
        print("\n❌ 没有成功提取到任何结果")

    return all_results


def analyze_existing_logs(log_dir="output", pattern="**/train_*.log"):
    """分析已有的日志文件"""
    print("🔍 分析现有日志文件...")

    log_files = list(Path(log_dir).glob(pattern))
    all_results = []

    for log_file in log_files:
        print(f"📊 分析: {log_file}")
        map_results = extract_map_from_log(log_file)
        if map_results:
            all_results.append(map_results)
            print(f"  提取到 {len(map_results)} 个指标")

    if all_results:
        print(f"\n✅ 总共分析了 {len(all_results)} 个有效日志文件")
        # 使用相同的统计逻辑
        metrics = {}
        for result in all_results:
            for metric_name, value in result.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)

        for metric_name, values in metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric_name}: {mean_val:.2f} ± {std_val:.2f}")

    return all_results


if __name__ == "__main__":
    # 示例用法
    print("多次实验统计工具")
    print("选择模式:")
    print("1. 运行新实验")
    print("2. 分析现有日志")

    choice = input("请选择 (1/2): ").strip()

    if choice == "1":
        # 配置文件路径
        config_file = "./cfgs/custom_models/pointpillar.yaml"
        num_runs = 5
        epochs = 100

        # 运行多次实验
        results = run_multiple_experiments(
            config_file=config_file,
            run_id= 5,  # 运行5次
            # base_tag=" 2025_08_23_pointpillars",
            base_tag=" 2025_08_23_use_interpolation_image_augmentation_learnable_alpha",
            experiment_type="full",
            epochs=epochs
        )

    elif choice == "2":
        # 分析现有日志
        log_directory = input("请输入日志目录路径 (默认: output): ").strip() or "output"
        results = analyze_existing_logs(log_directory)

    else:
        print("无效选择")