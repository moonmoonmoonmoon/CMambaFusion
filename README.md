# PALIN: Pixel-Aligned LiDAR–NIR Multimodal 3D Object Detection with Benchmark

> **Paper under review at NeurIPS 2026 (Anonymous Submission)**

This repository contains the code and configuration for **PALIN** and **CMambaFusion**, a multimodal 3D object detection framework that fuses **LiDAR point clouds** with **Near-Infrared (NIR) imagery** captured by the Ouster OS1-128 sensor. The framework is built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and targets intelligent transportation systems (ITS) applications.

---

## Architecture Overview

```
NIR Image (128×1024)
    └──► YOLOv8-S Backbone ──► Multi-scale NIR Features [P3, P4, P5]
                                          │
                                          ▼
LiDAR Point Cloud                  CMambaFusion (×3 scales)
    └──► PointPillars               ┌──────────────────────────────┐
         (PillarVFE + Scatter       │ Stage 1: 4-Dir Cross-Mamba   │
          + 2D CNN Backbone)  ──►   │   (CrossMambaFusionBlock)    │  ──► Detection Head
         Multi-scale BEV Features   │ Stage 2: Gated Concat-Mamba  │      (AnchorHeadSingle)
         [S0, S1, S2]               │   (ConcatMambaFusionBlock)   │
                                    └──────────────────────────────┘
```

**CMambaFusion** consists of two sequential stages applied independently at each of 3 feature scales:
1. **4-Dir Cross-Mamba** (`CrossMambaFusionBlock` in `vmamba.py`): Four-directional 2D selective scanning with C-matrix swapping between LiDAR and NIR modalities for bidirectional cross-modal interaction.
2. **Gated Concat-Mamba** (`ConcatMambaFusionBlock` in `vmamba.py`, `use_gate=True`): Concatenates cross-modally enhanced features and refines them through a single-stream SSM with a difference-based sigmoid gate (`Fout = X0 + g ⊙ Y`).

Key source files:
- `third_party/fusion/attention/fusion_module.py` — `MultiModalFusionForPointPillars`, `SmartPaddingFusionModule`
- `third_party/fusion/wrappers/yolo_extractor.py` — `YOLOv8FeatureExtractor`
- `third_party/mamba/vmamba.py` — `CrossMambaFusionBlock`, `ConcatMambaFusionBlock`
- `third_party/pcdet/models/detectors/pointpillar.py` — Modified PointPillars with multimodal fusion
- `third_party/tools/cfgs/custom_models/pointpillar.yaml` — Model and training config
- `third_party/tools/cfgs/dataset_configs/custom_dataset.yaml` — Dataset config

---

## Dataset

The **PALIN dataset** is collected with two Ouster OS1-128 sensors, providing intrinsically pixel-aligned LiDAR point clouds and NIR images without cross-sensor calibration. It covers:
- **Bus-mounted (Amherst, MA)**: 6,585 annotated frames across 5 bus routes
- **Roadside static (Boston, MA)**: 2,002 annotated frames at a signalized intersection
- **Total**: 8,587 annotated frames, 11 object categories

> Dataset download: *[to be released upon acceptance]*

The dataset is split into two directories:
- **`data/Bus/`** — NIR images and 2D labels
- **`data/custom/`** — LiDAR point clouds (`.bin`) and 3D annotations

```
data/
├── Bus/
│   ├── train/
│   │   ├── images/         # NIR image files
│   │   └── labels/         # 2D label files
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── custom/
    ├── ImageSets/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    ├── training/
    │   ├── data/           # .bin LiDAR point cloud files
    │   └── label/          # 3D annotation files
    └── testing/
        ├── data/
        └── label/
```

---

## Environment Setup

### Requirements
- Ubuntu 18.04 / 20.04
- Python 3.8
- CUDA 11.x
- PyTorch 1.13.1

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/moonmoonmoonmoon/CMambaFusion.git
cd CMambaFusion
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Install OpenPCDet**
```bash
cd third_party/pcdet
pip install -e .
cd ../..
```

**4. Build Mamba CUDA extensions**
```bash
cd third_party/mamba/selective_scan
pip install -e .
cd ../../..
```

**5. Install YOLOv8 (Ultralytics)**
```bash
cd third_party/ultralytics
pip install -e .
cd ../..
```

---

## Step 0: Pre-train YOLOv8-S on NIR Images (Multimodal only)

Before multimodal training, first train a YOLOv8-S model on the PALIN NIR images in a **separate Ultralytics project**:

```bash
cd /path/to/ultralytics_part

python run_lidar_detection.py \
    --data_yaml data/Bus/data.yaml \
    --mode train \
    --model_size s \
    --epochs 200 \
    --batch_size 16 \
    --no_pretrain
```

The trained weights (e.g. `results/models_copy/yolov8_s/weights/best.pt`) will be passed to the main training script via `--pretrained_yolo`.

---

## Step 1: Generate Dataset Infos (Required for ALL experiments)

**This must be run before any training**, regardless of whether multimodal fusion is used. It generates the GT database and `.pkl` info files used by OpenPCDet.

```bash
cd third_party
python -m pcdet.datasets.custom.custom_dataset create_custom_infos \
    tools/cfgs/dataset_configs/custom_dataset.yaml
```

This creates `custom_infos_train.pkl`, `custom_infos_val.pkl`, and `custom_dbinfos_train.pkl` under `data/custom/`.

---

## Training

All training commands are run from `third_party/tools/`.

### 1. LiDAR-only PointPillars (baseline)

In `pointpillar.yaml`, set:
```yaml
ENABLE_MULTIMODAL_FUSION: False
```
In `custom_dataset.yaml`, set:
```yaml
ENABLE_MULTIMODAL: False
```

```bash
cd third_party/tools
python train.py \
    --cfg_file ./cfgs/custom_models/pointpillar.yaml \
    --ablation_mode baseline_only \
    --extra_tag base_run_01 \
    --epochs 80
```

---

### 2. Bidirectional Cross-Attention Fusion

In `pointpillar.yaml`, set:
```yaml
ENABLE_MULTIMODAL_FUSION: True
ABLATION_CONFIG:
    USE_CROSS_ATTENTION: True
    USE_SELF_ATTENTION: False
```

In `fusion_module.py`, activate the **Cross-Attention version** of `SmartPaddingFusionModule` (uses `nn.MultiheadAttention` for `img_cross_attn` / `lidar_cross_attn`). Comment out the Mamba-based class.

```bash
python train.py \
    --cfg_file ./cfgs/custom_models/pointpillar.yaml \
    --pretrained_yolo /path/to/yolov8_s/weights/best.pt \
    --pretrained_pointpillar /path/to/pointpillar/checkpoint_epoch_100.pth \
    --freeze_yolo --freeze_pointpillar \
    --ablation_mode full \
    --extra_tag cross_attention_run_01 \
    --epochs 80
```

---

### 3. 4-Dir Cross-Mamba only (ablation)

In `pointpillar.yaml`, same as above:
```yaml
ENABLE_MULTIMODAL_FUSION: True
ABLATION_CONFIG:
    USE_CROSS_ATTENTION: True
    USE_SELF_ATTENTION: False
```

In `fusion_module.py` (inside the active Mamba `SmartPaddingFusionModule`), comment out the `concat_mamba` line in `forward()` and use a simple conv instead:
```python
# Comment this out:
# fused_feat = self.concat_mamba(img_bhwc, lidar_bhwc).permute(0, 3, 1, 2).contiguous()

# Add this instead:
concat_feat = torch.cat([enhanced_img_feat, enhanced_lidar_feat], dim=1)
fused_feat = self.fusion_conv(concat_feat)
```

```bash
python train.py \
    --cfg_file ./cfgs/custom_models/pointpillar.yaml \
    --pretrained_yolo /path/to/yolov8_s/weights/best.pt \
    --pretrained_pointpillar /path/to/pointpillar/checkpoint_epoch_100.pth \
    --freeze_yolo --freeze_pointpillar \
    --ablation_mode full \
    --extra_tag 4d_cross_mamba_run_01 \
    --epochs 80
```

---

### 4. Full CMambaFusion — 4-Dir Cross-Mamba + Gated Concat-Mamba (default)

In `pointpillar.yaml`:
```yaml
ENABLE_MULTIMODAL_FUSION: True
ABLATION_CONFIG:
    USE_CROSS_ATTENTION: True
    USE_SELF_ATTENTION: False
```

In `fusion_module.py`, confirm both stages are active (this is the default state in the repo):
```python
self.cross_mamba = CrossMambaFusionBlock(hidden_dim=dim, mlp_ratio=0.0, d_state=4)
self.concat_mamba = ConcatMambaFusionBlock(hidden_dim=128, mlp_ratio=0.0, d_state=4, use_gate=True)
```

```bash
python train.py \
    --cfg_file ./cfgs/custom_models/pointpillar.yaml \
    --pretrained_yolo /path/to/yolov8_s/weights/best.pt \
    --pretrained_pointpillar /path/to/pointpillar/checkpoint_epoch_100.pth \
    --freeze_yolo --freeze_pointpillar \
    --ablation_mode full \
    --extra_tag 4d_concat_mamba_run_01 \
    --epochs 80
```

---

## Testing & Evaluation

> Before running test or demo, update `custom_dataset.yaml`: change `val` → `test` in the `DATA_SPLIT` and `INFO_PATH` fields.

### Evaluate LiDAR-only baseline
Set `ENABLE_MULTIMODAL_FUSION: False` and `ENABLE_MULTIMODAL: False` in both yaml files, then:
```bash
cd third_party/tools
python test.py \
    --cfg_file ./cfgs/custom_models/pointpillar.yaml \
    --ckpt /path/to/checkpoint.pth \
    --batch_size 4 --workers 4 \
    --extra_tag test_final_base
```

### Evaluate Cross-Attention fusion
Set `ENABLE_MULTIMODAL_FUSION: True`, activate the cross-attention `SmartPaddingFusionModule` in `fusion_module.py`, then:
```bash
python test.py \
    --cfg_file ./cfgs/custom_models/pointpillar.yaml \
    --ckpt /path/to/checkpoint.pth \
    --batch_size 4 --workers 4 \
    --extra_tag test_final_cross_attention
```

### Evaluate 4-Dir Cross-Mamba
Set `ENABLE_MULTIMODAL_FUSION: True`, activate Cross-Mamba only (no concat_mamba), then:
```bash
python test.py \
    --cfg_file ./cfgs/custom_models/pointpillar.yaml \
    --ckpt /path/to/checkpoint.pth \
    --batch_size 4 --workers 4 \
    --extra_tag test_final_cross_mamba
```

### Evaluate Full CMambaFusion
Set `ENABLE_MULTIMODAL_FUSION: True`, both stages active, then:
```bash
python test.py \
    --cfg_file ./cfgs/custom_models/pointpillar.yaml \
    --ckpt /path/to/checkpoint.pth \
    --batch_size 4 --workers 4 \
    --extra_tag test_final_cmambafusion
```

---

## Main Results (PALIN dataset, weighted mAP40 %)

| Method | BEV@0.7 | 3D@0.7 | BEV@0.5 | 3D@0.5 |
|---|---|---|---|---|
| PointPillars (LiDAR-only) | 80.97 | 71.20 | 88.09 | 86.42 |
| + Cross-Attention | 83.54 | 71.76 | 89.61 | 87.57 |
| + 4-Dir Cross-Mamba | 84.56 | 74.21 | 90.41 | 88.30 |
| + CMambaFusion (Ours) | **84.79** | **74.97** | **90.87** | **88.90** |

---

## Repository Structure

```
PALIN_Multimodal_detection/
├── config/                              # YOLOv8 architecture config (customed_yolov8s.yaml)
├── helper/                              # Data preprocessing scripts
│   ├── first_step/                      # Data selection utilities
│   └── second_step/                     # Label extraction from JSON annotations
├── third_party/
│   ├── fusion/
│   │   ├── attention/
│   │   │   └── fusion_module.py         # MultiModalFusionForPointPillars (core)
│   │   └── wrappers/
│   │       └── yolo_extractor.py        # YOLOv8FeatureExtractor
│   ├── mamba/
│   │   └── vmamba.py                    # CrossMambaFusionBlock, ConcatMambaFusionBlock
│   ├── pcdet/                           # Modified OpenPCDet
│   │   └── models/detectors/
│   │       └── pointpillar.py           # Modified: multimodal fusion integrated
│   ├── tools/
│   │   ├── train.py                     # Training entry point
│   │   ├── test.py                      # Evaluation entry point
│   │   └── cfgs/
│   │       ├── dataset_configs/
│   │       │   └── custom_dataset.yaml  # Dataset config
│   │       └── custom_models/
│   │           └── pointpillar.yaml     # Model and training config
│   └── ultralytics/                     # YOLOv8 (Ultralytics, AGPL-3.0)
└── requirements.txt
```

---

## Acknowledgements

This project builds upon:
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache 2.0) — 3D detection framework
- [VMamba](https://github.com/MzeroMiko/VMamba) — Visual State Space Model
- [Mamba](https://github.com/state-spaces/mamba) — Selective State Space Model
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) — NIR feature extraction
- [Segments.ai](https://segments.ai) — Dataset annotation platform

---

## License

This project is released under the [Apache 2.0 License](LICENSE).  
Note: `third_party/ultralytics/` is licensed under AGPL-3.0.

---

## Contact

For questions about the code or dataset, please open a GitHub issue.
