# CMFADet · DroneVehicle 复现工程

> 论文：*A Cross-Modality Feature Adaptive Interaction Approach for RGB-Infrared Object Detection in Aerial Imagery*（IEEE TGRS 2026）
> 本仓库是作者代码的**可运行复现版**：修复了若干阻塞训练/评估的问题，补齐了数据转换、冒烟测试与文档。**不包含数据集与权重**。

---

## 目录

- [1. 复现目标与当前进度](#1-复现目标与当前进度)
- [2. 环境安装](#2-环境安装)
- [3. 数据准备](#3-数据准备)
- [4. 训练](#4-训练)
- [5. 评估](#5-评估)
- [6. 冒烟自检（不训练）](#6-冒烟自检不训练)
- [7. 相对上游代码的修改](#7-相对上游代码的修改)
- [8. 已知问题与注意事项](#8-已知问题与注意事项)
- [9. 目录结构](#9-目录结构)

---

## 1. 复现目标与当前进度

**论文在 DroneVehicle test 上的结果**

| 指标 | 数值 |
|---|---|
| mAP@0.5 | **83.42** |
| mAP@0.5:0.95 | **69.1** |
| Params | 3.64 M |
| GFLOPs | 5.63 |
| FPS（RTX 4080） | 108.7 |

**本工程实测（A10，单 epoch 冒烟，非最终结果）**

| 项 | 数值 | 说明 |
|---|---|---|
| 模型参数 | **3,645,571（3.65 M）** | 与论文 3.64 M 对齐，说明结构无误 |
| 训练 1 epoch | 5 分 20 秒（7.0 it/s） | batch 8 / 640，A10 |
| 1 epoch 后 val | P 0.229 / R 0.268 / **mAP50 0.225** / mAP50-95 0.124 | 只训 1 轮的正常水平，可作为"链路是否正常"的对照 |

> 300 epoch 在 A10 上约 30 小时量级；150 epoch 约 15 小时。

---

## 2. 环境安装

Python **3.10**；GPU 需支持 CUDA（A10 / 4080 / 3090 等均可）。

### 方案 A：论文口径（torch 2.1.0 + cu118）

```bash
conda create -n cmfadet python=3.10 -y && conda activate cmfadet
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-cmfadet.txt
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

### 方案 B：复用云镜像自带的 torch（本工程实测通过）

若镜像已带 `Python 3.10 + torch 2.3.1+cu121 + torchvision 0.18.1+cu121`，不必重装 torch：

```bash
python3 -m venv --system-site-packages /path/to/cmfadet-env
source /path/to/cmfadet-env/bin/activate
pip install -U pip
pip install -r requirements-cmfadet.txt
pip install mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
```

> **mmcv 是硬依赖**：`OBB_ATAH → DyDCNv2` 用到 `mmcv.ops.modulated_deform_conv.ModulatedDeformConv2d`，
> 而源码里这行 import 被 `try/except ImportError: pass` 包着——缺包时不会在 import 阶段报错，而是**建模型时**
> 抛 `NameError: ModulatedDeformConv2d`，很容易误判成代码 bug。
> 索引规律：`mmcv 2.1.0 → torch 2.0/2.1`，`mmcv 2.2.0 → torch 2.2/2.3`；目录形如
> `https://download.openmmlab.com/mmcv/dist/{cu118|cu121}/{torch2.1|torch2.2|torch2.3}/index.html`。

---

## 3. 数据准备

本仓库**不含数据**（`datasets/` 已加入 `.gitignore`）。请自行获取 DroneVehicle 原始发布版，其结构为：

```text
<raw>/train/train/{trainimg, trainlabel, trainimgr, trainlabelr}
<raw>/val/val/{valimg, vallabel, valimgr, vallabelr}
<raw>/test/test/{testimg, testlabel, testimgr, testlabelr}
```

原始图 840×712、四边各有 100 px 白边，标注为 XML `<polygon>` 四点旋转框。本工程将其转换为训练格式：

```bash
python tools/prepare_dronevehicle.py \
    --src /path/to/raw_dronevehicle \
    --dst datasets/DroneVehicle/OBBCrop \
    --border 100 --quality 95 --workers 12
```

产物（**目录名是硬约束**，见 §8）：

```text
datasets/DroneVehicle/OBBCrop/
├── images/{train,val,test}/*.jpg   # RGB，640×512（裁掉白边）
├── image /{train,val,test}/*.jpg   # IR，640×512（同一裁剪窗口，与 RGB 同名）
└── labels/{train,val,test}/*.txt   # "cls x1 y1 x2 y2 x3 y3 x4 y4"（已归一化）
```

- 类别映射：`car=0, truck=1, bus=2, van=3, freight_car=4`（原数据把 freight 拼成 `feright car`，脚本已做别名）
- 划分：train **17 990** / val 1 469 / test 8 980 对（论文口径）；转换报告写在 `prepare_report.txt`
- 若原始数据有缺图，脚本会跳过并写入 `missing_files.txt`（本工程本地环境因解压不完整，train 实际为 17 951 对）

---

## 4. 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

`train.py` 里与论文对齐的关键设置（**改参数只需改这个文件**）：

| 参数 | 值 | 说明 |
|---|---|---|
| model | `ultralytics/cfg/models/multimodal/Multi-SFEM-IRAFAB-CIF-obb-ATAH.yaml` | CMFADet（非 Baseline） |
| data | `dataset/data_DV_Multimodel.yaml` | 相对路径，指向 `datasets/DroneVehicle/OBBCrop` |
| imgsz / epochs / batch | 640 / 300 / 8 | 论文 DroneVehicle 设置 |
| optimizer | SGD，lr0 0.01，lrf 0.01，momentum 0.937，wd 5e-4 | |
| 增强 | mosaic 1.0、fliplr 0.5、close_mosaic 0（作者设定） | |
| amp | False（作者设定） | 想提速可试 True，先跑 1 epoch 对比 |
| patience | 300 | 防止默认 100 轮早停 |

产出：`runs/DV/train/CMFADet-640/`（`weights/best.pt`、`last.pt`、`results.csv`、各类曲线）

断点续训（云实例被回收后）：

```python
model = YOLO(str(ROOT / 'runs/DV/train/CMFADet-640/weights/last.pt'))
model.train(resume=True)   # 其余参数从 args.yaml 自动读回
```

---

## 5. 评估

```bash
python val.py     # 默认在 test 划分上评估
```

产出：`runs/test/DV/CMFADet-640/`

```
paper_data.txt            # Model Info + Model Metrice（每类 P/R/F1/mAP50/mAP75/mAP50-95）
PR_curve.png  P_curve.png  R_curve.png  F1_curve.png
confusion_matrix.png  confusion_matrix_normalized.png
labels/                   # 逐图预测（save_txt=True）
```

> `paper_data.txt` 里的 **GFLOPs 会显示 0.0**：`get_flops()` 内部用 3 通道 dummy 输入探测，而本模型是 6 通道，
> 异常被内部 `try/except` 吞掉。参数量、FPS、mAP 均为真实值。

---

## 6. 冒烟自检（不训练）

```bash
python tools/smoke_test.py --imgsz 640 --batch 2 --device 0
# 期望：params≈3.65 M / img (2,6,640,640) / total loss≈28 / SMOKE TEST PASSED

python tools/smoke_val.py --n 8 --imgsz 640 --batch 2 --device 0
# 随机权重跑完整验证链路（OBBValidator/ProbIoU/绘图/save_txt/指标表），mAP≈0 属正常
```

---

## 7. 相对上游代码的修改

| 文件 | 修改 | 原因 |
|---|---|---|
| `train.py` | 模型改为 CMFADet yaml、数据改为 DroneVehicle、`imgsz 512→640`、`epochs 100→300`、输出目录、`patience=300`、路径改绝对 | 上游默认是 **Baseline + OGSOD + 512 + 100ep**，直接跑复现不了论文 |
| `val.py` | 权重/数据/imgsz/输出目录；`task == 'detection'` → `in {'detection','obb'}`；每类指标按 `ap_class_index` 取 | 上游门控会让 OBB 模型**永远不生成** `paper_data.txt`；硬索引 5 类在缺类时 IndexError |
| `dataset/data_DV_Multimodel.yaml` | 改为相对 yaml 的路径 + `nc: 5` | 原文件是作者机器上的 `/home/csyu/...` |
| `ultralytics/data/utils.py:313` | `check_det_dataset()` 补上 `test_ir` | 原循环漏了它，`split='test'` 时 IR 路径不解析，评估必崩 |
| `ultralytics/utils/plotting.py:874` | `plot_labels()` 取 `boxes[:, :4]` | OBB 标签 5 列，原代码 `DataFrame(columns=4)` 抛 `ValueError`（被 `@TryExcept` 吞掉，只是没有 `labels.jpg`） |
| `requirements-cmfadet.txt` | 新增 | 上游没有 requirements；含 mmcv 安装说明与 `efficientnet-pytorch`（`attention.py` 模块级 import） |
| `tools/*.py` | 新增 | 数据转换 + 两个冒烟脚本 |
| `REPRODUCE_DroneVehicle.md` | 新增 | 更详细的复现流程与排错说明 |

**未改动**：网络结构（SFEM / IR-AFAB / CIFusion / ATAH）、损失、6 通道数据管道，全部沿用作者实现。

---

## 8. 已知问题与注意事项

1. **6 通道输入**：`BaseDataset.load_image()` 把 RGB 与 IR `np.dstack` 成 6 通道，`BaseModel._predict_once()` 里
   `torch.chunk(x, 2, dim=1)` 再拆开。因此**不能**用 3 通道图片直接 `model.predict()`。
2. **yaml 里 `from` 列的 `-4 / -3` 是自定义语义**：`-4` = IR 分支，`-3` = 融合层（`CIFusion`）。不要按官方
   Ultralytics 的 `from` 语义改这个 yaml。
3. **标签必须是 9 列 OBB**（1 类 + 8 坐标）。5 列 HBB 标签会被 `Format(return_obb=True)` 清空成空框。
4. **目录名硬约束**：RGB 目录必须叫 `images`、IR 必须叫 `image`（源码用 `path.replace('images','image')` 找 IR），
   标签必须在同级 `labels/`。
5. **不要 `pip install ultralytics`**：`train.py` 用的是本仓库内的 `ultralytics/` 目录（含自定义模块），
   装官方包会覆盖成没有 SFEM/ATAH 的版本。
6. `close_mosaic=0`、`amp=False` 都是作者原设定，改动会偏离论文条件。
7. 若更换标签内容，记得删除 `labels/*.cache`，否则会复用旧标签缓存。

---

## 9. 目录结构

```text
.
├── train.py                  # 训练入口（DroneVehicle 配置）
├── val.py                    # 评估入口（test 划分 + paper_data.txt）
├── requirements-cmfadet.txt  # 依赖与 mmcv 安装说明
├── REPRODUCE_DroneVehicle.md # 详细复现/排错文档
├── dataset/                  # 数据 yaml（data_DV_Multimodel.yaml 等）
├── datasets/                 # 【不提交】转换后的 DroneVehicle
├── runs/DV/train/            # 短程训练快照（见 §10）
├── tools/
│   ├── prepare_dronevehicle.py   # 原始数据 → YOLO-OBB
│   ├── smoke_test.py             # 建模型 + 建数据集 + 前向/loss
│   └── smoke_val.py              # 验证链路冒烟
└── ultralytics/              # 作者代码（含自定义模块），未改动结构
```

---

## 10. 附：实验记录（`runs/`）

仓库里保留了 3 次短程训练（A10，每次 10 epoch）的快照，用来证明链路可跑、并给后续长训做参照：

| 目录 | epochs | batch | 内容 |
|---|---|---|---|
| `runs/DV/train/CMFADet-640` | 10 | 8 | 首次试跑，仅保留 `args.yaml`、`labels*.jpg`、`train_batch*.jpg` |
| `runs/DV/train/CMFADet-6402` | 10 | 32 | batch=32 对照（看 batch 对吞吐/收敛的影响） |
| `runs/DV/train/CMFADet-6403` | 10 | 8 | **完整**：`weights/`（best/last，FP16+FP32）、`results.csv`、`results.png`、PR/P/R/F1 曲线、混淆矩阵、val 可视化 |

`CMFADet-6403` 第 10 轮（val 划分，`results.csv` 末行）：

```
P 0.5561 | R 0.6025 | mAP50 0.5770 | mAP50-95 0.4314
val loss: box 0.8094 / cls 0.6704 / dfl 1.7805
```

> 10 轮即到 57.7 mAP50，趋势正常；按论文 300 轮还有较大提升空间（目标 83.4）。
> 这些只是短跑记录，**不是**论文复现结果。

仓库只为这些快照放开了 `runs/**/*.pt`（`.gitignore` 里的负向规则），其它位置的 `*.pt` 仍被忽略，避免误传大权重。

---

## 致谢与许可

- 模型与核心实现来自 CMFADet 作者（Yooyoo95/CMFADet）与 Ultralytics YOLO（AGPL-3.0）。
- 本仓库仅作学术复现用途；数据集 DroneVehicle 请遵循其原始许可。
