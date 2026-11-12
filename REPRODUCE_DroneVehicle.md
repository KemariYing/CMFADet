# DroneVehicle 复现：数据准备 + 代码修补说明（CMFADet）

> 本次任务只做了三件事：**① 把 DroneVehicle 原始数据转成 CMFADet 需要的格式并落在本项目里；② 对阻塞训练的少量代码做最小修补；③ 用 CPU 做冒烟验证（不训练、不跑优化器）**。
> 训练留给 A10 服务器，本文档给的是服务器上的操作步骤。
> 网络结构（SFEM / IR-AFAB / CIFusion / ATAH）、损失、6 通道数据管道**一行未改**，沿用作者实现。

---

## 1. 一页速览

| 项目 | 位置 / 结果 |
|---|---|
| 原始数据（未处理，保持原样） | `X:\7788\下载\CMFADet源码\{train,val,test}\<split>\<split>{img,imgr,label,labelr}` |
| 处理后的数据集 | `CMFADet-main/datasets/DroneVehicle/OBBCrop/`（7.5 GB） |
| 数据可用对 | **train 17 951 / val 1 469 / test 8 980**（原始 17 990 / 1 469 / 8 980） |
| 数据转换脚本 | `CMFADet-main/tools/prepare_dronevehicle.py` |
| 数据报告 | `datasets/DroneVehicle/OBBCrop/prepare_report.txt`、`missing_files.txt` |
| 训练脚本 | `train.py`（已改 5 处，见 §3.1） |
| 评估脚本 | `val.py`（已改 5 处 + OBB 任务门控，见 §3.2） |
| 数据配置 | `dataset/data_DV_Multimodel.yaml`（已改为本项目相对路径） |
| 依赖清单（新增） | `requirements-cmfadet.txt` |
| 冒烟脚本（新增） | `tools/smoke_test.py` |

---

## 2. 数据集

### 2.1 原始数据 → 目标格式

原始（作者发布版，未处理）：

```text
X:\7788\下载\CMFADet源码\
├── train\train\{trainimg, trainlabel, trainimgr, trainlabelr}
├── val\val\{valimg, vallabel, valimgr, vallabelr}
└── test\test\{testimg, testlabel, testimgr, testlabelr}
```

- 图像 840×712，四边各有 **100 px 纯白边**（已逐像素确认：第 99 列 ≈255，第 100 列开始是内容）；
- 标注是 XML `<polygon>x1y1..x4y4`（旋转框四点），不是 YOLO txt；
- 类别实际写作 `car / truck / bus / van / feright car`（最后一个在原数据里拼错）。

目标（源码要求）：

```text
CMFADet-main/datasets/DroneVehicle/OBBCrop/
├── images\{train,val,test}\*.jpg     # RGB，640×512（裁掉 100 px 白边）
├── image\{train,val,test}\*.jpg      # IR，640×512（同一裁剪窗口，与 RGB 同名）
└── labels\{train,val,test}\*.txt     # "cls x1 y1 x2 y2 x3 y3 x4 y4"，已归一化
```

命名约束（**改错会直接报错**，源码写死的逻辑）：

| 约束 | 源码依据 |
|---|---|
| RGB 目录必须叫 `images`，IR 必须叫 `image` | `ultralytics/data/base.py:163` `self.im_files[i].replace('images','image')` |
| 标签目录必须是与 `images` 同级的 `labels` | `ultralytics/data/utils.py:44-47` `img2label_paths()` |
| RGB/IR 必须同名 | 同上（IR 路径由 RGB 路径字符串替换得到） |
| 每行 9 个数（1 类 + 8 坐标），坐标 ∈[0,1] | `verify_image_label()` 列数 >6 走多边形分支；断言 `points.max()<=1` |

### 2.2 转换脚本用法

```bash
python tools/prepare_dronevehicle.py \
    --src "X:/7788/下载/CMFADet源码" \
    --dst "datasets/DroneVehicle/OBBCrop" \
    --border 100 --quality 95 --workers 12
```

脚本行为：RGB/IR 用**同一窗口** `(100,100,740,612)` 裁剪 → 存 JPEG(q95, subsampling=0) → 解析 `<polygon>`，
四点各减 (100,100)，按 640/512 归一化并 clip 到 [0,1] → 写 9 列 txt。
类别映射：`car=0, truck=1, bus=2, van=3, freight_car=4`，并把原数据的拼写错误
`feright car`（以及零星的 `feright`、`truvk`）做了别名。

### 2.3 转换结果（`prepare_report.txt`）

```
[train] processed=17951 boxes=286026 dropped_outside=0 degenerate=2 unknown_class_boxes=1
        skipped_no_rgb=16 skipped_no_ir=23 errors=0
[train] class histogram: {'*': 1, 'bus': 10366, 'car': 246061, 'feright': 1,
                          'feright car': 8697, 'truck': 13660, 'truvk': 1, 'van': 7263}
[val]   processed=1469  boxes=22462  skipped_no_rgb=0 skipped_no_ir=0 errors=0
[test]  processed=8980  boxes=143313 skipped_no_rgb=0 skipped_no_ir=0 errors=0
```

**⚠️ 训练集少了 39 对（17990 → 17951）**：解压不完整，`trainimg` 缺 16 个 jpg、`trainimgr` 缺 23 个 jpg（都是连续编号，典型的解压中断）。缺失清单在
`datasets/DroneVehicle/OBBCrop/missing_files.txt`。

- 影响：训练集少 0.22%，对精度影响可忽略；
- 如果你要补齐：重新从原始渠道下载 `train.zip`，只解压这 39 个文件到对应目录，再重跑上面的命令即可（脚本幂等，会覆盖生成）。
- 有 1 个框的 `<name>` 是 `*`（无类别信息），已跳过；另有 2 个框是拼写错误，已通过别名保留。

### 2.4 数据配置

`dataset/data_DV_Multimodel.yaml` 已改为**相对 yaml 自身**的路径（`check_det_dataset` 会以 yaml 所在目录解析），
所以整个 `CMFADet-main/` 目录连同 `datasets/` 一起拷到服务器即可，不用改路径：

```yaml
train: ../datasets/DroneVehicle/OBBCrop/images/train
val:   ../datasets/DroneVehicle/OBBCrop/images/val
test:  ../datasets/DroneVehicle/OBBCrop/images/test
train_ir: ../datasets/DroneVehicle/OBBCrop/image/train
val_ir:   ../datasets/DroneVehicle/OBBCrop/image/val
test_ir:  ../datasets/DroneVehicle/OBBCrop/image/test
nc: 5
names: {0: car, 1: truck, 2: bus, 3: van, 4: freight_car}
```

---

## 3. 代码修补清单（最小改动）

### 3.1 `train.py`

| 项 | 改前 | 改后 | 原因 |
|---|---|---|---|
| 模型 | `Multi-Baseline-obb.yaml`（绝对路径 `/home/csyu/...`） | `ultralytics/cfg/models/multimodal/Multi-SFEM-IRAFAB-CIF-obb-ATAH.yaml` | 原来训的是 Baseline，不是 CMFADet |
| 数据 | `data_OGSOD_Multimodel.yaml` | `dataset/data_DV_Multimodel.yaml` | 原来是 OGSOD |
| `imgsz` | 512 | **640** | 论文 DroneVehicle 用 640 |
| `epochs` | 100 | **300** | 论文 300 |
| `project` / `name` | `runs/OGSOD/train` / `Multi-Baseline-obb-512` | `runs/DV/train` / `CMFADet-640` | 避免结果混进 OGSOD 目录 |
| `patience` | 未写（默认 100） | **300** | 默认 100 轮无提升就早停，跑不满 300 epoch |
| 其余 | `batch=8, close_mosaic=0, workers=4, optimizer='SGD', amp=False, cache=False` | **保持原样** | 与论文/作者设定一致 |

### 3.2 `val.py`

| 项 | 改前 | 改后 |
|---|---|---|
| `model_path` | `/home/csyu/.../OGSOD/...best.pt` | `runs/DV/train/CMFADet-640/weights/best.pt` |
| `data` | `/home/csyu/ultralytics-yolo11-main/...OGSOD...` | `dataset/data_DV_Multimodel.yaml` |
| `imgsz` | 256 | **640** |
| `project` / `name` | `runs/test/OGSOD` / `Multi-Baseline-obb-256` | `runs/test/DV` / `CMFADet-640` |
| 任务门控 | `if model.task == 'detection':` | `if model.task in {'detection', 'obb'}:` |

> 门控必须改：OBB 模型的 `model.task == 'obb'`，原代码会跳过整块表格逻辑，`paper_data.txt` 永远不会生成。
> `OBBMetrics` 内部同样有 `self.box`（`metrics.py:1843`），所以 P/R/F1/mAP50/mAP75/mAP50-95 都能正常取到。

### 3.3 其他

| 文件 | 说明 |
|---|---|
| `dataset/data_DV_Multimodel.yaml` | 路径改为相对路径 + 显式 `nc: 5`（内容见 §2.4） |
| `ultralytics/data/utils.py:313` | `check_det_dataset()` 里补上 `test_ir`：原来只解析 `train/val/train_ir/val_ir/test`，`test_ir` 会保留相对路径，导致 `split='test'` 评估时 IR 目录解析错位（训练不受影响，评估必崩）。**一处单词级修复** |
| `ultralytics/utils/plotting.py:874` | `plot_labels()` 里 `boxes[:1e6]` 改为 `boxes[:1e6, :4]`：OBB 标签是 5 列（多一个角度），原来会让 `pandas.DataFrame(columns=4)` 抛 `ValueError`（被 `@TryExcept` 吞掉，只是没有 `labels.jpg`） |
| `train.py` / `val.py` 的路径 | 改用 `Path(__file__).parent` 拼绝对路径，脚本可在任意工作目录启动 |
| `val.py` 的指标表 | 改为按 `result.box.ap_class_index` 取数（原来按 5 个类名硬索引，某类无预测时会 `IndexError`；`all_ap/f1` 在无检测时是 list，也做了归一化） |
| `requirements-cmfadet.txt` | **新增**。上游没有 `requirements.txt`（原文已核对），这里补一份可安装的清单 + 安装顺序说明；**含 `efficientnet-pytorch`**（`attention.py:6` 在模块级 import，缺了连 `import ultralytics` 都过不去） |
| `tools/prepare_dronevehicle.py` | **新增**。原始数据 → YOLO-OBB 转换（见 §2.2） |
| `tools/smoke_test.py` | **新增**。不训练的前向/损失冒烟检查（见 §5） |
| `tools/smoke_val.py` | **新增**。不训练的验证链路冒烟检查（OBBValidator/绘图/save_txt/指标表） |
| `ultralytics/**` 的其他部分 | **未改动**（网络结构、loss、dataloader、6 通道处理全部保持作者原样） |

---

## 4. 服务器（A10 / Linux）环境

Python **3.10**（源码里的 `.pyc`、`.so` 都是 cp310）。

```bash
conda create -n cmfadet python=3.10 -y && conda activate cmfadet

# 1) torch：论文用 2.1.0；A10 是 sm_86，cu118 足够
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 2) 其余依赖（文件里不锁 mmcv，避免 pip 去源码编译）
pip install -r requirements-cmfadet.txt

# 3) 关键：mmcv 必须带 ops，否则 OBB_ATAH 构建时报 NameError
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

验证（三条都过再开训）：

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d;print('mmcv ops OK')"
python tools/smoke_test.py --imgsz 640 --batch 2 --device 0     # 见 §5
```

> 如果服务器的 `$HOME` 不可写（ultralytics 要在 `~/.config/Ultralytics` 写 settings/字体缓存），
> 先 `export YOLO_CONFIG_DIR=/your/scratch/ultralytics-cfg` 再跑。

> **为什么必须装 mmcv**：`OBB_ATAH.forward()` 里的 `DyDCNv2` 用到
> `mmcv.ops.modulated_deform_conv.ModulatedDeformConv2d`，而 `block.py:61-66` 的 import 被
> `try/except ImportError: pass` 包住——缺包时不会在 import 阶段报错，而是在建模型时报
> `NameError: name 'ModulatedDeformConv2d' is not defined`，很容易误判成"代码 bug"。

---

## 5. 冒烟验证（不训练）

```bash
cd CMFADet-main
python tools/smoke_test.py --imgsz 640 --batch 2 --device 0    # 建模型 + 建数据集 + 前向/损失
python tools/smoke_val.py  --n 8 --imgsz 640 --batch 2 --device 0   # 验证链路（val.py 的路径）
```

`smoke_test.py` 依次做：
1. 打印 `ultralytics` 版本 / torch / GPU / mmcv；
2. 从 yaml 建模型（会走 6 通道的 stride 探针，等于验证 SFEM / IR-AFAB / CIFusion / OBB_ATAH 都能实例化）；
3. 解析 `data_DV_Multimodel.yaml` 并统计三个 split 的文件数；
4. 构建 train/val/test 数据集（首次会扫描全部标签并写 `labels/*.cache`）；
5. 取一个 batch，打印 `img` 形状（应为 `[B, 6, 640, 640]`）与 `bboxes` 形状（`[N, 5]`）；
6. 前向 + 计算 loss（**不 backward、不 step**），打印 `box/cls/dfl` 三项。

`smoke_val.py` 会复制 8 张 val 图做迷你数据集，用**随机初始化**权重存一个 `.pt`，然后完整跑一遍
`YOLO(pt).val(...)`（OBBValidator + ProbIoU 匹配 + 旋转 NMS + 绘图 + save_txt + 混淆矩阵 + 指标表），
mAP 必然≈0，目的是确认**每个 epoch 结束时的验证流程不会崩**。

### 本次在 3050Ti 笔记本（CPU 模式）上的实测结果

```text
ultralytics : 8.3.9
torch       : 2.1.0+cpu | cuda available: False
mmcv        : 2.1.0 (ops OK)
task=obb  params=3.65 M  stride=[8.0, 16.0, 32.0]    # 论文 3.64 M，吻合
head: OBB_ATAH
train: 17951 files | val: 1469 | test: 8980 | nc=5
img: (2, 6, 640, 640) uint8  bboxes: (N, 5)  cls: (N, 1)
total loss = 27.4781   items(box,cls,dfl) = [4.9622, 4.0958, 4.681]
                                -> SMOKE TEST PASSED

val(8 imgs, 112 instances) -> confusion_matrix(_normalized).png, val_batch*_{labels,pred}.jpg,
                              labels/, paper_data.txt 全部生成
                                -> VAL SMOKE TEST PASSED
```

> 说明：本地是 **CPU 模式**（笔记本 3050Ti 4 GB，且 Windows 版 PyPI torch 是 CPU 轮子），所以
> **没有**跑 GPU、**没有**跑训练、也没有跑完整的 8980 张 test 评估。GPU/CUDA 相关只影响速度，不影响这些代码路径。

---

## 6. 训练（服务器，**未在本地执行**）

```bash
cd CMFADet-main
CUDA_VISIBLE_DEVICES=0 python train.py
```

产出：`runs/DV/train/CMFADet-640/`

```text
args.yaml  results.csv  results.png  labels.jpg
train_batch0..2.jpg  val_batch0_labels.jpg / val_batch0_pred.jpg
confusion_matrix.png  confusion_matrix_normalized.png
PR_curve.png  P_curve.png  R_curve.png  F1_curve.png
weights/best.pt  weights/last.pt
```

- 断点续训：把 `train.py` 第 12 行的注释打开（`YOLO('runs/DV/train/CMFADet-640/weights/last.pt')`）
  并把 `# resume=True` 打开；
- 想先快速验证链路：临时把 `epochs` 改小（如 3）+ `fraction=0.05`（`fraction` 是 `default.yaml:34` 的合法参数）；
- 显存不够：`batch=4` 即可，`nbs=64` 会自动把梯度累积调到 16，等效 batch 不变。

## 7. 评估（服务器）

```bash
CUDA_VISIBLE_DEVICES=0 python val.py
```

产出：`runs/test/DV/CMFADet-640/`

```text
paper_data.txt          # Model Info + Model Metrice（含每类 P/R/F1/mAP50/mAP75/mAP50-95）
PR_curve.png  P_curve.png  R_curve.png  F1_curve.png
confusion_matrix.png  confusion_matrix_normalized.png
labels/                 # save_txt=True 的逐图预测
```

论文目标值（DroneVehicle test）：mAP@0.5 **83.42**，mAP@0.5:0.95 **69.1**，Params 3.64 M，GFLOPs 5.63。

---

## 8. 已知限制 / 需要知道的坑

| # | 事项 | 说明 |
|---|---|---|
| 1 | 训练集 39 对缺失 | 见 §2.3；脚本已自动跳过，可以用 `missing_files.txt` 补齐后重跑 |
| 2 | `paper_data.txt` 里 **GFLOPs 显示 0.0** | `model_info()`→`get_flops()` 内部用 **3 通道** dummy 输入探测，而本模型是 6 通道（`chunk(2)` 会失败），异常被 `get_flops` 的 `try/except` 吞掉返回 0.0。**不影响训练/评估**，只是这一列没意义；参数量、FPS、mAP 都是真实值 |
| 3 | 图片按 640×512 训练 | 裁边后原图就是 640×512，`LetterBox` 只做 padding 到 640×640（不再放大）。这与作者数据形态一致 |
| 4 | `amp=False`、`close_mosaic=0` | 保留作者设定；不要为了提速擅自打开 AMP |
| 5 | 不要用 Windows 训练 | 本地只做数据与冒烟；`.so` 是 Linux 的，且多进程 dataloader 在 Windows 上更易出问题 |
| 6 | `labels/*.cache` | 冒烟测试会生成缓存；**换机器后由于绝对路径变化会自动重建**，不需要手动删。如果改了标签内容，请手动删掉 `labels/*.cache` |
| 7 | `dataset/xml2txt.py` | 作者原文件，仅支持 VOC `<bndbox>`→5 列 HBB，对 DroneVehicle 的 `<polygon>` 会全部失败，**不要用**；用 `tools/prepare_dronevehicle.py` |

---

## 9. 本次任务的环境（仅本地验证用，未放进项目）

- 位置：`X:\7788\cmfadet-env`（Python 3.10.22），pip 缓存 `X:\7788\pip-cache`，临时目录 `X:\7788\tmp` —— **全部在 X 盘，未占用 C 盘**；
- `YOLO_CONFIG_DIR=X:\7788\ultralytics-cfg`（ultralytics 的 settings/字体缓存也放到 X 盘）；
- 本地 torch 是 **2.1.0+cpu**、mmcv 是 **CPU 轮子**（`dist/cpu/torch2.1`）——只用于建模型/前向/验证链路，**没有用 GPU**；服务器上要按 §4 装 CUDA 版；
- 本地只做：数据转换（CPU）、标签校验、模型构建、单 batch 前向+loss、8 张图的 val 链路；
- 本地机器是 3050Ti Laptop（4 GB），**没有**跑训练，也没有跑完整评估。
