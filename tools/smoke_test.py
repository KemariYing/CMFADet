#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline smoke test for the patched CMFADet repo (NO training, NO optimiser).

It checks, in order:
  1. the local `ultralytics` package imports and reports its version,
  2. the CMFADet model yaml can be parsed / instantiated (exercises SFEM,
     IR_AFAB, CIFusion, OBB_ATAH and the 6-channel stride probe -> needs mmcv),
  3. `dataset/data_DV_Multimodel.yaml` resolves and all three splits exist,
  4. the train/val/test datasets can be built (label scan + `.cache`),
  5. one batch goes through dataset -> transforms -> collate,
  6. one forward + loss (no backward, no step) on that batch.

Usage (from the repository root):
    python tools/smoke_test.py --imgsz 640 --batch 2 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ultralytics/cfg/models/multimodal/Multi-SFEM-IRAFAB-CIF-obb-ATAH.yaml")
    ap.add_argument("--data", default="dataset/data_DV_Multimodel.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu | 0 | cuda:0")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--no-forward", action="store_true", help="skip step 6 (forward + loss)")
    args = ap.parse_args()
    os.chdir(ROOT)

    # check_det_dataset() resolves a *relative* data path against the ultralytics
    # "datasets_dir" setting, so always hand it an absolute path.
    data_path = str((ROOT / args.data).resolve()) if not os.path.isabs(args.data) else args.data

    section("1) import ultralytics")
    import torch
    import ultralytics

    print("ultralytics :", ultralytics.__version__, "->", ultralytics.__file__)
    print("torch       :", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu         :", torch.cuda.get_device_name(0))
    try:
        import mmcv
        from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d  # noqa: F401

        print("mmcv        :", mmcv.__version__, "(ops OK)")
    except Exception as exc:  # noqa: BLE001
        print("mmcv        : MISSING ->", type(exc).__name__, exc)
        print("!! OBB_ATAH will fail with NameError: ModulatedDeformConv2d")

    section("2) build model from yaml")
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    # the trainer would attach its args (an IterableSimpleNamespace) to the model in
    # set_model_attributes(); the loss reads model.args.box / .cls / .dfl, so do the same here.
    cfg = get_cfg()
    cfg.imgsz = args.imgsz
    cfg.task = "obb"

    t0 = time.time()
    yolo = YOLO(args.model)
    model = yolo.model
    model.args = cfg
    n_params = sum(p.numel() for p in model.parameters())
    print(f"task={yolo.task}  classes={getattr(model, 'nc', None)}  params={n_params/1e6:.2f} M  "
          f"stride={model.stride.tolist()}  ({time.time()-t0:.1f}s)")
    print("head:", type(model.model[-1]).__name__)

    section("3) resolve dataset yaml")
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(data_path, autodownload=False)
    for k in ("train", "val", "test", "train_ir", "val_ir", "test_ir"):
        v = data.get(k)
        n = len(list(Path(v).glob("*"))) if v and Path(v).is_dir() else "MISSING"
        print(f"{k:<9}: {v}  ({n} files)")
    print("nc =", data["nc"], "| names =", data["names"])

    section("4) build datasets (label scan, writes labels/*.cache the first time)")
    from ultralytics.data import build_yolo_dataset

    datasets = {}
    for split in args.splits:
        key, key_ir = split, f"{split}_ir"
        if not data.get(key) or not data.get(key_ir):
            print(f"[{split}] skipped (missing key)")
            continue
        t0 = time.time()
        ds = build_yolo_dataset(cfg, data[key], data[key_ir], args.batch, data,
                                mode="train" if split == "train" else "val", stride=32)
        datasets[split] = ds
        print(f"[{split}] images={len(ds)}  ({time.time()-t0:.1f}s)  labels={len(ds.labels)}")

    if not datasets:
        sys.exit("no dataset could be built")

    section("5) one batch through the dataloader")
    from ultralytics.data import build_dataloader

    split = "train" if "train" in datasets else next(iter(datasets))
    ds = datasets[split]
    loader = build_dataloader(ds, args.batch, workers=0, shuffle=False, rank=-1)
    batch = next(iter(loader))
    print("split        :", split)
    print("img          :", tuple(batch["img"].shape), batch["img"].dtype, float(batch["img"].min()), float(batch["img"].max()))
    print("bboxes       :", tuple(batch["bboxes"].shape), "| cls:", tuple(batch["cls"].shape))
    print("im_file[0]   :", batch["im_file"][0])
    print("label[0][:3] :", batch["bboxes"][:3].tolist())

    if args.no_forward:
        print("\n[skip] forward/loss disabled by --no-forward")
        return

    section("6) forward + loss (no backward, no optimiser step)")
    device = torch.device(args.device if args.device == "cpu" else f"cuda:{args.device}")
    model = model.to(device)
    model.train()
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    # same normalisation as DetectionTrainer.preprocess_batch()
    batch["img"] = batch["img"].float() / 255
    t0 = time.time()
    with torch.no_grad():
        loss, loss_items = model(batch)
    print(f"total loss = {float(loss):.4f}  items(box,cls,dfl) = "
          f"{[round(float(x), 4) for x in loss_items]}  ({time.time()-t0:.1f}s)")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
