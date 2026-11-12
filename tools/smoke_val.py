#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end validation-path smoke test (NO training).

Training runs a validation pass at the end of *every* epoch, so a crash here would
kill a 300-epoch job on the server.  This script reproduces that path on a tiny
subset, on CPU:

  1. builds a mini dataset (N images copied out of the real val split),
  2. saves a random-initialised checkpoint (.pt) from the CMFADet yaml,
  3. runs `YOLO(ckpt).val(...)` exactly like val.py does (OBBValidator, probiou,
     rotated NMS, plots, save_txt, confusion matrix),
  4. re-runs val.py's table logic (result.box.p / r / f1 / ap50 / all_ap) and
     writes paper_data.txt.

Expect mAP ~ 0 (the weights are random) -- the point is that every code path runs.

Usage (from the repository root):
    python tools/smoke_val.py --n 8 --imgsz 640 --batch 2 --device cpu
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_mini(dst: Path, n: int) -> Path:
    src = ROOT / "datasets/DroneVehicle/OBBCrop"
    for sub in ("images", "image", "labels"):
        (dst / sub / "val").mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (src / "images/val").glob("*.jpg"))[:n]
    for name in names:
        stem = Path(name).stem
        shutil.copy2(src / "images/val" / name, dst / "images/val" / name)
        shutil.copy2(src / "image/val" / name, dst / "image/val" / name)
        shutil.copy2(src / "labels/val" / f"{stem}.txt", dst / "labels/val" / f"{stem}.txt")
    yaml = dst / "mini.yaml"
    q = lambda p: str(p).replace("\\", "/")  # noqa: E731
    yaml.write_text(
        f"train: {q(dst / 'images/val')}\n"
        f"val: {q(dst / 'images/val')}\n"
        f"test: {q(dst / 'images/val')}\n"
        f"train_ir: {q(dst / 'image/val')}\n"
        f"val_ir: {q(dst / 'image/val')}\n"
        f"test_ir: {q(dst / 'image/val')}\n"
        "nc: 5\n"
        "names: {0: car, 1: truck, 2: bus, 3: van, 4: freight_car}\n",
        encoding="utf-8",
    )
    print(f"mini dataset: {n} images -> {dst}")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ultralytics/cfg/models/multimodal/Multi-SFEM-IRAFAB-CIF-obb-ATAH.yaml")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--work", default=str(Path(os.environ.get("TEMP", "/tmp")) / "cmfadet_smoke_val"))
    args = ap.parse_args()
    os.chdir(ROOT)

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    mini_yaml = make_mini(work / "mini", args.n)

    import torch
    from ultralytics import YOLO

    print("\n[1/3] build model from yaml and save a random checkpoint")
    y = YOLO(str(ROOT / args.model))
    ckpt = {
        "epoch": -1,
        "best_fitness": None,
        "model": y.model.half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {},
        "train_metrics": {},
        "train_results": {},
        "date": "smoke",
        "version": "8.3.9",
    }
    pt = work / "rand_init.pt"
    torch.save(ckpt, pt)
    print("checkpoint:", pt, f"({pt.stat().st_size/1e6:.1f} MB)")

    print("\n[2/3] run YOLO(...).val() like val.py (random weights -> mAP ~ 0)")
    model = YOLO(str(pt))
    print("loaded task:", model.task)
    t0 = time.time()
    result = model.val(data=str(mini_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
                       iou=0.7, conf=0.001, save_conf=True, save_txt=True, plots=True,
                       project=str(work / "runs"), name="smoke", verbose=True, device=args.device)
    print(f"val finished in {time.time()-t0:.1f}s -> {result.save_dir}")

    print("\n[3/3] val.py table logic on the OBB result object")
    names = list(result.names.values())
    ap_all = np.asarray(result.box.all_ap, dtype=float).reshape(-1, 10) if len(result.box.all_ap) else np.zeros((0, 10))
    f1_all = np.asarray(result.box.f1, dtype=float)
    rows = []
    for i, c in enumerate(result.box.ap_class_index):
        rows.append((names[c], float(result.box.p[i]), float(result.box.r[i]),
                     float(result.box.f1[i]), float(result.box.ap50[i]),
                     float(ap_all[i, 5]), float(result.box.ap[i])))
    summary = (float(result.results_dict["metrics/precision(B)"]),
               float(result.results_dict["metrics/recall(B)"]),
               float(f1_all.mean()) if f1_all.size else 0.0,
               float(result.results_dict["metrics/mAP50(B)"]),
               float(ap_all[:, 5].mean()) if ap_all.shape[0] else 0.0,
               float(result.results_dict["metrics/mAP50-95(B)"]))
    for r in rows:
        print("  %-14s P=%.4f R=%.4f F1=%.4f mAP50=%.4f mAP75=%.4f mAP50-95=%.4f" % r)
    print("  %-14s P=%.4f R=%.4f F1=%.4f mAP50=%.4f mAP75=%.4f mAP50-95=%.4f" % (("all",) + summary))

    out = result.save_dir / "paper_data.txt"
    out.write_text("\n".join(str(r) for r in rows) + "\n" + str(summary) + "\n", encoding="utf-8")
    print("paper_data.txt written:", out)
    print("\nfiles in save_dir:", sorted(p.name for p in Path(result.save_dir).iterdir()))
    print("\nVAL SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
