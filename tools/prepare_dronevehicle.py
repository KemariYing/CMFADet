#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert the raw DroneVehicle release into the layout expected by CMFADet.

Raw layout (as released / as extracted from train.zip, val.zip, test.zip):

    <src>/<split>/<split>/<split>img/*.jpg      RGB  (840x712, 100 px white border)
    <src>/<split>/<split>/<split>imgr/*.jpg     IR   (840x712, 100 px white border)
    <src>/<split>/<split>/<split>label/*.xml    RGB annotations, <polygon> x1..y4
    <src>/<split>/<split>/<split>labelr/*.xml   IR  annotations (identical boxes, unused)

Output layout (what ultralytics/CMFADet needs):

    <dst>/images/<split>/<stem>.jpg    RGB, cropped to 640x512
    <dst>/image/<split>/<stem>.jpg     IR,  cropped to 640x512 (same crop window)
    <dst>/labels/<split>/<stem>.txt    "cls x1 y1 x2 y2 x3 y3 x4 y4", normalized

Notes
-----
* Only the RGB label directory is used: the dataloader derives label paths from
  the RGB image path (`img2label_paths`) and derives the IR path by replacing
  "images" -> "image" (`BaseDataset.load_image`).
* Class ids follow dataset/data_DV_Multimodel.yaml:
  car=0, truck=1, bus=2, van=3, freight_car=4.  The raw annotations spell the
  last class "feright car", which is handled by ALIASES below.
* Boxes completely outside the crop window (or degenerated to < MIN_SIZE px)
  are dropped; the counts are written to the report.
* This script never modifies the raw data.

Usage
-----
    python tools/prepare_dronevehicle.py \
        --src "X:/7788/下载/CMFADet源码" \
        --dst "datasets/DroneVehicle/OBBCrop" \
        --border 100 --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_SIZE = 2.0  # px, same threshold as v8OBBLoss target filtering (rw, rh >= 2)

# canonical class order (must match dataset/data_DV_Multimodel.yaml)
CLASSES = ["car", "truck", "bus", "van", "freight_car"]
ALIASES = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "van": "van",
    "freight car": "freight_car",
    "freight_car": "freight_car",
    "feright car": "freight_car",  # real spelling in the released annotations
    "feright_car": "freight_car",
    "feright": "freight_car",      # stray typo found in a few train annotations
    "truvk": "truck",              # stray typo found in a few train annotations
}
CLASS2ID = {name: i for i, name in enumerate(CLASSES)}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _norm_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def parse_label(xml_path: Path):
    """Return (list of 4-point polygons in original pixel coords, counter of names)."""
    polys, names = [], Counter()
    root = ET.parse(str(xml_path)).getroot()

    # image size recorded in the annotation (fallback only, actual image is used)
    size = root.find("size")
    xml_size = None
    if size is not None:
        try:
            xml_size = (int(float(size.findtext("width"))), int(float(size.findtext("height"))))
        except Exception:
            xml_size = None

    for obj in root.iter("object"):
        raw = obj.findtext("name")
        if raw is None:
            continue
        key = _norm_name(raw)
        names[key] += 1
        cls = ALIASES.get(key)
        if cls is None:
            continue

        poly = obj.find("polygon")
        pts = None
        if poly is not None:
            try:
                pts = [
                    (float(poly.findtext("x1")), float(poly.findtext("y1"))),
                    (float(poly.findtext("x2")), float(poly.findtext("y2"))),
                    (float(poly.findtext("x3")), float(poly.findtext("y3"))),
                    (float(poly.findtext("x4")), float(poly.findtext("y4"))),
                ]
            except (TypeError, ValueError):
                pts = None
        if pts is None:  # fall back to a horizontal box, if any
            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                x1 = float(box.findtext("xmin")); y1 = float(box.findtext("ymin"))
                x2 = float(box.findtext("xmax")); y2 = float(box.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        polys.append((CLASS2ID[cls], pts))
    return polys, names, xml_size


def convert_one(job):
    """Crop one image pair and write the YOLO-OBB label. Returns a dict of stats."""
    stem, rgb_path, ir_path, xml_path, dst_root, border, quality, split = job
    from PIL import Image  # local import: keeps the parent process light

    rgb_path, ir_path, xml_path, dst_root = Path(rgb_path), Path(ir_path), Path(xml_path), Path(dst_root)
    out_img = dst_root / "images" / split / f"{stem}.jpg"
    out_ir = dst_root / "image" / split / f"{stem}.jpg"
    out_lb = dst_root / "labels" / split / f"{stem}.txt"

    stats = {"written": 0, "boxes": 0, "dropped": 0, "degenerate": 0, "unknown": 0,
             "skipped_no_rgb": 0, "skipped_no_ir": 0, "error": ""}
    try:
        with Image.open(rgb_path) as im_rgb:
            w, h = im_rgb.size
            if w <= 2 * border or h <= 2 * border:
                raise ValueError(f"image too small for border={border}: {w}x{h}")
            box = (border, border, w - border, h - border)
            cw, ch = w - 2 * border, h - 2 * border
            im_rgb.convert("RGB").crop(box).save(out_img, "JPEG", quality=quality, subsampling=0)

        with Image.open(ir_path) as im_ir:
            if im_ir.size != (w, h):
                raise ValueError(f"RGB/IR size mismatch: {w}x{h} vs {im_ir.size[0]}x{im_ir.size[1]}")
            im_ir.convert("RGB").crop(box).save(out_ir, "JPEG", quality=quality, subsampling=0)

        polys, names, _ = parse_label(xml_path)
        lines = []
        for cls, pts in polys:
            xs = [p[0] - border for p in pts]
            ys = [p[1] - border for p in pts]
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            if bw < MIN_SIZE or bh < MIN_SIZE:
                stats["degenerate"] += 1
                continue
            # boxes fully outside the crop window
            if max(xs) <= 0 or max(ys) <= 0 or min(xs) >= cw or min(ys) >= ch:
                stats["dropped"] += 1
                continue
            u = [min(max(x, 0.0), float(cw - 1)) / cw for x in xs]
            v = [min(max(y, 0.0), float(ch - 1)) / ch for y in ys]
            lines.append(
                "%d " % cls
                + " ".join("%.6f %.6f" % (u[i], v[i]) for i in range(4))
            )
        out_lb.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["written"] = 1
        stats["boxes"] = len(lines)
        stats["degenerate"] = stats["degenerate"]
        stats["unknown"] = sum(v for k, v in names.items() if k not in ALIASES)
        stats["all_names"] = dict(names)
        return stats
    except Exception as exc:  # noqa: BLE001
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return stats


def find_dirs(src_root: Path, split: str):
    """Locate the four sub directories for one split (tolerates one extra nesting level)."""
    candidates = [src_root / split / split, src_root / split]
    base = next((p for p in candidates if p.is_dir()), None)
    if base is None:
        return None
    suffix = {"img": f"{split}img", "imgr": f"{split}imgr",
              "label": f"{split}label", "labelr": f"{split}labelr"}
    out = {}
    for key, name in suffix.items():
        p = base / name
        if not p.is_dir():  # tolerate an extra level, e.g. train/train/trainimg already handled
            alt = base / split / name
            p = alt if alt.is_dir() else p
        out[key] = p
    return out


def main():
    ap = argparse.ArgumentParser(description="Prepare DroneVehicle for CMFADet (YOLO-OBB, 640x512)")
    ap.add_argument("--src", required=True, help="root that contains train/ val/ test/")
    ap.add_argument("--dst", required=True, help="output root, e.g. datasets/DroneVehicle/OBBCrop")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--border", type=int, default=100, help="white border width to remove (px)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality of the cropped images")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0, help="debug: only process N samples per split")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    if not src_root.is_dir():
        sys.exit(f"--src not found: {src_root}")

    print(f"src      : {src_root}")
    print(f"dst      : {dst_root}")
    print(f"border   : {args.border}px   ->  crop size depends on the source image")
    print(f"workers  : {args.workers}")

    report_lines, missing_lines = [], []
    grand = Counter()

    for split in args.splits:
        dirs = find_dirs(src_root, split)
        if dirs is None:
            report_lines.append(f"[{split}] MISSING directory: {src_root / split}")
            continue
        for key, p in dirs.items():
            if not p.is_dir():
                report_lines.append(f"[{split}] MISSING sub dir {key}: {p}")

        rgb_files = {p.stem: p for p in dirs["img"].glob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
        ir_files = {p.stem: p for p in dirs["imgr"].glob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
        xml_files = {p.stem: p for p in dirs["label"].glob("*.xml")}

        for sub in ("images", "image", "labels"):
            (dst_root / sub / split).mkdir(parents=True, exist_ok=True)

        stems = sorted(xml_files)
        if args.limit:
            stems = stems[: args.limit]

        jobs, skipped = [], Counter()
        for stem in stems:
            rgb, ir, xml = rgb_files.get(stem), ir_files.get(stem), xml_files.get(stem)
            if rgb is None:
                skipped["no_rgb"] += 1
                missing_lines.append(f"{split}\t{stem}\tmissing RGB image")
                continue
            if ir is None:
                skipped["no_ir"] += 1
                missing_lines.append(f"{split}\t{stem}\tmissing IR image")
                continue
            jobs.append((stem, str(rgb), str(ir), str(xml), str(dst_root),
                         args.border, args.quality, split))

        print(f"\n[{split}] labels={len(xml_files)} rgb={len(rgb_files)} ir={len(ir_files)} "
              f"-> to process={len(jobs)} skipped(no_rgb={skipped['no_rgb']}, no_ir={skipped['no_ir']})")

        agg = Counter()
        names_all = Counter()
        errors = []
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(convert_one, j) for j in jobs]
            for fut in as_completed(futures):
                st = fut.result()
                done += 1
                for k in ("written", "boxes", "dropped", "degenerate", "unknown"):
                    agg[k] += st.get(k, 0)
                for k, v in st.get("all_names", {}).items():
                    names_all[k] += v
                if st.get("error"):
                    errors.append(st["error"])
                if done % 500 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}  boxes={agg['boxes']}  dropped={agg['dropped']}  "
                          f"degenerate={agg['degenerate']}  errors={len(errors)}", flush=True)

        out_counts = {sub: len(list((dst_root / sub / split).glob("*"))) for sub in ("images", "image", "labels")}
        report_lines.append(
            f"[{split}] processed={agg['written']} boxes={agg['boxes']} dropped_outside={agg['dropped']} "
            f"degenerate={agg['degenerate']} unknown_class_boxes={agg['unknown']} "
            f"skipped_no_rgb={skipped['no_rgb']} skipped_no_ir={skipped['no_ir']} errors={len(errors)} "
            f"| out files: {out_counts}"
        )
        report_lines.append(f"[{split}] annotation class histogram: {dict(sorted(names_all.items()))}")
        for e in errors[:20]:
            report_lines.append(f"[{split}] ERROR {e}")
        grand.update(agg)
        grand.update(skipped)
        if errors:
            print(f"  !! {len(errors)} errors, first: {errors[0]}")

    report = "\n".join(report_lines) + "\n"
    (dst_root).mkdir(parents=True, exist_ok=True)
    (dst_root / "prepare_report.txt").write_text(report, encoding="utf-8")
    (dst_root / "missing_files.txt").write_text("\n".join(missing_lines) + ("\n" if missing_lines else ""),
                                                encoding="utf-8")

    print("\n================ SUMMARY ================")
    print(report)
    print(f"report : {dst_root / 'prepare_report.txt'}")
    print(f"missing: {dst_root / 'missing_files.txt'} ({len(missing_lines)} entries)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        traceback.print_exc()
        sys.exit(130)
