#!/usr/bin/env python3
"""
Simple Dataset Labeling Tool
==============================

Browse captured images and assign/correct labels.

Usage:
    python tools/label_tool.py                        # Label data/captures/
    python tools/label_tool.py --dir data/ar_dataset  # Label specific directory

Controls:
    1-7  — Assign category (real, static, video, mask, makeup, ar, deepfake)
    n    — Next image (skip)
    p    — Previous image
    d    — Delete current image
    q    — Quit and save
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

CATEGORIES = {
    "1": "real",
    "2": "static_image",
    "3": "video_replay",
    "4": "mask_3d",
    "5": "heavy_makeup",
    "6": "ar_filter",
    "7": "deepfake_inject",
}


def main():
    parser = argparse.ArgumentParser(description="Dataset Labeling Tool")
    parser.add_argument("--dir", type=str, default="data/captures", help="Directory to label")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    images = sorted(data_dir.glob("*.jpg"))
    if not images:
        print(f"No images found in {data_dir}")
        return

    # Load existing labels
    labels_path = data_dir / "labels.json"
    labels = {}
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    print(f"Labeling {len(images)} images in {data_dir}")
    print(f"Existing labels: {len(labels)}")
    print(f"\nKeys: 1=Real 2=Static 3=Video 4=Mask 5=Makeup 6=AR 7=Deepfake")
    print(f"       n=Next  p=Prev  d=Delete  q=Quit\n")

    idx = 0
    modified = False

    while 0 <= idx < len(images):
        img_path = images[idx]
        fname = img_path.name
        img = cv2.imread(str(img_path))

        if img is None:
            idx += 1
            continue

        # Draw info
        display = img.copy()
        h, w = display.shape[:2]

        current_label = labels.get(fname, "unlabeled")
        info = f"[{idx + 1}/{len(images)}] {fname} | Label: {current_label}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Category legend
        y = 60
        for key, cat in CATEGORIES.items():
            color = (0, 255, 0) if cat == current_label else (180, 180, 180)
            cv2.putText(display, f"{key}: {cat}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 22

        cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Label Tool", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Label Tool", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            idx += 1
        elif key == ord("p"):
            idx = max(0, idx - 1)
        elif key == ord("d"):
            print(f"  Deleted: {fname}")
            img_path.unlink()
            labels.pop(fname, None)
            images.pop(idx)
            modified = True
        elif chr(key) in CATEGORIES:
            label = CATEGORIES[chr(key)]
            labels[fname] = label
            print(f"  {fname} -> {label}")
            modified = True
            idx += 1

    cv2.destroyAllWindows()

    # Save labels
    if modified or labels:
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        labeled_count = sum(1 for v in labels.values() if v != "unlabeled")
        print(f"\nSaved {labeled_count} labels to {labels_path}")

        # Stats
        from collections import Counter
        counts = Counter(labels.values())
        print("\nLabel distribution:")
        for label, count in sorted(counts.items()):
            print(f"  {label:>15s}: {count}")


if __name__ == "__main__":
    main()
