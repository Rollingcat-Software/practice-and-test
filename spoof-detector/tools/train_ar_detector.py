#!/usr/bin/env python3
"""
AR Filter Detector Training Script
====================================

Trains a MobileNetV3-Small classifier on labeled face crops to detect
AR filters. Exports to ONNX for CPU-optimized inference.

Requirements:
    pip install torch torchvision onnx

Usage:
    python tools/train_ar_detector.py                         # Train
    python tools/train_ar_detector.py --epochs 50 --lr 0.001  # Custom params
    python tools/train_ar_detector.py --export model.onnx     # Export only
    python tools/train_ar_detector.py --evaluate               # Evaluate

Dataset structure expected:
    data/ar_dataset/
      real/         *.jpg (224x224 face crops)
      ar_filter/    *.jpg
      snapchat/     *.jpg (merged into ar_filter)
      instagram/    *.jpg
      ...
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIR = Path(__file__).parent.parent / "data" / "ar_dataset"
MODELS_DIR = Path(__file__).parent.parent / "models"


def check_dataset():
    """Check dataset availability and print stats."""
    if not DATASET_DIR.exists():
        print("Dataset not found. Run: python tools/collect_ar_dataset.py --label real")
        return False

    real_count = len(list((DATASET_DIR / "real").glob("*.jpg"))) if (DATASET_DIR / "real").exists() else 0

    ar_count = 0
    ar_labels = ["ar_filter", "snapchat", "instagram", "tiktok", "faceapp", "obs"]
    for label in ar_labels:
        label_dir = DATASET_DIR / label
        if label_dir.exists():
            ar_count += len(list(label_dir.glob("*.jpg")))

    print(f"Dataset: {real_count} real + {ar_count} AR filter = {real_count + ar_count} total")

    if real_count < 50 or ar_count < 50:
        print(f"Need at least 50 samples per class. Collect more data first.")
        return False

    return True


def train(epochs: int = 30, lr: float = 0.0005, batch_size: int = 32):
    """Train MobileNetV3-Small on AR filter dataset."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms, models
    except ImportError:
        print("PyTorch not installed. Run: pip install torch torchvision")
        return

    if not check_dataset():
        return

    print(f"\nTraining MobileNetV3-Small (epochs={epochs}, lr={lr}, batch={batch_size})")
    print(f"GPU: {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Dataset
    import cv2
    import numpy as np

    class ARDataset(Dataset):
        def __init__(self, root: Path, transform=None):
            self.samples = []
            self.transform = transform

            # Real faces = label 0
            real_dir = root / "real"
            if real_dir.exists():
                for p in real_dir.glob("*.jpg"):
                    self.samples.append((str(p), 0))

            # AR filters = label 1
            ar_labels = ["ar_filter", "snapchat", "instagram", "tiktok", "faceapp", "obs"]
            for label in ar_labels:
                label_dir = root / label
                if label_dir.exists():
                    for p in label_dir.glob("*.jpg"):
                        self.samples.append((str(p), 1))

            np.random.shuffle(self.samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            if self.transform:
                import PIL.Image
                img = PIL.Image.fromarray(img)
                img = self.transform(img)
            else:
                img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0
            return img, label

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ARDataset(DATASET_DIR, transform=transform)
    split = int(len(dataset) * 0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)  # Binary output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(imgs).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.float().to(device)
                outputs = model(imgs).squeeze(-1)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1) * 100
        val_acc = val_correct / max(val_total, 1) * 100

        print(f"  Epoch {epoch + 1:>3d}/{epochs}: loss={train_loss / len(train_loader):.4f} "
              f"train_acc={train_acc:.1f}% val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "ar_filter_detector.pth")

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved: {MODELS_DIR / 'ar_filter_detector.pth'}")

    # Export to ONNX
    export_onnx(model, device)


def export_onnx(model=None, device=None):
    """Export trained model to ONNX."""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError:
        print("PyTorch not installed.")
        return

    MODELS_DIR.mkdir(exist_ok=True)
    onnx_path = MODELS_DIR / "ar_filter_detector.onnx"

    if model is None:
        pth_path = MODELS_DIR / "ar_filter_detector.pth"
        if not pth_path.exists():
            print(f"No trained model found at {pth_path}")
            return

        model = models.mobilenet_v3_small()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
        device = torch.device("cpu")

    model.eval()
    model = model.to(device)
    dummy = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX exported: {onnx_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="AR Filter Detector Training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--export", type=str, help="Export existing model to ONNX")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing model")
    args = parser.parse_args()

    if args.export:
        export_onnx()
    elif args.evaluate:
        check_dataset()
    else:
        train(epochs=args.epochs, lr=args.lr, batch_size=args.batch)


if __name__ == "__main__":
    main()
