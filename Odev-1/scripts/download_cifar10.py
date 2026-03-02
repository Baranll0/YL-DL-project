#!/usr/bin/env python3
"""
CIFAR-10 veri setini indirir ve dataset/train, dataset/val, dataset/test
yapısında sınıf bazlı klasörlere kaydeder.
Val seti resmi olmadığı için train verisinin %10'u validation olarak ayrılır.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# Proje kökü (script scripts/ içinde olduğu için bir üst dizin)
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"

# CIFAR-10 sınıf isimleri (torchvision sırası)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Validation oranı (train'in %10'u)
VAL_RATIO = 0.10
RANDOM_SEED = 42


def save_images(images, labels, split: str, indices: np.ndarray) -> None:
    """Görüntüleri split (train/val/test) altında sınıf klasörlerine kaydeder."""
    out_root = DATASET_DIR / split
    for i in tqdm(indices, desc=f"Kaydediliyor: {split}", unit="img"):
        img = images[i]
        label = labels[i]
        class_name = CLASS_NAMES[label]
        class_dir = out_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        # torchvision .data: (H, W, C) veya (C, H, W); PIL için (H, W, C) uint8
        if img.shape[0] == 3:
            arr = np.transpose(img, (1, 2, 0))
        else:
            arr = img
        pil_img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))
        path = class_dir / f"{i:06d}.png"
        pil_img.save(path)


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("CIFAR-10 indiriliyor (train)...")
    train_ds = CIFAR10(root=str(ROOT / "data_download"), train=True, download=True)
    print("CIFAR-10 indiriliyor (test)...")
    test_ds = CIFAR10(root=str(ROOT / "data_download"), train=False, download=True)

    # torchvision: data (N, 3, 32, 32) — NCHW
    train_data = np.array(train_ds.data)
    train_labels = np.array(train_ds.targets)
    test_data = np.array(test_ds.data)
    test_labels = np.array(test_ds.targets)

    rng = np.random.default_rng(RANDOM_SEED)
    n_train = len(train_data)
    indices = np.arange(n_train)
    rng.shuffle(indices)

    n_val = int(n_train * VAL_RATIO)
    n_tr = n_train - n_val
    train_idx = indices[:n_tr]
    val_idx = indices[n_tr:]

    print(f"Train: {n_tr}, Val: {n_val}, Test: {len(test_data)}")

    save_images(train_data, train_labels, "train", train_idx)
    save_images(train_data, train_labels, "val", val_idx)
    save_images(test_data, test_labels, "test", np.arange(len(test_data)))

    print("Bitti. Veri seti:", DATASET_DIR)
    print("  train/", [d.name for d in (DATASET_DIR / "train").iterdir() if d.is_dir()])
    print("  val/  ", [d.name for d in (DATASET_DIR / "val").iterdir() if d.is_dir()])
    print("  test/ ", [d.name for d in (DATASET_DIR / "test").iterdir() if d.is_dir()])


if __name__ == "__main__":
    main()
