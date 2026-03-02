"""
CIFAR-10 veri setini diskten okur.
dataset/train, dataset/val, dataset/test içindeki sınıf klasörlerinden
görüntüleri yükler; düzleştirilmiş piksel veya HOG özelliği döndürür.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist

# Proje kökü
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"

# Sınıf isimleri (download_cifar10.py ile aynı sıra)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_split(split: str, dataset_dir: Path = None, max_samples: int = None):
    """
    Belirtilen split (train, val, test) için görüntüleri diskten okur.

    Parametreler
    -----------
    split : str
        "train", "val" veya "test"
    dataset_dir : Path, opsiyonel
        dataset kök dizini (varsayılan: proje/dataset)
    max_samples : int, opsiyonel
        Her sınıftan en fazla kaç örnek alınacak (hız için alt örnekleme)

    Döndürür
    --------
    X : np.ndarray, shape (n_samples, 3072)
        Görüntüler düzleştirilmiş (32*32*3), uint8 veya float
    y : np.ndarray, shape (n_samples,)
        Sınıf indeksleri 0-9
    class_names : list
        Sınıf adları
    """
    dataset_dir = dataset_dir or DATASET_DIR
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split dizini bulunamadı: {split_dir}")

    X_list = []
    y_list = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        paths = sorted(class_dir.glob("*.png"))
        if max_samples is not None:
            paths = paths[:max_samples]
        for path in paths:
            img = Image.open(path)
            arr = np.array(img)  # (32, 32, 3)
            arr_flat = arr.reshape(-1).astype(np.float32)  # (3072,)
            X_list.append(arr_flat)
            y_list.append(class_idx)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y, CLASS_NAMES.copy()


# HOG parametreleri (32x32 için uygun)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def load_split_hog(split: str, dataset_dir: Path = None, max_samples: int = None):
    """
    Split için görüntüleri diskten okuyup HOG (Histogram of Oriented Gradients)
    özellik vektörü olarak döndürür. k-NN ile daha yüksek doğruluk (~%50-58) için kullanılır.

    Döndürür: X (n_samples, n_hog_features), y, class_names
    """
    dataset_dir = dataset_dir or DATASET_DIR
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split dizini bulunamadı: {split_dir}")

    X_list = []
    y_list = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        paths = sorted(class_dir.glob("*.png"))
        if max_samples is not None:
            paths = paths[:max_samples]
        for path in paths:
            img = Image.open(path)
            arr = np.array(img)  # (32, 32, 3)
            gray = rgb2gray(arr)  # (32, 32)
            feats = hog(
                gray,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                feature_vector=True,
                block_norm="L2-Hys",
            )
            X_list.append(feats)
            y_list.append(class_idx)

    X = np.stack(X_list, axis=0).astype(np.float64)
    y = np.array(y_list, dtype=np.int32)
    return X, y, CLASS_NAMES.copy()


# Renk histogramı: kanal başına bin sayısı
COLOR_HIST_BINS = 8
# LBP parametreleri
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"  # 59 bin histogram


def _color_histogram(arr):  # (32, 32, 3) uint8
    """Görüntü için kanal başına histogram, normalize (toplam 1)."""
    h = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c].ravel(), bins=COLOR_HIST_BINS, range=(0, 256))
        hist = hist.astype(np.float64) / (hist.sum() + 1e-8)
        h.append(hist)
    return np.concatenate(h)


def _lbp_histogram(gray):  # (32, 32) uint8 önerilir (float ise uint8'e çevrilir)
    """Gri görüntü için LBP uniform histogram (59 bin)."""
    if np.issubdtype(gray.dtype, np.floating):
        gray = (np.asarray(gray, dtype=np.float64).clip(0, 1) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-8)
    return hist


def load_split_hog_enhanced(
    split: str,
    dataset_dir: Path = None,
    max_samples: int = None,
    add_color_hist: bool = True,
    add_lbp: bool = True,
    preprocess: str = "norm",
):
    """
    Gelişmiş özellikler: HOG + isteğe renk histogramı + LBP.
    preprocess: "none" | "norm" (gri için ortalama/std normalizasyon) | "equalize" (histogram eşitleme).
    Döndürür: X (n_samples, n_features), y, class_names
    """
    dataset_dir = dataset_dir or DATASET_DIR
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split dizini bulunamadı: {split_dir}")

    X_list = []
    y_list = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        paths = sorted(class_dir.glob("*.png"))
        if max_samples is not None:
            paths = paths[:max_samples]
        for path in paths:
            img = Image.open(path)
            arr = np.array(img)  # (32, 32, 3)
            gray = rgb2gray(arr).astype(np.float64)

            # Ön işleme: kontrast / parlaklık normalleştirme
            if preprocess == "norm":
                mean, std = gray.mean(), gray.std()
                gray = (gray - mean) / (std + 1e-8)
            elif preprocess == "equalize":
                gray = equalize_hist(gray)

            feats_hog = hog(
                gray,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                feature_vector=True,
                block_norm="L2-Hys",
            )
            parts = [feats_hog]
            if add_color_hist:
                parts.append(_color_histogram(arr))
            if add_lbp:
                # LBP için uint8 gri (uyarıyı önler, sonuç daha kararlı)
                gray_lbp = (rgb2gray(arr) * 255).clip(0, 255).astype(np.uint8)
                parts.append(_lbp_histogram(gray_lbp))
            feats = np.concatenate(parts).astype(np.float64)
            X_list.append(feats)
            y_list.append(class_idx)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y, CLASS_NAMES.copy()
