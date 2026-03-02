#!/usr/bin/env python3
"""
k-En Yakın Komşu (k-NN) sınıflandırıcı.
Veri seti diskten okunur (dataset/train, val, test).
Farklı k değerleri denenir; sonuçlar results/ klasörüne ve rapora yazılır.

NOT - Doğruluk:
  raw/pca ile ~%25-40; daha yüksek doğruluk (%50-58) için FEATURE_MODE="hog" kullanın (HOG özellikleri).
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# sklearn cosine mesafesi büyük veride sayısal uyarı üretebiliyor; bastırıyoruz
warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning, module="sklearn")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Proje kökü
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_split, load_split_hog, load_split_hog_enhanced, DATASET_DIR

# Özellik modu: "raw" | "normalized" | "pca" | "hog" | "hog_enhanced"
#   hog          = sadece HOG (~%49)
#   hog_enhanced = HOG + renk histogramı + LBP + ön işleme + StandardScaler (~%52-58, %60'e yaklaşabilir)
FEATURE_MODE = "hog_enhanced"
PCA_VARIANCE = 0.90

# hog_enhanced ayarları
HOG_PREPROCESS = "norm"   # "none" | "norm" | "equalize"
HOG_ADD_COLOR = True      # renk histogramı
HOG_ADD_LBP = True        # LBP doku özelliği
HOG_USE_SCALER = True     # StandardScaler (mesafe için önerilir)

# k-NN ayarları (hog_enhanced için genelde distance + cosine 0.5-1 puan ekler)
KNN_WEIGHTS = "distance"  # "uniform" | "distance" (yakın komşulara daha fazla ağırlık)
KNN_METRIC = "cosine"     # "minkowski" (Öklid) | "cosine" | "manhattan"

# Denenecek k değerleri (15-21 arası genelde iyi çıkıyor)
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31]

# Alt örnekleme: None = tüm veri
TRAIN_SUBSET = None   # 45000
VAL_SUBSET = None     # 5000
TEST_SUBSET = None    # 10000


def run_experiments(
    feature_mode: Optional[str] = None,
    hog_preprocess: Optional[str] = None,
    hog_add_color: Optional[bool] = None,
    hog_add_lbp: Optional[bool] = None,
    hog_use_scaler: Optional[bool] = None,
    output_name: Optional[str] = None,
):
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Parametreleri fonksiyon çağrısı üzerinden override edilebilir yap
    feature_mode = feature_mode or FEATURE_MODE
    hog_preprocess = hog_preprocess or HOG_PREPROCESS
    hog_add_color = HOG_ADD_COLOR if hog_add_color is None else hog_add_color
    hog_add_lbp = HOG_ADD_LBP if hog_add_lbp is None else hog_add_lbp
    hog_use_scaler = HOG_USE_SCALER if hog_use_scaler is None else hog_use_scaler

    # Cosine için L2-normalize kullanacağız → metriği Öklid yapıyoruz (uyarı yok)
    _metric = KNN_METRIC
    _algorithm = "brute" if KNN_METRIC == "cosine" else "auto"

    if feature_mode == "hog":
        print("Veri seti diskten okunuyor (HOG hesaplanıyor)...")
        X_train, y_train, class_names = load_split_hog("train", max_samples=TRAIN_SUBSET)
        X_val, y_val, _ = load_split_hog("val", max_samples=VAL_SUBSET)
        X_test, y_test, _ = load_split_hog("test", max_samples=TEST_SUBSET)
        print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"  Özellik modu: hog (özellik boyutu: {X_train.shape[1]})")
        if hog_use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            print("  HOG için StandardScaler uygulandı.")
    elif feature_mode == "hog_enhanced":
        print("Veri seti diskten okunuyor (HOG + renk + LBP + ön işleme)...")
        X_train, y_train, class_names = load_split_hog_enhanced(
            "train",
            max_samples=TRAIN_SUBSET,
            add_color_hist=hog_add_color,
            add_lbp=hog_add_lbp,
            preprocess=hog_preprocess,
        )
        X_val, y_val, _ = load_split_hog_enhanced(
            "val",
            max_samples=VAL_SUBSET,
            add_color_hist=hog_add_color,
            add_lbp=hog_add_lbp,
            preprocess=hog_preprocess,
        )
        X_test, y_test, _ = load_split_hog_enhanced(
            "test",
            max_samples=TEST_SUBSET,
            add_color_hist=hog_add_color,
            add_lbp=hog_add_lbp,
            preprocess=hog_preprocess,
        )
        print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(
            f"  Özellik modu: hog_enhanced (boyut: {X_train.shape[1]}, "
            f"preprocess: {hog_preprocess}, renk: {hog_add_color}, lbp: {hog_add_lbp})"
        )
        if hog_use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            print("  StandardScaler uygulandı.")
        # Cosine: satırları L2-normalize edip Öklid kullanıyoruz (aynı komşular, uyarı yok)
        if KNN_METRIC == "cosine":
            eps = 1e-10
            X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + eps)
            X_val = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + eps)
            X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + eps)
            _metric, _algorithm = "minkowski", "auto"
            print("  Satır L2 normalizasyonu (cosine ≈ Öklid birim vektörde).")
    else:
        print("Veri seti diskten okunuyor...")
        X_train, y_train, class_names = load_split("train", max_samples=TRAIN_SUBSET)
        X_val, y_val, _ = load_split("val", max_samples=VAL_SUBSET)
        X_test, y_test, _ = load_split("test", max_samples=TEST_SUBSET)
        print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"  Özellik boyutu (ham): {X_train.shape[1]}")

        # Ön işleme: pikseli [0,1] yap
        X_train = np.asarray(X_train, dtype=np.float64) / 255.0
        X_val = np.asarray(X_val, dtype=np.float64) / 255.0
        X_test = np.asarray(X_test, dtype=np.float64) / 255.0

        if feature_mode == "raw":
            print("  Özellik modu: raw (piksel/255)")
        elif feature_mode == "pca":
            pca = PCA(n_components=PCA_VARIANCE, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            print(
                f"  Özellik modu: {feature_mode} "
                f"(PCA bileşen: {X_train.shape[1]}, varyans: {PCA_VARIANCE})"
            )
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            print(f"  Özellik modu: {feature_mode} (StandardScaler ile)")

    records = []

    for k in K_VALUES:
        print(f"\n--- k = {k} ---")
        clf = KNeighborsClassifier(
            n_neighbors=k,
            weights=KNN_WEIGHTS,
            metric=_metric,
            n_jobs=-1,
            algorithm=_algorithm,
        )

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_val_pred = clf.predict(X_val)
        val_pred_time = time.perf_counter() - t0
        val_acc = accuracy_score(y_val, y_val_pred)

        t0 = time.perf_counter()
        y_test_pred = clf.predict(X_test)
        test_pred_time = time.perf_counter() - t0
        test_acc = accuracy_score(y_test, y_test_pred)

        record = {
            "k": k,
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "fit_time_sec": round(fit_time, 2),
            "val_pred_time_sec": round(val_pred_time, 2),
            "test_pred_time_sec": round(test_pred_time, 2),
        }
        records.append(record)
        print(f"  Val accuracy:  {val_acc:.4f}  (tahmin süresi: {val_pred_time:.1f}s)")
        print(f"  Test accuracy: {test_acc:.4f}  (tahmin süresi: {test_pred_time:.1f}s)")

    # En iyi k'ya göre test için confusion matrix ve classification report
    best_record = max(records, key=lambda r: r["val_accuracy"])
    best_k = best_record["k"]
    print(f"\nEn iyi k (validation'a göre): {best_k}")

    clf_best = KNeighborsClassifier(
        n_neighbors=best_k,
        weights=KNN_WEIGHTS,
        metric=_metric,
        n_jobs=-1,
        algorithm=_algorithm,
    )
    clf_best.fit(X_train, y_train)
    y_test_pred_best = clf_best.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred_best)
    report = classification_report(
        y_test, y_test_pred_best, target_names=class_names, output_dict=True
    )

    # Sonuçları kaydet
    results = {
        "feature_mode": feature_mode,
        "pca_variance": PCA_VARIANCE if feature_mode == "pca" else None,
        "feature_dim": int(X_train.shape[1]),
        "k_values": K_VALUES,
        "records": records,
        "best_k": best_k,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "hog_preprocess": hog_preprocess if feature_mode == "hog_enhanced" else None,
        "hog_add_color": hog_add_color if feature_mode == "hog_enhanced" else None,
        "hog_add_lbp": hog_add_lbp if feature_mode == "hog_enhanced" else None,
        "hog_use_scaler": hog_use_scaler if feature_mode in {"hog", "hog_enhanced"} else None,
    }

    # Çıktı dosya adı: default "knn_results.json", override edilebilir
    base_name = output_name or "knn_results"
    if not base_name.endswith(".json"):
        base_name = base_name + ".json"
    out_json = results_dir / base_name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSonuçlar kaydedildi: {out_json}")

    # Grafik: k vs test accuracy (her deney için ayrı dosya)
    plot_stem = base_name.replace(".json", "").strip()
    plot_name = f"{plot_stem}_accuracy.png" if plot_stem != "knn_results" else "knn_accuracy_vs_k.png"
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ks = [r["k"] for r in records]
        val_accs = [r["val_accuracy"] for r in records]
        test_accs = [r["test_accuracy"] for r in records]
        ax.plot(ks, val_accs, "o-", label="Validation")
        ax.plot(ks, test_accs, "s-", label="Test")
        ax.set_xlabel("k")
        ax.set_ylabel("Doğruluk (Accuracy)")
        ax.set_title(f"k-NN: k Değerine Göre Doğruluk — {plot_stem}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(results_dir / plot_name, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Grafik kaydedildi: {results_dir / plot_name}")
    except Exception as e:
        print(f"Grafik oluşturulamadı: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CIFAR-10 üzerinde k-NN deneyleri (farklı özellik ve ön işleme ayarları ile)."
    )
    parser.add_argument(
        "--feature-mode",
        choices=["raw", "normalized", "pca", "hog", "hog_enhanced"],
        default=FEATURE_MODE,
        help="Kullanılacak özellik modu.",
    )
    parser.add_argument(
        "--hog-preprocess",
        choices=["none", "norm", "equalize"],
        default=HOG_PREPROCESS,
        help="hog_enhanced için gri seviye ön işleme.",
    )
    parser.add_argument(
        "--hog-add-color",
        type=int,
        choices=[0, 1],
        default=1 if HOG_ADD_COLOR else 0,
        help="hog_enhanced için renk histogramı ekle (1) / ekleme (0).",
    )
    parser.add_argument(
        "--hog-add-lbp",
        type=int,
        choices=[0, 1],
        default=1 if HOG_ADD_LBP else 0,
        help="hog_enhanced için LBP özelliği ekle (1) / ekleme (0).",
    )
    parser.add_argument(
        "--hog-use-scaler",
        type=int,
        choices=[0, 1],
        default=1 if HOG_USE_SCALER else 0,
        help="HOG / hog_enhanced için StandardScaler kullan (1) / kullanma (0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="knn_results",
        help="Sonuç JSON dosya adı (uzantısız, results/ klasörü içinde). Varsayılan: knn_results",
    )

    args = parser.parse_args()

    run_experiments(
        feature_mode=args.feature_mode,
        hog_preprocess=args.hog_preprocess,
        hog_add_color=bool(args.hog_add_color),
        hog_add_lbp=bool(args.hog_add_lbp),
        hog_use_scaler=bool(args.hog_use_scaler),
        output_name=args.output,
    )
