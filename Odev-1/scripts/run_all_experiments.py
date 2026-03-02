#!/usr/bin/env python3
"""
Tüm k-NN deneylerini sırayla çalıştırır: raw, normalized, pca, hog (scaler yok/var),
hog_enhanced (preprocess: none/norm/equalize, scaler yok/var).
Sonuçlar results/knn_results_<ad>.json ve grafikler results/<ad>_accuracy.png olarak kaydedilir.
Çalıştırmak: python scripts/run_all_experiments.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_knn import run_experiments

# (kısa_ad, feature_mode, hog_preprocess, hog_add_color, hog_add_lbp, hog_use_scaler)
EXPERIMENTS = [
    ("raw", "raw", None, None, None, False),
    ("normalized", "normalized", None, None, None, True),
    ("pca", "pca", None, None, None, False),
    ("hog_no_scaler", "hog", None, None, None, False),
    ("hog_scaler", "hog", None, None, None, True),
    ("hog_enh_none_no_scaler", "hog_enhanced", "none", True, True, False),
    ("hog_enh_none_scaler", "hog_enhanced", "none", True, True, True),
    ("hog_enh_norm_no_scaler", "hog_enhanced", "norm", True, True, False),
    ("hog_enh_norm_scaler", "hog_enhanced", "norm", True, True, True),
    ("hog_enh_equalize_no_scaler", "hog_enhanced", "equalize", True, True, False),
    ("hog_enh_equalize_scaler", "hog_enhanced", "equalize", True, True, True),
]


def main():
    for i, (short_name, feat_mode, hog_prep, hog_color, hog_lbp, hog_scaler) in enumerate(EXPERIMENTS):
        out_name = f"knn_results_{short_name}"
        print("\n" + "=" * 60)
        print(f"Deney {i + 1}/{len(EXPERIMENTS)}: {short_name}")
        print("=" * 60)
        run_experiments(
            feature_mode=feat_mode,
            hog_preprocess=hog_prep,
            hog_add_color=hog_color,
            hog_add_lbp=hog_lbp,
            hog_use_scaler=hog_scaler,
            output_name=out_name,
        )
    print("\nTüm deneyler tamamlandı. Karşılaştırma raporu için:")
    print("  python3 scripts/generate_report_all.py")


if __name__ == "__main__":
    main()
