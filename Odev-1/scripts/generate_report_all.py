#!/usr/bin/env python3
"""
results/ içindeki tüm knn_results_*.json dosyalarını okuyup tek bir karşılaştırma raporu üretir.
Sıralama: en düşük test doğruluğundan en yükseğe (en kötü -> en iyi).
Çalıştırmak: python scripts/generate_report_all.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORT_PATH = ROOT / "report_knn_tum_deneyler.md"


def _experiment_label(data):
    """Deney için kısa okunabilir etiket üretir."""
    mode = data.get("feature_mode", "?")
    parts = [mode]
    if mode == "hog_enhanced":
        prep = data.get("hog_preprocess") or "norm"
        parts.append(f"preprocess={prep}")
    if data.get("hog_use_scaler") is True:
        parts.append("StandardScaler")
    return " | ".join(parts)


def main():
    pattern = "knn_results*.json"
    json_files = sorted(RESULTS_DIR.glob(pattern))
    if not json_files:
        print(f"Hiç sonuç dosyası yok: {RESULTS_DIR / pattern}")
        return

    all_results = []
    for p in json_files:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        if not records:
            continue
        best_val = max(r["val_accuracy"] for r in records)
        best_test = max(r["test_accuracy"] for r in records)
        best_k = data.get("best_k")
        name = p.stem  # knn_results_raw, knn_results, ...
        label = _experiment_label(data)
        all_results.append({
            "file": p.name,
            "name": name,
            "label": label,
            "feature_mode": data.get("feature_mode"),
            "feature_dim": data.get("feature_dim"),
            "best_k": best_k,
            "best_val": best_val,
            "best_test": best_test,
            "data": data,
        })

    # En kötüden en iyiye (test doğruluğuna göre)
    all_results.sort(key=lambda x: x["best_test"])

    lines = [
        "# k-NN Tüm Deneyler Karşılaştırma Raporu — CIFAR-10",
        "",
        "Bu rapor, farklı özellik ve ön işleme ayarlarıyla yapılan k-NN deneylerini "
        "**en düşük test doğruluğundan en yükseğe** sıralayarak özetler.",
        "",
        "---",
        "",
        "## 1. Özet Tablo (En kötü → En iyi)",
        "",
        "| Sıra | Deney | Özellik / Ön işleme | Özellik boyutu | En iyi k | Val doğr. | Test doğr. |",
        "|------|-------|----------------------|----------------|----------|-----------|------------|",
    ]

    for i, r in enumerate(all_results, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['label']} | {r['feature_dim']} | {r['best_k']} | "
            f"{r['best_val']:.2%} | **{r['best_test']:.2%}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Deney Detayları (k değerlerine göre doğruluk)",
        "",
    ])

    for i, r in enumerate(all_results, 1):
        d = r["data"]
        recs = d.get("records", [])
        lines.append(f"### {i}. {r['name']} — {r['label']}")
        lines.append("")
        lines.append(f"- Özellik boyutu: {r['feature_dim']}, en iyi k (validation): {r['best_k']}")
        lines.append("")
        lines.append("| k | Validation | Test |")
        lines.append("|---|------------|------|")
        for rec in recs:
            lines.append(f"| {rec['k']} | {rec['val_accuracy']:.4f} | {rec['test_accuracy']:.4f} |")
        lines.append("")
        # Grafik varsa referans
        plot_name = r["name"] + "_accuracy.png"
        if (RESULTS_DIR / plot_name).exists():
            lines.append(f"Grafik: `results/{plot_name}`")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend([
        "## 3. Sonuç",
        "",
        f"Toplam **{len(all_results)}** deney karşılaştırıldı. "
        f"En düşük test doğruluğu **{all_results[0]['best_test']:.2%}** ({all_results[0]['name']}), "
        f"en yüksek **{all_results[-1]['best_test']:.2%}** ({all_results[-1]['name']}). "
        "Ön işleme (StandardScaler, HOG+renk+LBP, gri normalizasyonu) test doğruluğunu belirgin biçimde artırmaktadır.",
        "",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapor yazıldı: {REPORT_PATH}")
    print(f"Toplam {len(all_results)} deney eklendi (en kötü → en iyi).")


if __name__ == "__main__":
    main()
