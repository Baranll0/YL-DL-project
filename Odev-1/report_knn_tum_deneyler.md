# k-NN Tüm Deneyler Karşılaştırma Raporu — CIFAR-10

Bu rapor, farklı özellik ve ön işleme ayarlarıyla yapılan k-NN deneylerini **en düşük test doğruluğundan en yükseğe** sıralayarak özetler.

---

## 1. Özet Tablo (En kötü → En iyi)

| Sıra | Deney | Özellik / Ön işleme | Özellik boyutu | En iyi k | Val doğr. | Test doğr. |
|------|-------|----------------------|----------------|----------|-----------|------------|
| 1 | knn_results_raw | raw | 3072 | 5 | 38.94% | **38.15%** |
| 2 | knn_results_normalized | normalized | 3072 | 9 | 43.84% | **44.52%** |
| 3 | knn_results_pca | pca | 99 | 9 | 46.28% | **46.16%** |
| 4 | knn_results_hog_no_scaler | hog | 324 | 17 | 50.48% | **49.59%** |
| 5 | knn_results_hog_enh_none_no_scaler | hog_enhanced | preprocess=none | 358 | 11 | 53.50% | **52.89%** |
| 6 | knn_results_hog_enh_equalize_scaler | hog_enhanced | preprocess=equalize | StandardScaler | 358 | 19 | 56.86% | **56.65%** |
| 7 | knn_results_hog_scaler | hog | StandardScaler | 324 | 21 | 56.90% | **57.19%** |
| 8 | knn_results_hog_enh_none_scaler | hog_enhanced | preprocess=none | StandardScaler | 358 | 17 | 60.14% | **60.05%** |
| 9 | knn_results_hog_enh_norm_scaler | hog_enhanced | preprocess=norm | StandardScaler | 358 | 17 | 60.14% | **60.05%** |
| 10 | knn_results | hog_enhanced | preprocess=norm | 358 | 17 | 60.50% | **60.28%** |

---

## 2. Deney Detayları (k değerlerine göre doğruluk)

### 1. knn_results_raw — raw

- Özellik boyutu: 3072, en iyi k (validation): 5

| k | Validation | Test |
|---|------------|------|
| 1 | 0.3716 | 0.3659 |
| 3 | 0.3738 | 0.3752 |
| 5 | 0.3894 | 0.3792 |
| 7 | 0.3852 | 0.3815 |
| 9 | 0.3804 | 0.3794 |
| 11 | 0.3786 | 0.3789 |
| 13 | 0.3760 | 0.3726 |
| 15 | 0.3740 | 0.3707 |
| 17 | 0.3716 | 0.3679 |
| 19 | 0.3694 | 0.3675 |
| 21 | 0.3644 | 0.3675 |
| 25 | 0.3608 | 0.3620 |
| 31 | 0.3632 | 0.3587 |


---

### 2. knn_results_normalized — normalized

- Özellik boyutu: 3072, en iyi k (validation): 9

| k | Validation | Test |
|---|------------|------|
| 1 | 0.4064 | 0.4060 |
| 3 | 0.4224 | 0.4199 |
| 5 | 0.4300 | 0.4392 |
| 7 | 0.4350 | 0.4399 |
| 9 | 0.4384 | 0.4445 |
| 11 | 0.4350 | 0.4436 |
| 13 | 0.4314 | 0.4452 |
| 15 | 0.4322 | 0.4447 |
| 17 | 0.4322 | 0.4435 |
| 19 | 0.4308 | 0.4433 |
| 21 | 0.4294 | 0.4433 |
| 25 | 0.4254 | 0.4411 |
| 31 | 0.4252 | 0.4404 |

Grafik: `results/knn_results_normalized_accuracy.png`

---

### 3. knn_results_pca — pca

- Özellik boyutu: 99, en iyi k (validation): 9

| k | Validation | Test |
|---|------------|------|
| 1 | 0.4136 | 0.4192 |
| 3 | 0.4296 | 0.4393 |
| 5 | 0.4568 | 0.4531 |
| 7 | 0.4614 | 0.4577 |
| 9 | 0.4628 | 0.4604 |
| 11 | 0.4558 | 0.4613 |
| 13 | 0.4564 | 0.4599 |
| 15 | 0.4578 | 0.4616 |
| 17 | 0.4570 | 0.4597 |
| 19 | 0.4554 | 0.4593 |
| 21 | 0.4538 | 0.4590 |
| 25 | 0.4486 | 0.4552 |
| 31 | 0.4410 | 0.4531 |

Grafik: `results/knn_results_pca_accuracy.png`

---

### 4. knn_results_hog_no_scaler — hog

- Özellik boyutu: 324, en iyi k (validation): 17

| k | Validation | Test |
|---|------------|------|
| 1 | 0.4592 | 0.4603 |
| 3 | 0.4838 | 0.4762 |
| 5 | 0.4950 | 0.4823 |
| 7 | 0.4952 | 0.4864 |
| 9 | 0.5022 | 0.4912 |
| 11 | 0.5012 | 0.4941 |
| 13 | 0.5028 | 0.4916 |
| 15 | 0.5028 | 0.4918 |
| 17 | 0.5048 | 0.4959 |
| 19 | 0.5020 | 0.4937 |
| 21 | 0.5012 | 0.4934 |
| 25 | 0.4982 | 0.4904 |
| 31 | 0.4954 | 0.4887 |

Grafik: `results/knn_results_hog_no_scaler_accuracy.png`

---

### 5. knn_results_hog_enh_none_no_scaler — hog_enhanced | preprocess=none

- Özellik boyutu: 358, en iyi k (validation): 11

| k | Validation | Test |
|---|------------|------|
| 1 | 0.4904 | 0.4823 |
| 3 | 0.5154 | 0.5044 |
| 5 | 0.5240 | 0.5126 |
| 7 | 0.5302 | 0.5245 |
| 9 | 0.5314 | 0.5289 |
| 11 | 0.5350 | 0.5275 |
| 13 | 0.5332 | 0.5237 |
| 15 | 0.5318 | 0.5239 |
| 17 | 0.5310 | 0.5235 |
| 19 | 0.5270 | 0.5217 |
| 21 | 0.5274 | 0.5224 |
| 25 | 0.5320 | 0.5226 |
| 31 | 0.5292 | 0.5163 |

Grafik: `results/knn_results_hog_enh_none_no_scaler_accuracy.png`

---

### 6. knn_results_hog_enh_equalize_scaler — hog_enhanced | preprocess=equalize | StandardScaler

- Özellik boyutu: 358, en iyi k (validation): 19

| k | Validation | Test |
|---|------------|------|
| 1 | 0.4856 | 0.4868 |
| 3 | 0.5060 | 0.5134 |
| 5 | 0.5410 | 0.5377 |
| 7 | 0.5550 | 0.5476 |
| 9 | 0.5620 | 0.5552 |
| 11 | 0.5648 | 0.5581 |
| 13 | 0.5684 | 0.5635 |
| 15 | 0.5662 | 0.5645 |
| 17 | 0.5680 | 0.5661 |
| 19 | 0.5686 | 0.5657 |
| 21 | 0.5644 | 0.5665 |
| 25 | 0.5674 | 0.5661 |
| 31 | 0.5662 | 0.5632 |

Grafik: `results/knn_results_hog_enh_equalize_scaler_accuracy.png`

---

### 7. knn_results_hog_scaler — hog | StandardScaler

- Özellik boyutu: 324, en iyi k (validation): 21

| k | Validation | Test |
|---|------------|------|
| 1 | 0.5006 | 0.4981 |
| 3 | 0.5196 | 0.5241 |
| 5 | 0.5492 | 0.5437 |
| 7 | 0.5604 | 0.5533 |
| 9 | 0.5668 | 0.5607 |
| 11 | 0.5628 | 0.5633 |
| 13 | 0.5642 | 0.5661 |
| 15 | 0.5676 | 0.5696 |
| 17 | 0.5682 | 0.5668 |
| 19 | 0.5684 | 0.5675 |
| 21 | 0.5690 | 0.5690 |
| 25 | 0.5658 | 0.5719 |
| 31 | 0.5614 | 0.5665 |

Grafik: `results/knn_results_hog_scaler_accuracy.png`

---

### 8. knn_results_hog_enh_none_scaler — hog_enhanced | preprocess=none | StandardScaler

- Özellik boyutu: 358, en iyi k (validation): 17

| k | Validation | Test |
|---|------------|------|
| 1 | 0.5300 | 0.5323 |
| 3 | 0.5580 | 0.5564 |
| 5 | 0.5782 | 0.5801 |
| 7 | 0.5890 | 0.5906 |
| 9 | 0.5926 | 0.5922 |
| 11 | 0.5962 | 0.5959 |
| 13 | 0.5962 | 0.5947 |
| 15 | 0.5988 | 0.5976 |
| 17 | 0.6014 | 0.5985 |
| 19 | 0.5932 | 0.5987 |
| 21 | 0.5948 | 0.6005 |
| 25 | 0.5998 | 0.5986 |
| 31 | 0.5924 | 0.5949 |

Grafik: `results/knn_results_hog_enh_none_scaler_accuracy.png`

---

### 9. knn_results_hog_enh_norm_scaler — hog_enhanced | preprocess=norm | StandardScaler

- Özellik boyutu: 358, en iyi k (validation): 17

| k | Validation | Test |
|---|------------|------|
| 1 | 0.5300 | 0.5323 |
| 3 | 0.5580 | 0.5564 |
| 5 | 0.5782 | 0.5801 |
| 7 | 0.5890 | 0.5906 |
| 9 | 0.5926 | 0.5922 |
| 11 | 0.5962 | 0.5959 |
| 13 | 0.5962 | 0.5947 |
| 15 | 0.5988 | 0.5976 |
| 17 | 0.6014 | 0.5985 |
| 19 | 0.5932 | 0.5987 |
| 21 | 0.5948 | 0.6005 |
| 25 | 0.5998 | 0.5986 |
| 31 | 0.5924 | 0.5949 |

Grafik: `results/knn_results_hog_enh_norm_scaler_accuracy.png`

---

### 10. knn_results — hog_enhanced | preprocess=norm

- Özellik boyutu: 358, en iyi k (validation): 17

| k | Validation | Test |
|---|------------|------|
| 1 | 0.5300 | 0.5323 |
| 3 | 0.5594 | 0.5580 |
| 5 | 0.5810 | 0.5827 |
| 7 | 0.5916 | 0.5933 |
| 9 | 0.5960 | 0.5947 |
| 11 | 0.5990 | 0.5985 |
| 13 | 0.6002 | 0.5982 |
| 15 | 0.6022 | 0.6000 |
| 17 | 0.6050 | 0.6003 |
| 19 | 0.5984 | 0.6010 |
| 21 | 0.5982 | 0.6028 |
| 25 | 0.6026 | 0.6010 |
| 31 | 0.5950 | 0.5967 |


---

## 3. Sonuç

Toplam **10** deney karşılaştırıldı. En düşük test doğruluğu **38.15%** (knn_results_raw), en yüksek **60.28%** (knn_results). Ön işleme (StandardScaler, HOG+renk+LBP, gri normalizasyonu) test doğruluğunu belirgin biçimde artırmaktadır.
