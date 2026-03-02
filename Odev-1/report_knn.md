# k-En Yakın Komşu (k-NN) Sınıflandırıcı Raporu — CIFAR-10

## Özet

- **Özellik:** hog_enhanced (boyut: 358)
- **Denenen k değerleri:** 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31.
- **En iyi k (validation'a göre):** 17 (validation doğruluğu: **60.50%**).
- **En yüksek test doğruluğu:** **60.28%** (k = 21).
- **Veri:** Eğitim 45000, Doğrulama 5000, Test 10000 örnek.

Grafik: `results/knn_accuracy_vs_k.png`

---

## 1. Amaç ve Yöntem

Bu raporda, CIFAR-10 veri seti üzerinde **k-En Yakın Komşu (k-NN)** sınıflandırıcısı farklı **k** değerleri ile test edilmiş ve sonuçlar karşılaştırılmıştır. Veri seti diskten okunmuştur (`dataset/train`, `dataset/val`, `dataset/test`). Özellik modu: **hog_enhanced**, özellik boyutu: 358.

**Veri bölümleri:**
- Eğitim: 45000 örnek
- Doğrulama: 5000 örnek
- Test: 10000 örnek

---

## 2. Farklı k Değerleri ile Sonuçlar

| k | Validation Doğruluğu | Test Doğruluğu | Eğitim süresi (s) | Val tahmin (s) | Test tahmin (s) |
|---|----------------------|----------------|--------------------|----------------|------------------|
| 1 | 0.5300 | 0.5323 | 0.01 | 6.19 | 11.66 |
| 3 | 0.5594 | 0.5580 | 0.04 | 3.17 | 7.62 |
| 5 | 0.5810 | 0.5827 | 0.02 | 4.3 | 10.43 |
| 7 | 0.5916 | 0.5933 | 0.05 | 5.0 | 11.9 |
| 9 | 0.5960 | 0.5947 | 0.01 | 4.54 | 9.26 |
| 11 | 0.5990 | 0.5985 | 0.0 | 5.26 | 10.69 |
| 13 | 0.6002 | 0.5982 | 0.01 | 4.69 | 9.74 |
| 15 | 0.6022 | 0.6000 | 0.02 | 4.17 | 11.34 |
| 17 | 0.6050 | 0.6003 | 0.04 | 4.43 | 9.26 |
| 19 | 0.5984 | 0.6010 | 0.0 | 4.9 | 9.74 |
| 21 | 0.5982 | 0.6028 | 0.03 | 4.34 | 9.63 |
| 25 | 0.6026 | 0.6010 | 0.02 | 4.4 | 9.84 |
| 31 | 0.5950 | 0.5967 | 0.0 | 4.77 | 10.78 |

---

## 3. En İyi k Seçimi ve Test Sonuçları

Validation doğruluğuna göre seçilen **en iyi k = 17**.

### 3.1 Sınıf Bazlı Metrikler (Test seti, k = 17)

| Sınıf | Precision | Recall | F1-Score | Destek |
|-------|-----------|--------|----------|--------|
| airplane | 0.626 | 0.686 | 0.655 | 1000 |
| automobile | 0.669 | 0.790 | 0.725 | 1000 |
| bird | 0.512 | 0.510 | 0.511 | 1000 |
| cat | 0.459 | 0.296 | 0.360 | 1000 |
| deer | 0.555 | 0.445 | 0.494 | 1000 |
| dog | 0.532 | 0.533 | 0.532 | 1000 |
| frog | 0.573 | 0.751 | 0.650 | 1000 |
| horse | 0.710 | 0.657 | 0.683 | 1000 |
| ship | 0.565 | 0.770 | 0.652 | 1000 |
| truck | 0.827 | 0.565 | 0.671 | 1000 |
| **macro avg** | 0.603 | 0.600 | 0.593 | 10000 |
| **weighted avg** | 0.603 | 0.600 | 0.593 | 10000 |

### 3.2 Karmaşıklık Matrisi (Confusion Matrix)

Satır: gerçek sınıf, Sütun: tahmin edilen sınıf. Sınıf sırası: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

```
     airp auto bird  cat deer  dog frog hors ship truc
airp  686   18   66   10   19    9   10    5  172    5
auto   29  790    9    6    8    3   24    6  102   23
bird   80   24  510   60   72   75  101   26   47    5
 cat   41   52  105  296   78  204  114   51   41   18
deer   57   24   95   52  445   36  175   67   36   13
 dog   21   12   85  116   60  533   70   63   26   14
frog   22   31   45   26   33   38  751   20   32    2
hors   19   20   48   49   59   78   35  657   22   13
ship   91   63   13    9    7    6    8    8  770   25
truc   49  146   20   21   21   20   22   22  114  565
```

---

## 4. Sonuçların Analizi

### 4.1 k Değerinin Etkisi

Denenen k değerleri: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31. Bu denemede **validation doğruluğu en yüksek k = 17** (doğruluk: 60.50%); **test doğruluğu en yüksek k = 21** (doğruluk: 60.28%). Küçük k (örn. k=1) gürültüye daha duyarlıdır; orta k değerleri (yaklaşık 15–21) hem validation hem test için dengeli ve yüksek doğruluk vermiştir.

### 4.2 Validation ile Test Karşılaştırması

Validation ve test doğrulukları birbirine yakındır; model seçimi validation üzerinden yapıldığı için overfitting belirgin değildir. Test doğruluğu %60 bandına ulaşmıştır.

### 4.3 Özellik Temsili ve Süre

Kullanılan özellik modu: **hog_enhanced** (boyut: 358). Eğitim süresi düşüktür; tahmin süresi örnek sayısı ve özellik boyutuna bağlıdır. Sonuçlar \`results/knn_accuracy_vs_k.png\` grafiğinde özetlenmiştir.

### 4.4 Sınıf Bazlı Değerlendirme

Karmaşıklık matrisi ve F1 skorları (yukarıdaki tablolar), hangi sınıfların birbirine karıştığını gösterir (örn. köpek-kedi, kamyon-otomobil). Zayıf sınıflar için daha iyi özellikler veya veri dengesi düşünülebilir.
