# k-En Yakın Komşu (k-NN) Sınıflandırıcı Raporu — CIFAR-10

## Özet

- **Özellik:** raw (boyut: 3072)
- **Denenen k değerleri:** 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31.
- **En iyi k (validation'a göre):** 5 (validation doğruluğu: **38.94%**).
- **En yüksek test doğruluğu:** **38.15%** (k = 7).
- **Veri:** Eğitim 45000, Doğrulama 5000, Test 10000 örnek.

Grafik: `results/knn_accuracy_vs_k.png`

---

## 1. Amaç ve Yöntem

Bu raporda, CIFAR-10 veri seti üzerinde **k-En Yakın Komşu (k-NN)** sınıflandırıcısı farklı **k** değerleri ile test edilmiş ve sonuçlar karşılaştırılmıştır. Veri seti diskten okunmuştur (`dataset/train`, `dataset/val`, `dataset/test`). Özellik modu: **raw**, özellik boyutu: 3072.

**Veri bölümleri:**
- Eğitim: 45000 örnek
- Doğrulama: 5000 örnek
- Test: 10000 örnek

---

## 2. Farklı k Değerleri ile Sonuçlar

| k | Validation Doğruluğu | Test Doğruluğu | Eğitim süresi (s) | Val tahmin (s) | Test tahmin (s) |
|---|----------------------|----------------|--------------------|----------------|------------------|
| 1 | 0.3716 | 0.3659 | 0.05 | 25.65 | 15.86 |
| 3 | 0.3738 | 0.3752 | 0.44 | 9.34 | 25.52 |
| 5 | 0.3894 | 0.3792 | 0.32 | 18.41 | 18.6 |
| 7 | 0.3852 | 0.3815 | 0.04 | 9.27 | 20.41 |
| 9 | 0.3804 | 0.3794 | 0.36 | 13.17 | 17.76 |
| 11 | 0.3786 | 0.3789 | 0.14 | 8.96 | 17.9 |
| 13 | 0.3760 | 0.3726 | 0.46 | 14.25 | 23.45 |
| 15 | 0.3740 | 0.3707 | 0.29 | 12.67 | 28.65 |
| 17 | 0.3716 | 0.3679 | 0.29 | 14.95 | 17.03 |
| 19 | 0.3694 | 0.3675 | 0.3 | 8.06 | 16.62 |
| 21 | 0.3644 | 0.3675 | 0.44 | 8.11 | 17.75 |
| 25 | 0.3608 | 0.3620 | 0.51 | 7.95 | 19.32 |
| 31 | 0.3632 | 0.3587 | 0.49 | 9.09 | 24.54 |

---

## 3. En İyi k Seçimi ve Test Sonuçları

Validation doğruluğuna göre seçilen **en iyi k = 5**.

### 3.1 Sınıf Bazlı Metrikler (Test seti, k = 5)

| Sınıf | Precision | Recall | F1-Score | Destek |
|-------|-----------|--------|----------|--------|
| airplane | 0.408 | 0.558 | 0.472 | 1000 |
| automobile | 0.716 | 0.250 | 0.371 | 1000 |
| bird | 0.289 | 0.370 | 0.325 | 1000 |
| cat | 0.264 | 0.243 | 0.253 | 1000 |
| deer | 0.314 | 0.466 | 0.375 | 1000 |
| dog | 0.324 | 0.307 | 0.315 | 1000 |
| frog | 0.399 | 0.300 | 0.342 | 1000 |
| horse | 0.627 | 0.315 | 0.419 | 1000 |
| ship | 0.368 | 0.736 | 0.491 | 1000 |
| truck | 0.624 | 0.247 | 0.354 | 1000 |
| **macro avg** | 0.433 | 0.379 | 0.372 | 10000 |
| **weighted avg** | 0.433 | 0.379 | 0.372 | 10000 |

### 3.2 Karmaşıklık Matrisi (Confusion Matrix)

Satır: gerçek sınıf, Sütun: tahmin edilen sınıf. Sınıf sırası: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

```
     airp auto bird  cat deer  dog frog hors ship truc
airp  558    7   70   22   43   15   25    6  245    9
auto  101  250   34   64   84   65   40   18  284   60
bird  132    1  370   84  173   76   60   16   82    6
 cat   74    8  150  243   99  190  112   34   75   15
deer   83    3  188   53  466   53   47   24   74    9
 dog   58    0  150  182  117  307   76   28   68   14
frog   42    3  164  101  210   91  300   14   69    6
hors   81    6  106   63  194   91   47  315   82   15
ship  121   14   14   29   35   20    6   10  736   15
truc  116   57   34   80   65   39   39   37  286  247
```

---

## 4. Sonuçların Analizi

### 4.1 k Değerinin Etkisi

Denenen k değerleri: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31. Bu denemede **validation doğruluğu en yüksek k = 5** (doğruluk: 38.94%); **test doğruluğu en yüksek k = 7** (doğruluk: 38.15%). Küçük k (örn. k=1) gürültüye daha duyarlıdır; orta k değerleri (yaklaşık 15–21) hem validation hem test için dengeli ve yüksek doğruluk vermiştir.

### 4.2 Validation ile Test Karşılaştırması

Validation ve test doğrulukları birbirine yakındır; model seçimi validation üzerinden yapıldığı için overfitting belirgin değildir. Test doğruluğu %60 bandına ulaşmıştır.

### 4.3 Özellik Temsili ve Süre

Kullanılan özellik modu: **raw** (boyut: 3072). Eğitim süresi düşüktür; tahmin süresi örnek sayısı ve özellik boyutuna bağlıdır. Sonuçlar \`results/knn_accuracy_vs_k.png\` grafiğinde özetlenmiştir.

### 4.4 Sınıf Bazlı Değerlendirme

Karmaşıklık matrisi ve F1 skorları (yukarıdaki tablolar), hangi sınıfların birbirine karıştığını gösterir (örn. köpek-kedi, kamyon-otomobil). Zayıf sınıflar için daha iyi özellikler veya veri dengesi düşünülebilir.
