# k-NN ile CIFAR-10: Doğruluk ve Özellik Seçenekleri

## 1. k-NN sınıflandırıcısı doğru mu?

Evet. Ham piksel ile k-NN CIFAR-10'da literatürde **%25–35** doğruluk verir; PCA ile **%38–40**, **HOG** ile **%50–58** civarı beklenir.

## 2. Neden ham piksel düşük?

- **Yüksek boyut:** 3072 boyutta “en yakın komşu” anlamını yitirir (curse of dimensionality).
- **Ham piksel:** Öklid mesafesi görsel benzerliği iyi temsil etmez; ışık/kaydırma çok etkiler.

## 3. Özellik modları (run_knn.py)

| FEATURE_MODE | Açıklama | Beklenen doğruluk |
|--------------|----------|--------------------|
| raw | Piksel/255 | ~%25–35 |
| normalized | Piksel/255 + StandardScaler | ~%33–38 |
| pca | Piksel/255 + PCA | ~%38–40 |
| **hog** | Sadece HOG | ~%48–50 |
| **hog_enhanced** | HOG + renk histogramı + LBP + ön işleme (norm) + StandardScaler | **~%52–58** |

**Daha yüksek doğruluk için** `FEATURE_MODE = "hog_enhanced"` kullanın (varsayılan). Veri ön işleme (kontrast normalizasyonu), renk histogramı ve LBP ile özellik kalitesi artar; %55–58 bandına çıkılabilir.

## 4. %60–70 için

k-NN + el yapımı özelliklerle CIFAR-10'da genelde **%55–58** pratik üst sınırdır. **%60–70+** için SVM (RBF çekirdeği) veya basit CNN gerekir; ödev k-NN ile sınırlıysa **hog_enhanced** ile %55’e yaklaşmak hedeflenebilir.
