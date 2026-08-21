# 🧠 Parkinson Hastalığı Sınıflandırması

Bu projede Parkinson hastalığına ait ses özelliklerinden oluşan bir veri seti kullanılarak farklı makine öğrenmesi sınıflandırma algoritmaları karşılaştırılmıştır.

Amacım sadece bir model oluşturmak değil, farklı modellerin performanslarını incelemek, hiperparametrelerini karşılaştırmak ve hangi modelin bu veri setinde daha başarılı olduğunu gözlemlemekti.

## 📊 Veri Seti

Projede UCI Parkinson's Disease veri seti kullanılmıştır.

Veri setinde toplam **195 gözlem** ve **22 özellik** bulunmaktadır.

- 147 Parkinson
- 48 Sağlıklı

Kullanılan özellikler arasında ses frekansı, jitter, shimmer, HNR, RPDE, DFA, spread1, spread2, D2 ve PPE gibi değerler bulunmaktadır.

## 🔍 Veri Analizi

Modelleme öncesinde veri setini daha iyi anlamak için:

- Özelliklerin dağılımları incelendi.
- Korelasyon analizi yapıldı.
- Önemli özellikler için boxplot grafikleri oluşturuldu.
- Veriler eğitim ve test olarak ayrıldı.
- Özellikler `StandardScaler` ile ölçeklendirildi.

## 🤖 Kullanılan Modeller

Projede dört farklı sınıflandırma algoritması kullanıldı:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Support Vector Machine (SVM)

## ⚙️ Hiperparametre Optimizasyonu

Model performanslarını artırmak ve uygun parametreleri belirlemek için **5-Fold Cross Validation** kullanıldı.

### KNN

`K=1` ile `K=15` arasındaki değerler karşılaştırıldı.

En yüksek ortalama Cross Validation accuracy:

**K = 1 → %92.9**

Final test accuracy:

**%97.44**

### Decision Tree

`max_depth=1` ile `max_depth=10` arasındaki değerler karşılaştırıldı.

En yüksek ortalama Cross Validation accuracy:

**max_depth = 4 → %85.9**

Final test accuracy:

**%92**

### SVM

Öncelikle farklı `C` değerleri karşılaştırıldı.

En iyi sonuç:

**C = 100 → %91.0**

Daha sonra `C=100` sabit tutularak farklı `gamma` değerleri karşılaştırıldı.

En iyi sonuç:

**gamma = 0.1 → %93.6**

Final SVM modeli:

- Kernel: RBF
- C: 100
- Gamma: 0.1

Final test accuracy:

**%94.87**

## 🏆 Model Karşılaştırması

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | %90 |
| KNN | **%97.44** |
| Decision Tree | %92 |
| SVM | %94.87 |

Bu veri setinde en yüksek test accuracy değeri **KNN modeli** tarafından elde edilmiştir.

## 📈 Kullanılan Görseller

Projede model performanslarını ve veri setini incelemek için çeşitli grafikler kullanılmıştır.

Bunlar arasında:

- Korelasyon heatmap
- Özellik dağılımları
- Boxplot grafikleri
- KNN K değeri karşılaştırması
- Decision Tree Max Depth karşılaştırması
- Decision Tree görselleştirmesi
- SVM C değeri karşılaştırması
- SVM Gamma değeri karşılaştırması
- Confusion Matrix grafikleri
- Model performans karşılaştırmaları

bulunmaktadır.

## 🛠️ Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- ucimlrepo

## ▶️ Projeyi Çalıştırma

Projeyi klonladıktan sonra gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt

##Medium yazısı:
https://medium.com/@rumeysakurtuloglu21/sesimiz-hastal%C4%B1k-hakk%C4%B1nda-ne-s%C3%B6yl%C3%BCyor-5697765eb49a
