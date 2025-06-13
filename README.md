# Turkish Natural Language Inference (NLI) Classification

Bu proje, Türkçe cümle çiftleri arasındaki mantıksal ilişkileri tespit eden bir **Natural Language Inference (Doğal Dil Çıkarımı)** modeli geliştirmektedir. BERT tabanlı derin öğrenme modeli kullanılarak **entailment**, **neutral** ve **contradiction** sınıflarında sınıflandırma yapılır.

## 📋 İçindekiler
- [Proje Özeti](#proje-özeti)
- [Veri Seti](#veri-seti)
- [Kurulum](#kurulum)
- [Veri Analizi](#veri-analizi)
- [Model Eğitimi](#model-eğitimi)
- [Model Değerlendirme](#model-değerlendirme)
- [Inference (Tahmin)](#inference-tahmin)
- [Dosya Yapısı](#dosya-yapısı)
- [Sonuçlar](#sonuçlar)
- [Kullanım](#kullanım)

## 🎯 Proje Özeti

Bu proje, **570K** örnek içeren büyük Türkçe NLI veri setinden **100K** örnek kullanarak eğitilmiş bir doğal dil çıkarımı modelidir. Model, iki cümle arasındaki mantıksal ilişkiyi analiz ederek:

- **Entailment (Gerektirme)**: İlk cümle ikinci cümleyi mantıksal olarak gerektiriyor
- **Contradiction (Çelişki)**: İlk cümle ikinci cümle ile çelişiyor  
- **Neutral (Nötr)**: İlk ve ikinci cümle arasında net bir mantıksal ilişki yok

### 🔧 Teknik Özellikler
- **Base Model**: `dbmdz/bert-base-turkish-cased`
- **Veri Seti**: `boun-tabi/nli_tr` (SNLI-TR)
- **Eğitim Verisi**: 100K cümle çifti (570K'dan seçilmiş)
- **Performans**: Detaylı görsel analiz ve metrikler

## 📊 Veri Seti

[SNLI-TR Dataset](https://huggingface.co/datasets/boun-tabi/nli_tr) kullanılmıştır:

- **Kaynak**: SNLI'nin profesyonel Türkçe çevirisi
- **Toplam**: 570,152 cümle çifti
  - Train: 550,152 örnek
  - Validation: 10,000 örnek  
  - Test: 10,000 örnek
- **Kullanılan**: 100K örnek (dengeli sampling)

### Veri Dağılımı

<!-- VERİ DAĞILIMI GRAFİĞİ BURAYA GELECEKː statistics/data_stats/all_stats/etiket_dagilimi.png -->
![image](https://github.com/user-attachments/assets/474face5-fbac-45e9-90cd-2432abb7b6d4)


### Cümle Uzunluk Analizi

<!-- UZUNLUK ANALİZİ GRAFİĞİ BURAYA GELECEKː statistics/data_stats/all_stats/uzunluk_karsilastirmasi.png -->
![image](https://github.com/user-attachments/assets/67fb8885-d6f3-4828-9d6c-156c88c50361)

## 🛠 Kurulum

### Gereksinimler
- Python 3.7+
- PyTorch
- Transformers
- CUDA (opsiyonel, GPU hızlandırması için)

### Kurulum Adımları

```bash
# Projeyi klonlayın
git clone <repository-url>
cd kotucumle

# Sanal ortam oluşturun
python -m venv venv

# Sanal ortamı aktifleştirin
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

## 📈 Veri Analizi

Veri analizi `analyze.py` ile gerçekleştirilir:

```bash
python analyze.py
```

### Analiz Sonuçları

Analiz şunları içerir:
- Cümle çifti sayıları ve uzunluk istatistikleri
- Etiket dağılımları (train/validation/test)
- Premise vs Hypothesis karşılaştırmaları
- Görsel grafikler ve istatistik raporları

Tüm analiz sonuçları `statistics/data_stats/` klasöründe saklanır.

## 🚀 Model Eğitimi

### Veri Hazırlama

```bash
python preprocess.py
```

Bu script:
- 570K veri setinden 100K örnek seçer (dengeli sampling)
- CONLL formatına dönüştürür
- Train/validation/test olarak böler

### Model Eğitimi

```bash
python train_model.py
```

**Eğitim Parametreleri:**
- **Model**: Turkish BERT (`dbmdz/bert-base-turkish-cased`)
- **Epochs**: 3-5 (hiperparametre tuning ile)
- **Batch Size**: 16
- **Learning Rate**: 1e-5 to 3e-4 (arama ile bulunur)
- **Max Length**: 128 tokens
- **Hiperparametre Tuning**: Optuna ile otomatik

Eğitim sonuçları `results/` ve `statistics/` klasörlerinde saklanır.

## 📊 Model Değerlendirme

```bash
python evaluate_model.py
```

Bu script detaylı analiz sağlar:

### Performans Metrikleri

<!-- GENEL PERFORMANS METRİKLERİ BURAYA GELECEKː evaluate_model.py çıktısı -->
## Evaluation Metrics

| Metric | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| **Class 0** | 0.7744 | 0.7010 | 0.7359 | 3237 |
| **Class 1** | 0.7781 | 0.7996 | 0.7887 | 3368 |
| **Class 2** | 0.6633 | 0.7074 | 0.6846 | 3219 |
| **Macro Avg** | 0.7386 | 0.7360 | 0.7364 | 9824 |
| **Weighted Avg** | 0.7393 | 0.7369 | 0.7372 | 9824 |

## Additional Metrics

- **Eval Loss:** 0.7286
- **Eval Accuracy:** 0.7369
- **Model Preparation Time:** 0.002 sec
- **Eval Runtime:** 155.3466 sec
- **Samples per Second:** 63.239
- **Steps per Second:** 3.952


### Confusion Matrix

<!-- CONFUSION MATRIX GRAFİĞİ BURAYA GELECEKː statistics/confusion_matrix.png -->
![image](https://github.com/user-attachments/assets/ad9a2f71-a974-4e91-9c04-4bfc81e7125c)


### Sınıf Bazında Performans

<!-- SINIF BAZINDA PERFORMANS GRAFİĞİ BURAYA GELECEKː statistics/per_class_performance.png -->
![image](https://github.com/user-attachments/assets/3c5d3109-d105-42ec-b615-51b6669928c6)


### Model Öğrenim Analizi

<!-- MODEL ÖĞRENİM ANALİZİ GRAFİĞİ BURAYA GELECEKː statistics/learning_analysis.png -->
![image](https://github.com/user-attachments/assets/3b0f8a54-b39b-4e72-82a6-377b343c64e4)


### Tahmin Dağılımı

<!-- TAHMİN DAĞILIMI GRAFİĞİ BURAYA GELECEKː statistics/prediction_distribution.png -->
![image](https://github.com/user-attachments/assets/c9cd352c-1fc2-46f1-bac8-a81b2480f986)


## 🔮 Inference (Tahmin)

### Kullanım

1. **Giriş dosyasını hazırlayın** (`girdi_cikti/girdi.txt`):
```
Bugün hava çok güzel.
Dışarıda güneş parlıyor.
Kediler evcil hayvanlardır.
Kediler vahşi hayvanlardır.
```

2. **Tahmin çalıştırın**:
```bash
python inference.py
```

3. **Sonuçları kontrol edin** (`girdi_cikti/cikti.conll`):
```
Bugün	entailment
hava	entailment
çok	entailment
güzel	entailment
.	entailment
[SEP]	entailment
Dışarıda	entailment
güneş	entailment
parlıyor	entailment
.	entailment

Kediler	contradiction
evcil	contradiction
hayvanlardır	contradiction
.	contradiction
[SEP]	contradiction
Kediler	contradiction
vahşi	contradiction
hayvanlardır	contradiction
.	contradiction
```

## 📁 Dosya Yapısı

```
kotucumle/
├── 📄 README.md                    # Bu dosya
├── 📄 requirements.txt             # Python bağımlılıkları
├── 🔧 preprocess.py               # Veri ön işleme
├── 📊 analyze.py                  # Veri analizi
├── 🚀 train_model.py              # Model eğitimi
├── 📈 evaluate_model.py           # Model değerlendirme
├── 🔮 inference.py                # Tahmin yapma
├── 📁 data/                       # Veri dosyaları
│   ├── train.conll               # Eğitim verisi (~80K)
│   ├── validation.conll          # Doğrulama verisi (~20K)
│   └── test.conll                # Test verisi (10K)
├── 📁 model/                      # Eğitilmiş model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
├── 📁 girdi_cikti/               # Inference giriş/çıkış
│   ├── girdi.txt                 # Tahmin edilecek cümle çiftleri
│   └── cikti.conll              # Tahmin sonuçları
├── 📁 statistics/                # Analiz sonuçları
│   ├── 🖼️ confusion_matrix.png
│   ├── 🖼️ per_class_performance.png
│   ├── 🖼️ learning_analysis.png
│   ├── 🖼️ prediction_distribution.png
│   ├── 📊 detayli_model_analizi.json
│   ├── 📊 egitim_sonuclari.json
│   └── 📁 data_stats/            # Veri analizi sonuçları
└── 📁 results/                   # Eğitim çıktıları
```

## 📈 Sonuçlar

### Model Performansı

<!-- MODEL PERFORMANS TABLOSU BURAYA GELECEKː JSON sonuçlarından -->
**Doğruluk Oranı: 0.7369**
**Macro F1: 0.7364**
**Weighted F1: 0.7372**

### Sınıf Bazında Sonuçlar

<!-- SINIF BAZINDA DETAYLI SONUÇLAR BURAYA GELECEKː -->
**ENTAILMENT:**
  - F1-Score: 0.7359
  - Precision: 0.7744
  - Recall: 0.7010
  - Doğru Tahmin: 2269/3237
  - Öğrenme Kalitesi: Orta

**NEUTRAL:**
  - F1-Score: 0.7887
  - Precision: 0.7781
  - Recall: 0.7996
  - Doğru Tahmin: 2693/3368
  - Öğrenme Kalitesi: Orta

**CONTRADICTION:**
  - F1-Score: 0.6846
  - Precision: 0.6633
  - Recall: 0.7074
  - Doğru Tahmin: 2277/3219
  - Öğrenme Kalitesi: Orta

### En İyi ve En Kötü Öğrenilen Sınıflar

<!-- ÖĞRENİM KALİTESİ ANALİZİ BURAYA GELECEKː -->
**En iyi öğrenilen sınıf: neutral**
**En zayıf öğrenilen sınıf: entailment**

## 🎮 Kullanım Örnekleri

### Örnek 1: Entailment
```
Premise: "Çocuk parkta top oynuyor."
Hypothesis: "Bir çocuk dışarıda oyun oynuyor."
Sonuç: ENTAILMENT
```

### Örnek 2: Contradiction
```
Premise: "Kediler evcil hayvanlardır."
Hypothesis: "Kediler vahşi hayvanlardır."
Sonuç: CONTRADICTION
```

### Örnek 3: Neutral
```
Premise: "Adam kahve içiyor."
Hypothesis: "Adam çay seviyor."
Sonuç: NEUTRAL
```

## 🔬 Teknik Detaylar

### Model Mimarisi
- **Encoder**: Turkish BERT (12 layers, 768 hidden, 12 attention heads)
- **Classifier**: Linear layer (768 → 3 classes)
- **Input**: Sentence pairs with [SEP] token
- **Output**: Softmax probability distribution

### Eğitim Stratejisi
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: Linear warmup + decay
- **Loss Function**: CrossEntropy
- **Regularization**: Weight decay (0.01)
- **Early Stopping**: Validation accuracy based

### Hiperparametre Optimizasyonu
- **Framework**: Optuna
- **Search Space**: Learning rate, epochs, batch size
- **Objective**: Validation accuracy
- **Trials**: Multiple runs with different configurations

## 📚 Referanslar

- [Turkish NLI Dataset (SNLI-TR)](https://huggingface.co/datasets/boun-tabi/nli_tr)
- [Original SNLI Paper](https://nlp.stanford.edu/projects/snli/)
- [NLI-TR Paper](https://aclanthology.org/2020.emnlp-main.695/)
- [Turkish BERT Model](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- [Transformers Library](https://github.com/huggingface/transformers)

## 📞 İletişim

Proje hakkında sorularınız için issue açabilir veya iletişime geçebilirsiniz.

---

**📊 Bu proje, Türkçe doğal dil işleme alanında entailment classification için kapsamlı bir çözüm sunmaktadır.**
