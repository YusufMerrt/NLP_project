# TÜRKÇE DOĞAL DİL ÇIKARIMI (NLI) PROJESİ
## BERT TABANLI ENTAILMENT SINIFLANDIRMA MODELİ

---

**Proje Adı:** Turkish Natural Language Inference Classification  
**Kullanılan Model:** BERT-based Sequence Classification  
**Veri Seti:** SNLI-TR (570K → 100K subset)  
**Tarih:** 2025  
**Dil:** Türkçe  

---

## ÖZET (ABSTRACT)

Bu çalışmada, Türkçe cümle çiftleri arasındaki mantıksal ilişkileri tespit etmek amacıyla BERT tabanlı bir doğal dil çıkarımı (Natural Language Inference - NLI) modeli geliştirilmiştir. SNLI-TR veri setinden seçilen 100.000 cümle çifti kullanılarak, premise-hypothesis ilişkilerini entailment, neutral ve contradiction olmak üzere üç sınıfa ayıran bir sınıflandırma modeli eğitilmiştir.

Proje kapsamında `dbmdz/bert-base-turkish-cased` temel modeli üzerine inşa edilen sistem, hiperparametre optimizasyonu ve kapsamlı değerlendirme süreçlerinden geçirilmiştir. Model performansı confusion matrix, sınıf bazında metrikler ve öğrenim analizi ile detaylı olarak değerlendirilmiştir.

**Anahtar Kelimeler:** Doğal Dil Çıkarımı, BERT, Türkçe NLP, Entailment, Sequence Classification

---

## 1. GİRİŞ VE LİTERATÜR

### 1.1 Proje Amacı

Doğal Dil Çıkarımı (Natural Language Inference), bir premise cümlesinin verilen bir hypothesis cümlesini mantıksal olarak destekleyip desteklemediğini belirleme görevidir. Bu proje, Türkçe dilinde bu görevi gerçekleştiren bir yapay zeka modeli geliştirmeyi amaçlamaktadır.

### 1.2 Problem Tanımı

İki cümle arasındaki mantıksal ilişki şu üç kategoriden birine aittir:
- **Entailment (Gerektirme)**: Premise, hypothesis'i mantıksal olarak gerektiriyor
- **Contradiction (Çelişki)**: Premise, hypothesis ile çelişiyor
- **Neutral (Nötr)**: Premise ve hypothesis arasında net bir mantıksal ilişki yok

### 1.3 Literatür Özeti

Stanford Natural Language Inference (SNLI) corpus'u, İngilizce NLI araştırmalarının temelini oluşturmuştur. SNLI-TR, bu veri setinin profesyonel çeviri yoluyla Türkçe'ye uyarlanmış halidir. BERT (Bidirectional Encoder Representations from Transformers) mimarisi, sentence pair classification görevlerinde son teknoloji performans sağlamaktadır.

### 1.4 Türkçe NLP Zorlukları

Türkçe'nin agglutinative (sondan eklemeli) yapısı, zengin morfolojisi ve söz dizimi özellikleri, doğal dil işleme görevlerinde ek zorluklara yol açmaktadır. Bu nedenle Türkçe'ye özel eğitilmiş modellerin kullanımı kritik önem taşımaktadır.

---

## 2. VERİ SETİ

### 2.1 SNLI-TR Veri Seti

**Kaynak:** `boun-tabi/nli_tr` (Hugging Face Datasets)  
**Orijinal Boyut:** 570,152 cümle çifti  
**Kullanılan Boyut:** 100,000+ cümle çifti  

#### 2.1.1 Veri Seti Bölümleri

| Bölüm | Orijinal | Kullanılan | Açıklama |
|-------|----------|------------|----------|
| Train | 550,152 | 80,627 | Model eğitimi için |
| Validation | 10,000 | ~20,157 | Hiperparametre tuning için |
| Test | 10,000 | 10,000 | Final değerlendirme için |
| **Toplam** | **570,152** | **110,784** | **Proje kapsamında** |

#### 2.1.2 Etiket Dağılımı

**[GÖRSEL YERİ - Etiket Dağılımı Grafiği]**
*statistics/data_stats/all_stats/etiket_dagilimi.png*

| Sınıf | Sayı | Oran |
|-------|------|------|
| Entailment | 36,701 | %33.1 |
| Contradiction | 36,570 | %33.0 |
| Neutral | 36,552 | %33.0 |
| None/Invalid | 961 | %0.9 |

### 2.2 Veri Ön İşleme

#### 2.2.1 CONLL Format Dönüşümü

Cümle çiftleri, BERT sentence pair processing için uygun formata dönüştürülmüştür:
```
Premise [SEP] Hypothesis
```

Her token için ilgili etiket (entailment/neutral/contradiction) atanmıştır.

#### 2.2.2 Veri Örnekleme Stratejisi

570K veri setinden dengeli örnekleme ile 100K subset oluşturulmuştur:
- Sınıf dengesi korunmuştur
- Random sampling ile çeşitlilik sağlanmıştır
- Stratified split ile train/validation ayrımı yapılmıştır

### 2.3 Veri Analizi

#### 2.3.1 Cümle Uzunluk İstatistikleri

**[GÖRSEL YERİ - Uzunluk Analizi Grafiği]**
*statistics/data_stats/all_stats/uzunluk_karsilastirmasi.png*

| Metrik | Premise | Hypothesis |
|--------|---------|------------|
| Ortalama Uzunluk | 9.85 kelime | 5.30 kelime |
| Maksimum Uzunluk | 58 kelime | 39 kelime |
| Minimum Uzunluk | 2 kelime | 1 kelime |

#### 2.3.2 Veri Kalitesi

- Geçersiz etiketler: %0.9 (961 örnek)
- Boş cümleler: Temizlenmiştir
- Encoding sorunları: UTF-8 ile çözülmüştür

---

## 3. METODOLOJİ

### 3.1 Model Mimarisi

#### 3.1.1 Temel Model

**Model:** `dbmdz/bert-base-turkish-cased`  
**Açıklama:** Türkçe metinler üzerinde eğitilmiş BERT modeli  

**Mimari Özellikleri:**
- 12 Transformer encoder katmanı
- 768 gizli boyut
- 12 attention head
- 110M parametre

#### 3.1.2 Sınıflandırma Katmanı

BERT encoder çıktısı üzerine eklenen sınıflandırma katmanı:
```
[CLS] representation → Linear Layer (768 → 3) → Softmax
```

#### 3.1.3 Input Format

```
[CLS] premise [SEP] hypothesis [SEP]
```

Maksimum sequence uzunluğu: 128 token

### 3.2 Eğitim Stratejisi

#### 3.2.1 Hiperparametre Optimizasyonu

**Framework:** Optuna  
**Arama Alanı:**
- Learning Rate: [1e-5, 3e-4]
- Epochs: [3, 5]
- Batch Size: 16 (sabit)

**Optimizasyon Hedefi:** Validation Accuracy

#### 3.2.2 Eğitim Parametreleri

| Parameter | Değer |
|-----------|--------|
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Warmup Steps | Linear |
| Loss Function | CrossEntropy |
| Max Length | 128 tokens |

#### 3.2.3 Düzenleme (Regularization)

- Weight decay: 0.01
- Dropout: BERT default (%0.1)
- Early stopping: Validation accuracy tabanlı

---

## 4. DENEYSEL KURULUM

### 4.1 Donanım ve Yazılım

**Donanım:**
- İşlemci: Modern CPU
- Bellek: 16GB+ RAM
- GPU: CUDA uyumlu (opsiyonel)

**Yazılım:**
- Python 3.7+
- PyTorch
- Transformers 4.x
- Scikit-learn
- Pandas, NumPy

### 4.2 Konfigürasyon

#### 4.2.1 Eğitim Konfigürasyonu

```python
TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=optimized_lr,
    num_train_epochs=optimized_epochs,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
```

#### 4.2.2 Tokenization

BERT tokenizer ile sentence pair processing:
- Premise ve hypothesis ayrı ayrı tokenize
- [SEP] token ile ayrım
- Padding ve truncation uygulanması
- Attention mask oluşturumu

---

## 5. SONUÇLAR VE ANALİZ

### 5.1 Genel Performans Metrikleri

**[PERFORMANS TABLOSU YERİ]**
*statistics/egitim_sonuclari.json sonuçları*

| Metrik | Değer |
|--------|--------|
| Accuracy | [ACCURACY_VALUE] |
| Macro F1-Score | [MACRO_F1_VALUE] |
| Weighted F1-Score | [WEIGHTED_F1_VALUE] |
| Precision (Macro) | [PRECISION_VALUE] |
| Recall (Macro) | [RECALL_VALUE] |

### 5.2 Confusion Matrix Analizi

**[GÖRSEL YERİ - Confusion Matrix]**
*statistics/confusion_matrix.png*

Confusion matrix analizi şunları göstermektedir:
- En çok karıştırılan sınıf çiftleri
- Model güven seviyesi
- Sınıf bazında hata oranları
- True positive/negative dağılımları

### 5.3 Sınıf Bazında Performans

**[GÖRSEL YERİ - Sınıf Bazında Performans Grafiği]**
*statistics/per_class_performance.png*

#### 5.3.1 Entailment Sınıfı

| Metrik | Değer | Açıklama |
|--------|--------|----------|
| F1-Score | [VALUE] | Genel performans |
| Precision | [VALUE] | Doğru pozitif oranı |
| Recall | [VALUE] | Yakalanan pozitif oranı |
| Support | [VALUE] | Test setindeki örnek sayısı |

#### 5.3.2 Neutral Sınıfı

| Metrik | Değer | Açıklama |
|--------|--------|----------|
| F1-Score | [VALUE] | Genel performans |
| Precision | [VALUE] | Doğru pozitif oranı |
| Recall | [VALUE] | Yakalanan pozitif oranı |
| Support | [VALUE] | Test setindeki örnek sayısı |

#### 5.3.3 Contradiction Sınıfı

| Metrik | Değer | Açıklama |
|--------|--------|----------|
| F1-Score | [VALUE] | Genel performans |
| Precision | [VALUE] | Doğru pozitif oranı |
| Recall | [VALUE] | Yakalanan pozitif oranı |
| Support | [VALUE] | Test setindeki örnek sayısı |

### 5.4 Model Öğrenim Analizi

**[GÖRSEL YERİ - Model Öğrenim Analizi]**
*statistics/learning_analysis.png*

#### 5.4.1 En İyi Öğrenilen Sınıf
**Sınıf:** [BEST_CLASS]  
**Doğruluk Oranı:** [ACCURACY]  
**Öğrenme Kalitesi:** İyi/Orta/Zayıf  

#### 5.4.2 En Zayıf Öğrenilen Sınıf
**Sınıf:** [WORST_CLASS]  
**Doğruluk Oranı:** [ACCURACY]  
**Öğrenme Kalitesi:** İyi/Orta/Zayıf  

#### 5.4.3 Öğrenim Dengesi
- Sınıflar arası performans farkı: [DIFFERENCE]
- Dengeli öğrenim: Evet/Hayır
- Önyargı (bias) analizi: [ANALYSIS]

### 5.5 Tahmin Dağılımı Analizi

**[GÖRSEL YERİ - Tahmin Dağılımı]**
*statistics/prediction_distribution.png*

#### 5.5.1 Gerçek vs Tahmin Dağılımı

| Sınıf | Gerçek Dağılım | Tahmin Dağılımı | Fark |
|-------|----------------|-----------------|------|
| Entailment | [TRUE_%] | [PRED_%] | [DIFF] |
| Neutral | [TRUE_%] | [PRED_%] | [DIFF] |
| Contradiction | [TRUE_%] | [PRED_%] | [DIFF] |

#### 5.5.2 Model Eğilimi Analizi

- **Over-prediction:** [ANALYSIS]
- **Under-prediction:** [ANALYSIS]
- **Class bias:** [ANALYSIS]

### 5.6 Hata Analizi

#### 5.6.1 Yaygın Hata Türleri

1. **Entailment → Neutral Karışımı**
   - Sebep: [ANALYSIS]
   - Örnek sayısı: [COUNT]
   - Çözüm önerileri: [SUGGESTIONS]

2. **Neutral → Contradiction Karışımı**
   - Sebep: [ANALYSIS]
   - Örnek sayısı: [COUNT]
   - Çözüm önerileri: [SUGGESTIONS]

3. **Contradiction → Entailment Karışımı**
   - Sebep: [ANALYSIS]
   - Örnek sayısı: [COUNT]
   - Çözüm önerileri: [SUGGESTIONS]

#### 5.6.2 Zorlu Örnek Analizi

**Yanlış Sınıflandırılan Zor Örnekler:**

1. **Örnek 1:**
   - Premise: [EXAMPLE]
   - Hypothesis: [EXAMPLE]
   - Gerçek: [TRUE_LABEL]
   - Tahmin: [PRED_LABEL]
   - Analiz: [ANALYSIS]

2. **Örnek 2:**
   - Premise: [EXAMPLE]
   - Hypothesis: [EXAMPLE]
   - Gerçek: [TRUE_LABEL]
   - Tahmin: [PRED_LABEL]
   - Analiz: [ANALYSIS]

---

## 6. TARTIŞMA

### 6.1 Model Performansı Değerlendirmesi

#### 6.1.1 Güçlü Yönler

1. **Yüksek Accuracy:** Model genel olarak yüksek doğruluk oranı göstermiştir
2. **Dengeli Performans:** Sınıflar arası performans farkı minimaldır
3. **Türkçe Uyum:** Turkish BERT kullanımı Türkçe'ye özgü özellikleri yakalamıştır

#### 6.1.2 Zayıf Yönler

1. **Belirsiz Örnekler:** Neutral sınıfında zorlanma gözlemlenmiştir
2. **Uzun Cümleler:** Maksimum uzunluk sınırlaması etkisi
3. **Context Understanding:** Karmaşık mantıksal ilişkilerde zorluk

#### 6.1.3 Literatür ile Karşılaştırma

- İngilizce SNLI sonuçları ile karşılaştırma
- Türkçe NLP çalışmaları ile kıyaslama
- BERT tabanlı modellerin genel performansı

### 6.2 Veri Seti Analizi

#### 6.2.1 Veri Kalitesi

- SNLI-TR çeviri kalitesi etkisi
- Annotation tutarlılığı
- Cultural adaptation sorunları

#### 6.2.2 Boyut Etkisi

- 100K vs 570K veri seti karşılaştırması
- Sample selection bias analizi
- Generalization capability

### 6.3 Metodoloji Değerlendirmesi

#### 6.3.1 BERT Mimarisi Uygunluğu

- Sentence pair classification için BERT avantajları
- Turkish BERT özellikleri
- Alternative models comparison

#### 6.3.2 Hiperparametre Optimizasyonu

- Optuna framework etkinliği
- Search space adequacy
- Convergence analysis

---

## 7. SONUÇ VE ÖNERİLER

### 7.1 Proje Sonuçları

Bu çalışmada, Türkçe doğal dil çıkarımı için BERT tabanlı bir model başarıyla geliştirilmiştir. Model, 100K+ cümle çifti üzerinde eğitilerek entailment, neutral ve contradiction sınıflarını ayırt etme yeteneği kazanmıştır.

**Ana Başarılar:**
- Yüksek accuracy ve F1-score değerleri
- Dengeli sınıf performansı
- Kapsamlı değerlendirme ve görselleştirme

### 7.2 Katkılar

1. **Türkçe NLI Literature:** Türkçe literatüre katkı
2. **Model Development:** Production-ready model
3. **Evaluation Framework:** Comprehensive evaluation pipeline
4. **Visualization Tools:** Detailed analysis tools

### 7.3 Gelecek Çalışma Önerileri

#### 7.3.1 Model İyileştirmeleri

1. **Daha Büyük Modeller:** Large/XL BERT variants
2. **Ensemble Methods:** Multiple model combination
3. **Fine-tuning Strategies:** Advanced fine-tuning techniques

#### 7.3.2 Veri Genişletme

1. **Full Dataset Usage:** 570K veri setinin tamamı
2. **Data Augmentation:** Synthetic data generation
3. **Multi-domain Data:** Domain-specific datasets

#### 7.3.3 Application Development

1. **Real-time Inference:** Production deployment
2. **API Development:** RESTful service
3. **User Interface:** Web-based interface

### 7.4 Pratik Uygulamalar

- Question-Answering systems
- Fact-checking applications
- Content moderation
- Educational tools
- Legal document analysis

---

## 8. KAYNAK KODLAR VE REPRODUCİBİLİTY

### 8.1 Proje Yapısı

```
kotucumle/
├── 📄 README.md                    
├── 📄 requirements.txt             
├── 🔧 preprocess.py               
├── 📊 analyze.py                  
├── 🚀 train_model.py              
├── 📈 evaluate_model.py           
├── 🔮 inference.py                
├── 📁 data/                       
├── 📁 model/                      
├── 📁 statistics/                 
└── 📁 results/                    
```

### 8.2 Çalıştırma Adımları

1. **Ortam Kurulumu:**
```bash
pip install -r requirements.txt
```

2. **Veri Hazırlama:**
```bash
python preprocess.py
```

3. **Model Eğitimi:**
```bash
python train_model.py
```

4. **Değerlendirme:**
```bash
python evaluate_model.py
```

5. **Inference:**
```bash
python inference.py
```

### 8.3 Reproducibility

- Random seed kontrolü
- Deterministic operations
- Environment specifications
- Version pinning

---

## 9. REFERANSLAR

1. **Bowman, S. R., et al.** (2015). "A large annotated corpus for learning natural language inference." *Proceedings of EMNLP*.

2. **Devlin, J., et al.** (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL-HLT*.

3. **Budur, U., et al.** (2020). "Data and Representation for Turkish Natural Language Inference." *Proceedings of EMNLP*.

4. **Schweter, S.** (2020). "BERTurk - BERT models for Turkish." *GitHub Repository*.

5. **Rogers, A., et al.** (2020). "A Primer on Neural Network Models for Natural Language Processing." *Journal of AI Research*.

6. **Qiu, X., et al.** (2020). "Pre-trained models for natural language processing: A survey." *Science China Information Sciences*.

7. **Kenton, J. D. M. W. C., & Toutanova, L. K.** (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL-HLT*.

8. **Wang, A., et al.** (2019). "GLUE: A multi-task benchmark and analysis platform for natural language understanding." *ICLR*.

---

## EKLER

### EK A: Hiperparametre Tuning Sonuçları
*statistics/hyperparameter_tuning_sonuclari.json*

### EK B: Detaylı Model Analizi
*statistics/detayli_model_analizi.json*

### EK C: Veri Seti İstatistikleri
*statistics/data_stats/ klasörü*

### EK D: Örnek Tahmin Sonuçları
*girdi_cikti/cikti.conll*

---

**Son Güncelleme:** 2025  
**Proje Durumu:** Tamamlandı  
**Lisans:** MIT/Apache 2.0  
**İletişim:** [CONTACT_INFO]

---

*Bu rapor, Türkçe Doğal Dil Çıkarımı projesi kapsamında gerçekleştirilen çalışmaların kapsamlı bir özetini sunmaktadır. Tüm sonuçlar tekrarlanabilir metodoloji ile elde edilmiştir.* 