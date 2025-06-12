import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def fix_model():
    print("Model düzeltme işlemi başlıyor...")
    
    # Model yükle
    model_path = "model"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Mevcut model sınıf sayısı: {model.config.num_labels}")
    
    # Model'in son katmanını (classifier) 3 sınıfa düşür
    # Sadece ilk 3 sınıfı al (0: entailment, 1: neutral, 2: contradiction)
    
    if model.config.num_labels > 3:
        print("Model'i 3 sınıfa düşürüyor...")
        
        # Classifier'ın weight ve bias'ını kes
        old_classifier = model.classifier
        
        # Yeni classifier oluştur (sadece ilk 3 sınıf)
        new_weight = old_classifier.weight[:3, :].clone()  # İlk 3 satır
        new_bias = old_classifier.bias[:3].clone()  # İlk 3 eleman
        
        # Yeni classifier ata
        from torch import nn
        model.classifier = nn.Linear(old_classifier.in_features, 3)
        model.classifier.weight.data = new_weight
        model.classifier.bias.data = new_bias
        
        # Config'i güncelle
        model.config.num_labels = 3
        
        print("Model başarıyla 3 sınıfa düşürüldü!")
    
    # Config dosyasını güncelle
    config_path = f"{model_path}/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Sınıf sayısını güncelle
    config['num_labels'] = 3
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Modeli kaydet
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print("Model ve config başarıyla kaydedildi!")
    
    # Test et
    print("\nHızlı test:")
    model.eval()
    
    test_pairs = [
        ("Bu araba kırmızıdır.", "Bu araç kırmızı renklidir."),  # entailment bekleniyor
        ("Dışarıda kar yağıyor.", "Hava çok sıcak."),  # contradiction bekleniyor
    ]
    
    labels = ["entailment", "neutral", "contradiction"]
    
    for premise, hypothesis in test_pairs:
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            confidence = torch.softmax(outputs.logits, dim=-1).max().item()
        
        print(f"'{premise}' vs '{hypothesis}' → {labels[prediction]} ({confidence:.3f})")

if __name__ == "__main__":
    fix_model() 