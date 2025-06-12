import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def smart_inference():
    print("Akıllı inference testi başlıyor...")
    
    # Model yükle (5 sınıflı olarak)
    model_path = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    print(f"Model sınıf sayısı: {model.config.num_labels}")
    
    # 5 sınıfı 3'e mapping et
    # LABEL_0 → entailment
    # LABEL_1 → neutral  
    # LABEL_2 → contradiction
    # LABEL_3 → neutral (extra)
    # LABEL_4 → contradiction (extra)
    
    class_mapping = {
        0: "entailment",
        1: "neutral", 
        2: "contradiction",
        3: "neutral",     # Extra sınıfları da neutral'a map et
        4: "contradiction" # Bu da contradiction'a
    }
    
    # Test cümleleri
    test_pairs = [
        ("Bu araba kırmızıdır.", "Bu araç kırmızı renklidir."),     # entailment bekleniyor
        ("Dışarıda kar yağıyor.", "Hava çok sıcak."),              # contradiction bekleniyor  
        ("Mehmet işe gitti.", "Mehmet evde kaldı."),               # contradiction bekleniyor
        ("Ankara Türkiye'nin başkentidir.", "Ankara bir şehirdir."), # entailment bekleniyor
        ("Kedim çok tatlı.", "Köpeğim çok akıllı."),               # neutral bekleniyor
        ("Bugün pazartesi.", "Yarın salı."),                       # entailment bekleniyor
    ]
    
    print("\nAkıllı inference sonuçları:")
    print("=" * 80)
    
    model.eval()
    
    for i, (premise, hypothesis) in enumerate(test_pairs, 1):
        # Tokenize
        inputs = tokenizer(
            premise, 
            hypothesis, 
            truncation=True, 
            padding="max_length", 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Tahmin
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # 5 sınıftan 3'e mapping
        mapped_label = class_mapping[predicted_class]
        
        # 3 sınıf için yeniden olasılık hesapla
        entailment_prob = probabilities[0].item()  # LABEL_0
        neutral_prob = probabilities[1].item() + probabilities[3].item()  # LABEL_1 + LABEL_3  
        contradiction_prob = probabilities[2].item() + probabilities[4].item()  # LABEL_2 + LABEL_4
        
        # En yüksek olasılığı bul
        final_probs = [entailment_prob, neutral_prob, contradiction_prob]
        final_labels = ["entailment", "neutral", "contradiction"]
        final_prediction = final_labels[final_probs.index(max(final_probs))]
        final_confidence = max(final_probs)
        
        print(f"\nTest {i}:")
        print(f"Öncül: {premise}")
        print(f"Hipotez: {hypothesis}")
        print(f"Ham tahmin: LABEL_{predicted_class} ({confidence:.3f})")
        print(f"Mapping: {mapped_label}")
        print(f"Final tahmin: {final_prediction} ({final_confidence:.3f})")
        print(f"Dağılım: entailment={entailment_prob:.3f}, neutral={neutral_prob:.3f}, contradiction={contradiction_prob:.3f}")

if __name__ == "__main__":
    smart_inference() 