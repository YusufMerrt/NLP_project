import json
import os

from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_conll(file_path):
    """Read CONLL format data containing sentence pairs for entailment"""
    sentences = []
    labels = []
    sentence = []
    label = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip() == "":
                if sentence and label:
                    full_text = " ".join(sentence)
                    # Split premise and hypothesis at [SEP] token
                    if "[SEP]" in full_text:
                        parts = full_text.split("[SEP]")
                        premise = parts[0].strip()
                        hypothesis = parts[1].strip() if len(parts) > 1 else ""
                        
                        # Store as combined text for BERT sentence pair processing
                        sentences.append(f"{premise}[SEP]{hypothesis}")
                        # Take the first valid label (skip None/empty labels)
                        valid_label = None
                        for l in label:
                            if l and l.lower() in ["entailment", "neutral", "contradiction"]:
                                valid_label = l.lower()
                                break
                        if valid_label:
                            labels.append(valid_label)
                        else:
                            # Skip this sentence if no valid label found
                            sentences.pop()
                    sentence = []
                    label = []
            else:
                line_parts = line.strip().split()
                if len(line_parts) >= 2:
                    token, tag = line_parts[0], line_parts[1]
                sentence.append(token)
                label.append(tag)
        if sentence and label:
            full_text = " ".join(sentence)
            if "[SEP]" in full_text:
                parts = full_text.split("[SEP]")
                premise = parts[0].strip()
                hypothesis = parts[1].strip() if len(parts) > 1 else ""
                sentences.append(f"{premise}[SEP]{hypothesis}")
                # Take the first valid label (skip None/empty labels)
                valid_label = None
                for l in label:
                    if l and l.lower() in ["entailment", "neutral", "contradiction"]:
                        valid_label = l.lower()
                        break
                if valid_label:
                    labels.append(valid_label)
                else:
                    # Skip this sentence if no valid label found
                    sentences.pop()
    return sentences, labels


def create_dataset_dict(label_encoder):
    splits = ["test"]
    datasets = {}
    for split in splits:
        file_path = f"data/{split}.conll"
        sentences, labels = read_conll(file_path)
        labels = label_encoder.transform(labels)  # Encode labels
        datasets[split] = Dataset.from_dict({"text": sentences, "label": labels})
    return DatasetDict(datasets)


def tokenize_function(examples, tokenizer):
    """Tokenize sentence pairs for entailment classification"""
    # Split text at [SEP] for proper sentence pair processing
    premise_hypothesis_pairs = []
    for text in examples["text"]:
        if "[SEP]" in text:
            parts = text.split("[SEP]")
            premise = parts[0].strip()
            hypothesis = parts[1].strip() if len(parts) > 1 else ""
            premise_hypothesis_pairs.append((premise, hypothesis))
        else:
            # Fallback for malformed data
            premise_hypothesis_pairs.append((text, ""))
    
    premises = [pair[0] for pair in premise_hypothesis_pairs]
    hypotheses = [pair[1] for pair in premise_hypothesis_pairs]
    
    return tokenizer(
        premises, 
        hypotheses, 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return classification_report(p.label_ids, preds, output_dict=True)


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    """Confusion matrix görselleştirmesi"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - Model Performansı')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_per_class_metrics(results, label_names, save_path):
    """Sınıf bazında performans metrikleri grafiği"""
    metrics = ['precision', 'recall', 'f1-score']
    
    # Her sınıf için metrikleri topla
    class_metrics = {metric: [] for metric in metrics}
    
    for i, label in enumerate(label_names):
        if f"eval_{i}" in results:
            class_result = results[f"eval_{i}"]
            for metric in metrics:
                class_metrics[metric].append(class_result.get(metric, 0))
        else:
            for metric in metrics:
                class_metrics[metric].append(0)
    
    # Grafik oluştur
    x = np.arange(len(label_names))
    width = 0.25
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, class_metrics[metric], width, 
                label=metric.capitalize(), alpha=0.8)
    
    plt.xlabel('Sınıflar')
    plt.ylabel('Performans Skorları')
    plt.title('Sınıf Bazında Model Performansı')
    plt.xticks(x + width, label_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # Her bar üzerine değerleri yaz
    for i, metric in enumerate(metrics):
        for j, value in enumerate(class_metrics[metric]):
            plt.text(j + i*width, value + 0.01, f'{value:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'per_class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return class_metrics


def plot_prediction_distribution(y_true, y_pred, label_names, save_path):
    """Tahmin dağılımı analizi"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gerçek etiket dağılımı
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    true_labels = [label_names[i] for i in unique_true]
    ax1.pie(counts_true, labels=true_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Gerçek Etiket Dağılımı')
    
    # Tahmin edilen etiket dağılımı
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    pred_labels = [label_names[i] for i in unique_pred]
    ax2.pie(counts_pred, labels=pred_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Tahmin Edilen Etiket Dağılımı')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_model_learning(y_true, y_pred, label_names, save_path):
    """Model öğrenme analizi - hangi sınıfları daha iyi öğrenmiş"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Her sınıf için doğru tahmin oranı
    class_accuracy = []
    class_total = []
    class_correct = []
    
    for i in range(len(label_names)):
        total = cm[i].sum()
        correct = cm[i][i]
        accuracy = correct / total if total > 0 else 0
        
        class_accuracy.append(accuracy)
        class_total.append(total)
        class_correct.append(correct)
    
    # Görselleştirme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sınıf bazında doğruluk oranları
    colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in class_accuracy]
    bars1 = ax1.bar(label_names, class_accuracy, color=colors, alpha=0.7)
    ax1.set_xlabel('Sınıflar')
    ax1.set_ylabel('Doğruluk Oranı')
    ax1.set_title('Sınıf Bazında Model Başarısı')
    ax1.set_ylim(0, 1)
    
    # Her bar üzerine değerleri yaz
    for bar, acc, total in zip(bars1, class_accuracy, class_total):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}\n({class_correct[class_accuracy.index(acc)]}/{total})',
                ha='center', va='bottom', fontsize=10)
    
    # Veri sayısı dağılımı
    bars2 = ax2.bar(label_names, class_total, alpha=0.7)
    ax2.set_xlabel('Sınıflar')
    ax2.set_ylabel('Örnek Sayısı')
    ax2.set_title('Test Setinde Sınıf Dağılımı')
    
    # Her bar üzerine sayıları yaz
    for bar, total in zip(bars2, class_total):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_total)*0.01,
                str(total), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detaylı analiz raporu
    analysis_report = {
        "class_performance": {
            label_names[i]: {
                "accuracy": float(class_accuracy[i]),
                "correct_predictions": int(class_correct[i]),
                "total_samples": int(class_total[i]),
                "learning_quality": "İyi" if class_accuracy[i] > 0.8 else "Orta" if class_accuracy[i] > 0.6 else "Zayıf"
            } for i in range(len(label_names))
        },
        "best_learned_class": label_names[np.argmax(class_accuracy)],
        "worst_learned_class": label_names[np.argmin(class_accuracy)],
        "overall_insights": {
            "high_performance_classes": [label_names[i] for i, acc in enumerate(class_accuracy) if acc > 0.8],
            "low_performance_classes": [label_names[i] for i, acc in enumerate(class_accuracy) if acc < 0.6],
            "balanced_learning": bool(abs(max(class_accuracy) - min(class_accuracy)) < 0.2)
        }
    }
    
    return analysis_report


def main():
    print("Evaluating Turkish Entailment Classification Model")
    model_checkpoint = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

    # Load the labels and create a label encoder for entailment task
    label_list = ["entailment", "neutral", "contradiction"]
    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)

    dataset_dict = create_dataset_dict(label_encoder)
    tokenized_test_dataset = dataset_dict["test"].map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Running evaluation on test set...")
    results = trainer.evaluate(tokenized_test_dataset)
    
    # Tahminleri al
    predictions = trainer.predict(tokenized_test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    y_true = predictions.label_ids
    
    # İstatistik klasörünü oluştur
    os.makedirs("statistics", exist_ok=True)
    
    # Görselleştirmeleri oluştur
    print("Generating visualizations...")
    
    # Confusion Matrix
    cm = plot_confusion_matrix(y_true, y_pred, label_list, "statistics")
    print("✓ Confusion matrix kaydedildi")
    
    # Sınıf bazında performans
    class_metrics = plot_per_class_metrics(results, label_list, "statistics")
    print("✓ Sınıf bazında performans grafiği kaydedildi")
    
    # Tahmin dağılımı
    plot_prediction_distribution(y_true, y_pred, label_list, "statistics")
    print("✓ Tahmin dağılımı grafiği kaydedildi")
    
    # Model öğrenme analizi
    learning_analysis = analyze_model_learning(y_true, y_pred, label_list, "statistics")
    print("✓ Model öğrenme analizi kaydedildi")
    
    # Tüm sonuçları kaydet
    detailed_results = {
        "evaluation_metrics": results,
        "learning_analysis": learning_analysis,
        "confusion_matrix": cm.tolist(),
        "class_metrics": class_metrics
    }
    
    with open("statistics/detayli_model_analizi.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    
    # Eski format için de kaydet
    with open("statistics/egitim_sonuclari.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*60)
    print("MODEL PERFORMANS RAPORU")
    print("="*60)
    
    # Genel metrikler
    print(f"\n📊 GENEL PERFORMANS:")
    print(f"Doğruluk Oranı: {results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Macro F1: {results.get('eval_macro avg', {}).get('f1-score', 'N/A'):.4f}")
    print(f"Weighted F1: {results.get('eval_weighted avg', {}).get('f1-score', 'N/A'):.4f}")
    
    # Sınıf bazında sonuçlar
    print(f"\n📈 SINIF BAZINDA PERFORMANS:")
    for i, label in enumerate(label_list):
        if f"eval_{i}" in results:
            class_result = results[f"eval_{i}"]
            analysis = learning_analysis["class_performance"][label]
            print(f"\n{label.upper()}:")
            print(f"  - F1-Score: {class_result.get('f1-score', 'N/A'):.4f}")
            print(f"  - Precision: {class_result.get('precision', 'N/A'):.4f}")
            print(f"  - Recall: {class_result.get('recall', 'N/A'):.4f}")
            print(f"  - Doğru Tahmin: {analysis['correct_predictions']}/{analysis['total_samples']}")
            print(f"  - Öğrenme Kalitesi: {analysis['learning_quality']}")
    
    # Model analizi
    print(f"\n🎯 MODEL ÖĞRENİM ANALİZİ:")
    print(f"En iyi öğrenilen sınıf: {learning_analysis['best_learned_class']}")
    print(f"En zayıf öğrenilen sınıf: {learning_analysis['worst_learned_class']}")
    
    insights = learning_analysis['overall_insights']
    if insights['high_performance_classes']:
        print(f"Yüksek performans sınıfları: {', '.join(insights['high_performance_classes'])}")
    if insights['low_performance_classes']:
        print(f"Düşük performans sınıfları: {', '.join(insights['low_performance_classes'])}")
    
    print(f"\n💾 Tüm görselleştirmeler 'statistics/' klasörüne kaydedildi:")
    print("  - confusion_matrix.png")
    print("  - per_class_performance.png") 
    print("  - prediction_distribution.png")
    print("  - learning_analysis.png")
    print("  - detayli_model_analizi.json")


if __name__ == "__main__":
    main()
