import json
import os

from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def read_conll(file_path):
    """Read CONLL format data containing sentence pairs for entailment"""
    sentences = []
    labels = []
    sentence = []
    label = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip() == "":
                if sentence:
                    full_text = " ".join(sentence)
                    # Split premise and hypothesis at [SEP] token
                    if "[SEP]" in full_text:
                        parts = full_text.split("[SEP]")
                        premise = parts[0].strip()
                        hypothesis = parts[1].strip() if len(parts) > 1 else ""
                        
                        # Store as combined text for BERT sentence pair processing
                        sentences.append(f"{premise}[SEP]{hypothesis}")
                    labels.append(label[0])
                    sentence = []
                    label = []
            else:
                token, tag = line.strip().split()
                sentence.append(token)
                label.append(tag)
        if sentence:
            full_text = " ".join(sentence)
            if "[SEP]" in full_text:
                parts = full_text.split("[SEP]")
                premise = parts[0].strip()
                hypothesis = parts[1].strip() if len(parts) > 1 else ""
                sentences.append(f"{premise}[SEP]{hypothesis}")
            labels.append(label[0])
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
    
    # Save the evaluation results
    os.makedirs("statistics", exist_ok=True)
    with open("statistics/egitim_sonuclari.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation results saved to statistics/egitim_sonuclari.json")
    
    # Print key metrics
    print(f"\nKey Results:")
    print(f"Accuracy: {results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Macro F1: {results.get('eval_macro avg', {}).get('f1-score', 'N/A'):.4f}")
    print(f"Weighted F1: {results.get('eval_weighted avg', {}).get('f1-score', 'N/A'):.4f}")
    
    # Print per-class results
    for i, label in enumerate(label_list):
        if f"eval_{i}" in results:
            class_result = results[f"eval_{i}"]
            print(f"{label}: F1={class_result.get('f1-score', 'N/A'):.4f}, "
                  f"Precision={class_result.get('precision', 'N/A'):.4f}, "
                  f"Recall={class_result.get('recall', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
