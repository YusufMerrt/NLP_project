import os

import ssl
import certifi
import torch
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def preprocess_sentence_pairs(premise_list, hypothesis_list, tokenizer, max_length=128):
    """Preprocess sentence pairs for entailment classification"""
    inputs = tokenizer(
        premise_list,
        hypothesis_list,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs

def predict_entailment(premise_list, hypothesis_list, model, tokenizer, label_encoder):
    """Predict entailment relationships for sentence pairs"""
    model.eval()
    inputs = preprocess_sentence_pairs(premise_list, hypothesis_list, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_labels = label_encoder.inverse_transform(predictions.cpu().numpy())
    return predicted_labels

def write_conll_format(premise_list, hypothesis_list, predicted_labels, output_file):
    """Write entailment predictions to CONLL format"""
    with open(output_file, "w", encoding="utf-8") as f:
        for premise, hypothesis, label in zip(premise_list, hypothesis_list, predicted_labels):
            # Combine premise and hypothesis with [SEP] token
            combined_text = f"{premise} [SEP] {hypothesis}"
            tokens = combined_text.split()
            for token in tokens:
                f.write(f"{token}\t{label}\n")
            f.write("\n")

def read_sentence_pairs(file_path):
    """Read sentence pairs from input file
    
    Expected format:
    Line 1: Premise
    Line 2: Hypothesis  
    Line 3: (empty line or next premise)
    Line 4: Next hypothesis
    ...
    
    OR simply alternating lines:
    Premise 1
    Hypothesis 1
    Premise 2
    Hypothesis 2
    ...
    """
    premises = []
    hypotheses = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
    
    # If even number of lines, assume alternating premise-hypothesis
    if len(lines) % 2 == 0:
        for i in range(0, len(lines), 2):
            premises.append(lines[i])
            hypotheses.append(lines[i + 1])
    else:
        # If odd number, process what we can and warn
        print(f"Warning: Odd number of lines ({len(lines)}). Processing pairs and ignoring last line.")
        for i in range(0, len(lines) - 1, 2):
            premises.append(lines[i])
            hypotheses.append(lines[i + 1])
    
    return premises, hypotheses

def main():
    model_path = "model"
    os.makedirs("girdi_cikti", exist_ok=True)
    input_file = "girdi_cikti/girdi.txt"
    output_file = "girdi_cikti/cikti.conll"

    # Entailment labels
    label_list = ["entailment", "neutral", "contradiction"]
    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)

    print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(model_path)

    print("Reading sentence pairs from input file...")
    premises, hypotheses = read_sentence_pairs(input_file)
    
    if not premises:
        print("No sentence pairs found in input file!")
        print("Please provide sentence pairs in the following format:")
        print("Line 1: Premise 1")
        print("Line 2: Hypothesis 1")
        print("Line 3: Premise 2")
        print("Line 4: Hypothesis 2")
        print("...")
        return
    
    print(f"Found {len(premises)} sentence pairs")
    
    print("Predicting entailment relationships...")
    predicted_labels = predict_entailment(premises, hypotheses, model, tokenizer, label_encoder)

    print("Writing results to CONLL format...")
    write_conll_format(premises, hypotheses, predicted_labels, output_file)
    
    # Print results summary
    print(f"\nResults Summary:")
    print(f"Processed {len(premises)} sentence pairs")
    
    # Count predictions
    from collections import Counter
    label_counts = Counter(predicted_labels)
    for label, count in label_counts.items():
        print(f"{label}: {count} pairs")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('punkt')
    ssl._create_default_https_context = ssl.create_default_context

    main()
