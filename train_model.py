import json
import os
import shutil

import numpy as np
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainerCallback, TrainingArguments)


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
    splits = ["train", "validation", "test"]
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

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"eval_accuracy": acc["accuracy"], "eval_f1": f1["f1"]}

class SaveCheckpointAndDeleteOldCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.global_step > 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)

def main():
    print("Training Turkish Entailment Classification Model")
    model_checkpoint = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load the labels and create a label encoder for entailment task
    label_list = ["entailment", "neutral", "contradiction"]
    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)

    dataset_dict = create_dataset_dict(label_encoder)

    tokenized_datasets = dataset_dict.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(label_list)
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=None,
        report_to=None,  # TensorBoard logging'i devre dışı bırak
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=len(label_list)
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[SaveCheckpointAndDeleteOldCallback()],
    )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
        }

    best_run = trainer.hyperparameter_search(
        direction="maximize", hp_space=hp_space, n_trials=1
    )

    results_dir = "results"
    evaluation_results = {}
    best_f1 = 0
    best_run = None
    for run in os.listdir(results_dir):
        last_checkpoint = os.listdir(os.path.join(results_dir, run))[-1]
        run_path = os.path.join(results_dir, run, last_checkpoint)
        print(run_path)
        if os.path.isdir(run_path):
            print(f"Evaluating run: {run}")
            model = AutoModelForSequenceClassification.from_pretrained(run_path)
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            results = trainer.evaluate(tokenized_datasets["validation"])
            evaluation_results[run] = results
            if results["eval_f1"] > best_f1:
                best_f1 = results["eval_f1"]
                best_run = run

    os.makedirs("statistics", exist_ok=True)
    with open("statistics/hyperparameter_tuning_sonuclari.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

    with open("statistics/best_run.txt", "w") as f:
        f.write(best_run)

    print(f"Best run is {best_run} with F1 score: {best_f1}")

    last_checkpoint = os.listdir(os.path.join(results_dir, best_run))[-1]
    best_model_path = f"results/{best_run}/{last_checkpoint}"
    print(f"Using best model from {best_model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

if __name__ == "__main__":
    main()
