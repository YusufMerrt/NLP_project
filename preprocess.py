import os

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


def convert_to_conll_format(dataset, split, output_file):
    """Convert sentence pairs to CONLL format for entailment classification"""
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset[split]:
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            label = example["label"]
            
            # Write premise-hypothesis pair separated by [SEP]
            combined_text = f"{premise} [SEP] {hypothesis}"
            tokens = combined_text.split()
            
            for token in tokens:
                f.write(f"{token}\t{label}\n")
            f.write("\n")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Load Turkish NLI dataset (SNLI-TR) - using only validation and test sets for smaller dataset
    print("Loading Turkish SNLI dataset (smaller subset)...")
    raw_dataset = load_dataset("boun-tabi/nli_tr", "snli_tr", trust_remote_code=True)
    
    # Use only validation and test sets (10K + 10K = 20K examples total)
    print("Using only validation and test sets to create a smaller dataset...")

    # Convert validation and test to pandas DataFrame
    validation_df = pd.DataFrame(raw_dataset["validation"])
    test_df = pd.DataFrame(raw_dataset["test"])

    # Split validation set into train (70%) and validation (30%)
    val_train_df, val_val_df = train_test_split(validation_df, test_size=0.3, random_state=42, stratify=validation_df['label'])
    
    # Use test set as test
    test_df = pd.DataFrame(raw_dataset["test"])
    
    print(f"New dataset sizes:")
    print(f"  - Train: {len(val_train_df)} examples (from original validation set)")
    print(f"  - Validation: {len(val_val_df)} examples (from original validation set)")
    print(f"  - Test: {len(test_df)} examples (original test set)")
    
    # Map labels to our format
    label_mapping = {
        0: "entailment",    # entailment -> entailment
        1: "neutral",       # neutral -> neutral  
        2: "contradiction"  # contradiction -> contradiction
    }
    
    # Apply label mapping
    for df in [val_train_df, val_val_df, test_df]:
        df["label"] = df["label"].map(label_mapping)

    # Convert back to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(val_train_df)
    validation_dataset = Dataset.from_pandas(val_val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Create a DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}
    )

    # Convert each split to CONLL format and save
    for split in dataset_dict:
        output_file = f"data/{split}.conll"
        convert_to_conll_format(dataset_dict, split, output_file)
        print(f"{split} split converted to CONLL format and saved to {output_file}")
        print(f"  - Examples: {len(dataset_dict[split])}")

    print("\nDataset conversion completed!")
    print("Labels: entailment, neutral, contradiction")
    print(f"Total examples: {len(train_dataset) + len(validation_dataset) + len(test_dataset)}")
