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
    
    # Load Turkish NLI dataset (SNLI-TR) - using subset for 100K examples from 570K total
    print("Loading Turkish SNLI dataset (570K total available)...")
    raw_dataset = load_dataset("boun-tabi/nli_tr", "snli_tr", trust_remote_code=True)
    
    # Take 100K examples from the large train set (550K available)
    print("Taking 100K examples from the full dataset...")

    # Convert train set to pandas and sample 100K examples
    full_train_df = pd.DataFrame(raw_dataset["train"])
    
    # Sample 100K examples from 550K train data, maintaining label balance
    train_sample_df = full_train_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 100000//3), random_state=42)
    ).reset_index(drop=True)
    
    # If we don't have enough balanced samples, just take first 100K
    if len(train_sample_df) < 100000:
        train_sample_df = full_train_df.sample(n=100000, random_state=42)
    
    # Split sampled data into train (80%) and validation (20%)
    train_df, val_df = train_test_split(train_sample_df, test_size=0.2, random_state=42, 
                                       stratify=train_sample_df['label'])
    
    # Use original test set
    test_df = pd.DataFrame(raw_dataset["test"])
    
    print(f"Sampled dataset sizes (from 570K total):")
    print(f"  - Train: {len(train_df)} examples (from 550K train set)")
    print(f"  - Validation: {len(val_df)} examples (from 550K train set)")
    print(f"  - Test: {len(test_df)} examples (original test set)")
    print(f"  - Total training data: {len(train_df) + len(val_df)} examples")
    
    # Map labels to our format
    label_mapping = {
        0: "entailment",    # entailment -> entailment
        1: "neutral",       # neutral -> neutral  
        2: "contradiction"  # contradiction -> contradiction
    }
    
    # Apply label mapping
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].map(label_mapping)

    # Convert back to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(val_df)
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
