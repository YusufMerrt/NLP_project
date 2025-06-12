import os
from collections import Counter

import matplotlib.pyplot as plt


def read_conll_data(file_path):
    """Read CONLL data containing sentence pairs for entailment"""
    sentence_pairs = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as file:
        current_pair = []
        current_label = []
        for line in file:
            if line.strip() == "":
                if current_pair:
                    # Reconstruct the sentence pair
                    full_text = " ".join(current_pair)
                    if "[SEP]" in full_text:
                        parts = full_text.split("[SEP]")
                        premise = parts[0].strip()
                        hypothesis = parts[1].strip() if len(parts) > 1 else ""
                        sentence_pairs.append((premise, hypothesis))
                        labels.append(current_label[0] if current_label else "unknown")
                    current_pair = []
                    current_label = []
            else:
                token, tag = line.strip().split()
                current_pair.append(token)
                current_label.append(tag)
        
        # Handle last pair if file doesn't end with empty line
        if current_pair:
            full_text = " ".join(current_pair)
            if "[SEP]" in full_text:
                parts = full_text.split("[SEP]")
                premise = parts[0].strip()
                hypothesis = parts[1].strip() if len(parts) > 1 else ""
                sentence_pairs.append((premise, hypothesis))
                labels.append(current_label[0] if current_label else "unknown")
    
    return sentence_pairs, labels


def dataset_statistics(sentence_pairs, labels):
    """Calculate statistics for entailment dataset"""
    num_pairs = len(sentence_pairs)
    
    # Calculate token counts for premises and hypotheses
    premise_tokens = []
    hypothesis_tokens = []
    
    for premise, hypothesis in sentence_pairs:
        premise_tokens.extend(premise.split())
        hypothesis_tokens.extend(hypothesis.split())
    
    # Calculate lengths
    premise_lengths = [len(premise.split()) for premise, _ in sentence_pairs]
    hypothesis_lengths = [len(hypothesis.split()) for _, hypothesis in sentence_pairs]
    
    total_tokens = len(premise_tokens) + len(hypothesis_tokens)
    label_counter = Counter(labels)

    stats = {
        "Cümle çifti sayısı": num_pairs,
        "Toplam premise kelimesi": len(premise_tokens),
        "Toplam hypothesis kelimesi": len(hypothesis_tokens),
        "Toplam kelime sayısı": total_tokens,
        "Ortalama premise uzunluğu": sum(premise_lengths) / len(premise_lengths) if premise_lengths else 0,
        "Ortalama hypothesis uzunluğu": sum(hypothesis_lengths) / len(hypothesis_lengths) if hypothesis_lengths else 0,
        "Maksimum premise uzunluğu": max(premise_lengths) if premise_lengths else 0,
        "Maksimum hypothesis uzunluğu": max(hypothesis_lengths) if hypothesis_lengths else 0,
        "Minimum premise uzunluğu": min(premise_lengths) if premise_lengths else 0,
        "Minimum hypothesis uzunluğu": min(hypothesis_lengths) if hypothesis_lengths else 0,
        "Etiket dağılımı": dict(label_counter),
    }

    return stats


def save_statistics(stats, stats_path):
    """Save statistics to file"""
    os.makedirs(stats_path, exist_ok=True)
    with open(
        os.path.join(stats_path, "istatistikler.txt"), "w", encoding="utf-8"
    ) as f:
        for key, value in stats.items():
            if key == "Etiket dağılımı":
                f.write(f"{key}:\n")
                for tag, count in value.items():
                    f.write(f"  {tag}: {count}\n")
            else:
                f.write(f"{key}: {value}\n")


def plot_tag_distribution(stats, stats_path):
    """Plot label distribution for entailment classes"""
    tag_counter = stats["Etiket dağılımı"]

    tags = list(tag_counter.keys())
    counts = list(tag_counter.values())

    plt.figure(figsize=(10, 6))
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    bars = plt.bar(tags, counts, color=colors[:len(tags)])
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom')
    
    plt.xlabel("Etiketler")
    plt.ylabel("Sayılar")
    plt.title("Entailment Etiket Dağılımı")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(stats_path, "etiket_dagilimi.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_length_distributions(stats, stats_path):
    """Plot length distributions for premises and hypotheses"""
    # This function could be extended to show actual length distributions
    # For now, we'll create a simple comparison chart
    
    avg_premise = stats["Ortalama premise uzunluğu"]
    avg_hypothesis = stats["Ortalama hypothesis uzunluğu"]
    max_premise = stats["Maksimum premise uzunluğu"]
    max_hypothesis = stats["Maksimum hypothesis uzunluğu"]
    
    categories = ['Ortalama Uzunluk', 'Maksimum Uzunluk']
    premise_values = [avg_premise, max_premise]
    hypothesis_values = [avg_hypothesis, max_hypothesis]
    
    x = range(len(categories))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], premise_values, width, label='Premise', alpha=0.8)
    plt.bar([i + width/2 for i in x], hypothesis_values, width, label='Hypothesis', alpha=0.8)
    
    plt.xlabel('Ölçüm Türü')
    plt.ylabel('Kelime Sayısı')
    plt.title('Premise ve Hypothesis Uzunluk Karşılaştırması')
    plt.xticks(x, categories)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(stats_path, "uzunluk_karsilastirmasi.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_datasets(dataset_paths):
    """Analyze all datasets and generate statistics"""
    for dataset_path in dataset_paths:
        print(f"Analyzing {dataset_path}...")
        sentence_pairs, labels = read_conll_data(dataset_path)
        stats = dataset_statistics(sentence_pairs, labels)
        split_name = os.path.basename(dataset_path).replace(".conll", "")
        split_stats_path = os.path.join("statistics/data_stats", f"{split_name}_stats")
        
        save_statistics(stats, split_stats_path)
        plot_tag_distribution(stats, split_stats_path)
        plot_length_distributions(stats, split_stats_path)
        
        print(f"  - {stats['Cümle çifti sayısı']} sentence pairs")
        print(f"  - Label distribution: {stats['Etiket dağılımı']}")
        print(f"  - Statistics saved to {split_stats_path}")

    # Generate combined statistics
    print("\nGenerating combined statistics...")
    all_sentence_pairs = []
    all_labels = []
    
    for dataset_path in dataset_paths:
        pairs, labels = read_conll_data(dataset_path)
        all_sentence_pairs.extend(pairs)
        all_labels.extend(labels)
    
    all_stats = dataset_statistics(all_sentence_pairs, all_labels)
    save_statistics(all_stats, "statistics/data_stats/all_stats")
    plot_tag_distribution(all_stats, "statistics/data_stats/all_stats")
    plot_length_distributions(all_stats, "statistics/data_stats/all_stats")
    
    print(f"Combined statistics:")
    print(f"  - Total sentence pairs: {all_stats['Cümle çifti sayısı']}")
    print(f"  - Overall label distribution: {all_stats['Etiket dağılımı']}")
    print("All statistics saved successfully!")


if __name__ == "__main__":
    dataset_paths = ["data/train.conll", "data/validation.conll", "data/test.conll"]
    analyze_datasets(dataset_paths)
