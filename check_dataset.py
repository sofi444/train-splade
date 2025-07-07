import argparse
from pathlib import Path
from datasets import load_from_disk
import random
import matplotlib.pyplot as plt
import json

def main(data_dir, num_samples=5, save_dir=None):
    data_dir = Path(data_dir)
    print(f"Loading dataset from: {data_dir}")
    dataset = load_from_disk(data_dir)

    # Default: save plots and samples in the same directory as the data
    if save_dir is None:
        save_dir = data_dir
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for split in dataset.keys():
        print(f"\n--- Split: {split} ---")
        ds = dataset[split]
        print(f"Number of examples: {len(ds)}")

        # Show and collect random samples
        print(f"\n{num_samples} random samples:")
        indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
        samples = []
        for i in indices:
            sample = {
                "index": i,
                "anchor": ds[i]['anchor'],
                "positive": ds[i]['positive']
            }
            samples.append(sample)
            print(f"Sample {i}:")
            print(f"  anchor:   {sample['anchor']}")
            print(f"  positive: {sample['positive']}\n")

        # Save samples to JSON
        samples_path = save_dir / f"{split}_samples.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"Random samples saved to {samples_path}")

        # Visualize sentence length distributions
        anchor_lens = [len(s.split()) for s in ds['anchor']]
        positive_lens = [len(s.split()) for s in ds['positive']]

        plt.figure(figsize=(10, 4))
        plt.hist(anchor_lens, bins=50, alpha=0.6, label='anchor')
        plt.hist(positive_lens, bins=50, alpha=0.6, label='positive')
        plt.title(f"Sentence Length Distribution ({split})")
        plt.xlabel("Number of words")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        plot_path = save_dir / f"{split}_lengths.png"
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and visualize processed OPUS-100 dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the processed dataset directory.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to print per split.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the plots and samples. If not set, saves in the data directory.")
    args = parser.parse_args()

    main(args.data_dir, args.num_samples, args.save_dir)
