import argparse
import random
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict

MAX_WORDS = 80

def clean_text(text: str) -> str:
    """
    Cleaning function to apply to all sentences in the dataset.
    """
    text = text.strip("-:").strip()
    return text

def _process_and_format_split(
    dataset_split: Dataset,
    max_length_diff: int,
    num_samples: int = None,
    bidirectional: bool = True
) -> Dataset | None:
    """
    Processes a single split of the OPUS-100 dataset into anchor-positive pairs.
    Optionally includes bidirectional pairs for better bilingual training.
    """
    opus_pairs = []
    
    examples_to_process = dataset_split
    if num_samples:
        if num_samples > len(dataset_split):
            print(f"Warning: Requested {num_samples} samples, but split only has {len(dataset_split)}. Using all available samples.")
        examples_to_process = dataset_split.select(range(min(num_samples, len(dataset_split))))

    for example in examples_to_process:
        eng_sentence = example.get("translation", {}).get("en")
        fra_sentence = example.get("translation", {}).get("fr")

        if isinstance(eng_sentence, str) and isinstance(fra_sentence, str) and eng_sentence and fra_sentence:
            eng_sentence = clean_text(eng_sentence)
            fra_sentence = clean_text(fra_sentence)

            # Skip instances where both sentences are the same
            if eng_sentence == fra_sentence:
                continue

            # Skip if word count difference is too large
            len_en = len(eng_sentence.split())
            len_fr = len(fra_sentence.split())
            if abs(len_en - len_fr) > max_length_diff:
                continue

            # Skip if any of the two sentences is above MAX_WORDS
            if len_en > MAX_WORDS or len_fr > MAX_WORDS:
                continue

            # Add EN->FR pair
            opus_pairs.append([eng_sentence, fra_sentence])
            
            # Add FR->EN pair for bidirectional training
            if bidirectional:
                opus_pairs.append([fra_sentence, eng_sentence])

    if not opus_pairs:
        return None
    
    # Shuffle to mix EN->FR and FR->EN pairs
    random.shuffle(opus_pairs)
    return Dataset.from_dict({
        "anchor": [pair[0] for pair in opus_pairs],
        "positive": [pair[1] for pair in opus_pairs],
    })

def prepare_opus100_data(
    num_pairs: int = 1_000_000,
    output_dir: str = "data/en-fr-opus",
    max_length_diff: int = 7,
    bidirectional: bool = True,
) -> None:
    """
    Downloads and prepares the OPUS-100 English-French dataset for training.

    Fetches the 'en-fr' train, validation, and test splits from the 'Helsinki-NLP/opus-100' dataset, 
    processes them into the expected format for contrastive training with sentence-transformers 
    (Dataset with 'anchor' and 'positive' columns, where each row contains a pair of translated 
    sentences), and saves it to a local dir as a DatasetDict.

    Args:
        num_pairs (int): The total number of translation pairs to select from the original training dataset.
                         The validation and test sets are used in their entirety.
                         If bidirectional=True, the number of pairs will be 2x this.
        output_dir (str): The directory where the processed dataset will be saved.
        max_length_diff (int): The maximum allowed difference in word count between anchor and positive.
        bidirectional (bool): Whether to include both EN->FR and FR->EN pairs.
    """

    print(f"Loading dataset from the hub ('Helsinki-NLP/opus-100', 'en-fr' split)...")
    try:
        full_dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")
        print(f"Successfully loaded dataset with splits: {list(full_dataset.keys())}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dataset_dict = DatasetDict()

    print(f"\nProcessing train split, selecting up to {num_pairs} source pairs...")
    if bidirectional:
        print("Creating bidirectional pairs (EN->FR and FR->EN)...")
    
    train_dataset = _process_and_format_split(
        full_dataset["train"], 
        max_length_diff, 
        num_samples=num_pairs,
        bidirectional=bidirectional
    )
    
    if train_dataset:
        dataset_dict["train"] = train_dataset
        print(f"Created train set with {len(train_dataset)} total pairs.")
    else:
        print("Could not create a train set. Exiting.")
        return

    print("\nProcessing validation split...")
    validation_dataset = _process_and_format_split(full_dataset["validation"], max_length_diff)
    if validation_dataset:
        dataset_dict["validation"] = validation_dataset
        print(f"Created validation set with {len(validation_dataset)} pairs.")
    else:
        print("Validation set could not be created or is empty.")

    print("\nProcessing test split...")
    test_dataset = _process_and_format_split(full_dataset["test"], max_length_diff)
    if test_dataset:
        dataset_dict["test"] = test_dataset
        print(f"Created test set with {len(test_dataset)} pairs.")
    else:
        print("Test set could not be created or is empty.")
    
    print("\nFinal Dataset Structure:")
    print(dataset_dict)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving processed dataset to '{output_path}'...")
    dataset_dict.save_to_disk(output_path)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare OPUS-100 en-fr dataset for SPLADE training."
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=1_000_000,
        help="Number of sentence pairs to process from the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/en-fr-opus",
        help="Directory to save the processed dataset.",
    )
    parser.add_argument(
        "--max_length_diff",
        type=int,
        default=4,
        help="Maximum allowed difference in word count between anchor and positive. Pairs with a larger difference are excluded.",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Whether to include both EN->FR and FR->EN pairs.",
    )
    args = parser.parse_args()

    prepare_opus100_data(
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        max_length_diff=args.max_length_diff,
        bidirectional=args.bidirectional,
    )

""" To clean:
Sample 282148:
  anchor:   "
  positive: ".
"""