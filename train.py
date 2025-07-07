import argparse
import os
from pathlib import Path
import torch
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderTrainingArguments,
    SparseEncoderTrainer,
)
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.sparse_encoder.losses import (
    SparseMultipleNegativesRankingLoss,
    SparseCosineSimilarityLoss,
    SpladeLoss,
)
from datasets import load_from_disk, DatasetDict
from dotenv import load_dotenv

load_dotenv()

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()

OUTPUT_MODEL_ID = "splade-en-fr-eurobert-v1"
OUTPUT_DIR = "models/splade-en-fr-eurobert-v1"

def train_splade_model(
    data_path: str = "data/en-fr-opus",
    base_model: str = "EuroBERT/EuroBERT-210m",
    output_dir: str = OUTPUT_DIR,
    epochs: int = 1,
    train_batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.2,
    query_reg_weight: float = 5e-5,
    doc_reg_weight: float = 3e-5,
) -> None:
    """
    Trains a SPLADE model on a properly formatted dataset.

    Args:
        data_path (str): Path to the directory containing the processed dataset.
        base_model (str): The name of the model to use as the base (must be MLM).
        output_dir (str): Directory to save training checkpoints and the final model.
        epochs (int): Number of training epochs.
        train_batch_size (int): Batch size for training.
        learning_rate (float): The learning rate for the optimizer.
        warmup_ratio (float): The ratio of training steps to use for a linear warmup.
        query_reg_weight (float): Sparsity regularization weight for queries.
        doc_reg_weight (float): Sparsity regularization weight for documents.
    """
    ### --- Data ---
    dataset_path = Path(data_path)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at '{data_path}'.")
        print("Please run the 'prepare_data.py' script first.")
        return
        
    print(f"Loading dataset from '{dataset_path}'...")
    training_dataset_dict = load_from_disk(dataset_path)
    
    if not isinstance(training_dataset_dict, DatasetDict) or "train" not in training_dataset_dict:
        print("Error: Invalid dataset format. Expected a DatasetDict with a 'train' split.")
        return

    print("Dataset loaded successfully:")
    print(training_dataset_dict)

    eval_dataset = training_dataset_dict.get("validation")
    if eval_dataset:
        print("Validation set found and will be used for evaluation during training.")

    ### --- Model ---
    print(f"Initializing model with base: '{base_model}'")
    # Initialize the SPLADE architecture: MLMTransformer + SpladePooling
    # The base model must have a Masked Language Modeling (MLM) head.
    transformer = MLMTransformer(base_model)
    # The SpladePooling layer handles the aggregation and activation.
    pooler = SpladePooling(pooling_strategy="max")
    model = SparseEncoder(modules=[transformer, pooler])
    # or also directly:
    # model = SparseEncoder.from_pretrained(base_model)
    # MLMTransformer + SpladePooling is the default SPLADE architecture

    # Move model to GPU if available
    if CUDA_AVAILABLE:
        print("Moving model to CUDA device.")
        model = model.to("cuda")
    elif MPS_AVAILABLE:
        print("Moving model to MPS device.")
        model = model.to("mps")
    else:
        print("Warning: No CUDA or MPS device found. Training on CPU.")

    ### --- Loss ---
    # The primary loss is a contrastive loss that learns to differentiate
    # between positive and in-batch negative pairs.
    # primary_loss = SparseMultipleNegativesRankingLoss(model=model)
    
    primary_loss = SparseCosineSimilarityLoss(model=model)
    
    # SpladeLoss wraps the primary loss and adds the L1 sparsity regularization term.
    # This is what makes the embeddings sparse.
    splade_loss = SpladeLoss(
        model=model,
        loss=primary_loss,
        query_regularizer_weight=query_reg_weight,
        document_regularizer_weight=doc_reg_weight,
    )
    print(f"Loss function configured with {primary_loss.__class__.__name__}.")

    ### --- Training ---
    # These arguments control every aspect of the training loop.
    args = SparseEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        #fp16=USE_FP16, # base model requires fp32
        logging_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2, # Only keep the last 2 checkpoints
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=f"sofdog/{OUTPUT_MODEL_ID}",
        hub_token=os.getenv("HF_TOKEN"),
    )
    print(f"Training arguments set. Output will be saved to '{output_dir}'.")

    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=training_dataset_dict["train"],
        eval_dataset=eval_dataset,
        loss=splade_loss,
    )
    
    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Done! ---")

    ### --- Save ---
    final_model_path = f"{output_dir}-final"
    model.save_pretrained(final_model_path)
    print(f"Final model saved to '{final_model_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a bilingual SPLADE model.")
    parser.add_argument("--data_path", type=str, default="data/en-fr-opus", help="Path to the prepared dataset.")
    parser.add_argument("--base_model", type=str, default="EuroBERT/EuroBERT-210m", help="Base MLM model.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")

    args = parser.parse_args()

    train_splade_model(
        data_path=args.data_path, 
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
