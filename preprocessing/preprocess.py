"""
Data Preprocessing Pipeline for LeadSense AI.

Steps:
  1. Load final_dataset.csv
  2. Clean & de-duplicate
  3. Stratified Train / Validation split (80 / 20)
  4. Save splits as CSV
  5. Return HuggingFace DatasetDict for training
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH     = 128
LABEL_MAP      = {"Hot": 0, "Warm": 1, "Cold": 2}
ID2LABEL       = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------
def load_and_split(dataset_csv: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV, stratify-split into train/val DataFrames."""
    if dataset_csv is None:
        dataset_csv = DATA_DIR / "final_dataset" / "final_dataset.csv"

    if not dataset_csv.exists():
        raise FileNotFoundError(
            f"{dataset_csv} not found. Run `python data/build_dataset.py` first."
        )

    df = pd.read_csv(dataset_csv)
    print(f"Loaded {len(df)} samples from {dataset_csv}")

    # Ensure label_id column
    if "label_id" not in df.columns:
        df["label_id"] = df["label"].map(LABEL_MAP)

    df = df.dropna(subset=["text", "label_id"]).reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)

    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label_id"],
        random_state=42,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")
    print("Train label distribution:\n", train_df["label"].value_counts())
    print("Val label distribution:\n",   val_df["label"].value_counts())

    # Persist splits
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv",     index=False)

    return train_df, val_df


# ---------------------------------------------
def tokenize_datasets(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    tokenizer_name: str = TOKENIZER_NAME,
    max_length: int = MAX_LENGTH,
) -> DatasetDict:
    """Tokenize text columns and return a HF DatasetDict."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _to_hf(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict({
            "text":  df["text"].tolist(),
            "label": df["label_id"].tolist(),
        })

    raw = DatasetDict(train=_to_hf(train_df), validation=_to_hf(val_df))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = raw.map(tokenize_fn, batched=True, batch_size=64)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    print("Tokenization complete.")
    return tokenized, tokenizer


# ---------------------------------------------
def get_class_weights(train_df: pd.DataFrame) -> list[float]:
    """Compute inverse-frequency class weights for CrossEntropy."""
    counts = train_df["label_id"].value_counts().sort_index()
    total  = len(train_df)
    n_cls  = len(counts)
    weights = [total / (n_cls * counts[i]) for i in range(n_cls)]
    print(f"Class weights: {weights}")
    return weights


# ---------------------------------------------
if __name__ == "__main__":
    train_df, val_df = load_and_split()
    tokenized_ds, tok = tokenize_datasets(train_df, val_df)
    weights = get_class_weights(train_df)
    print(tokenized_ds)
