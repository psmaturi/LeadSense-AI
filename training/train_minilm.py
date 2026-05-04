"""
MiniLM Trainer for LeadSense AI -- v2 (Robustness Overhaul).

Model  : sentence-transformers/all-MiniLM-L6-v2
Task   : Sequence Classification (3 classes: Hot / Warm / Cold)

Features:
  [v] Class-weighted CrossEntropy loss
  [v] Label smoothing (default=0.1) -- reduces overconfident shortcut learning
  [v] Early stopping (patience=2)
  [v] Optional encoder layer freezing
  [v] Macro F1 as primary metric
  [v] 5 epochs default (was 4)
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.preprocess import (
    load_and_split,
    tokenize_datasets,
    get_class_weights,
    ID2LABEL,
    LABEL_MAP,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = ROOT / "models" / "minilm_lead_classifier"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------
# Custom Weighted-Loss Trainer
# ---------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer subclass with class-weighted CrossEntropy + label smoothing."""

    def __init__(self, class_weights: list[float], label_smoothing: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights   = torch.tensor(class_weights, dtype=torch.float)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        device  = logits.device
        weights = self.class_weights.to(device)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=self.label_smoothing,  # prevents overconfident shortcuts
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------
# Compute Metrics
# ---------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1}


# ---------------------------------------------
# Optional Encoder Freezing
# ---------------------------------------------
def freeze_encoder_layers(model, n_layers_to_freeze: int = 2):
    """Freeze the first N transformer encoder layers."""
    frozen = 0
    for name, param in model.named_parameters():
        if "encoder.layer." in name:
            layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_idx < n_layers_to_freeze:
                param.requires_grad = False
                frozen += 1
    print(f"Froze {frozen} parameters in first {n_layers_to_freeze} encoder layers.")


# ---------------------------------------------
# Save Evaluation Artefacts
# ---------------------------------------------
def save_confusion_matrix(labels, preds, save_path: Path):
    cm = confusion_matrix(labels, preds)
    class_names = [ID2LABEL[i] for i in range(3)]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title("LeadSense AI -- Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {save_path}")


# ---------------------------------------------
# Main Training Function
# ---------------------------------------------
def train(
    freeze_layers: int = 0,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
):
    print("=" * 60)
    print("  LEADSENSE AI -- MiniLM Training")
    print("=" * 60)

    # -- Data ---------------------------------
    train_df, val_df = load_and_split()
    tokenized_ds, tokenizer = tokenize_datasets(train_df, val_df)
    class_weights = get_class_weights(train_df)

    # -- Model ---------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    if freeze_layers > 0:
        freeze_encoder_layers(model, freeze_layers)

    # -- Training Arguments --------------------
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        learning_rate=lr,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        warmup_steps=50,
        seed=42,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # -- Trainer --------------------------------
    trainer = WeightedTrainer(
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # -- Train ---------------------------------
    trainer.train()

    # -- Evaluate ------------------------------
    print("\nRunning final evaluation ?")
    eval_results = trainer.evaluate()
    print(f"  Accuracy : {eval_results['eval_accuracy']:.4f}")
    print(f"  Macro F1 : {eval_results['eval_macro_f1']:.4f}")

    # -- Confusion Matrix ----------------------
    preds_output  = trainer.predict(tokenized_ds["validation"])
    all_preds     = np.argmax(preds_output.predictions, axis=-1)
    all_labels    = preds_output.label_ids
    save_confusion_matrix(all_labels, all_preds, RESULTS_DIR / "confusion_matrix.png")

    # -- Save Results --------------------------
    from sklearn.metrics import classification_report
    report = classification_report(
        all_labels, all_preds,
        target_names=[ID2LABEL[i] for i in range(3)],
        output_dict=True,
    )
    results = {
        "eval_accuracy": eval_results["eval_accuracy"],
        "eval_macro_f1": eval_results["eval_macro_f1"],
        "classification_report": report,
        "hyperparameters": {
            "model": MODEL_NAME,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "freeze_layers": freeze_layers,
            "label_smoothing": label_smoothing,
        },
    }
    results_path = RESULTS_DIR / "minilm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {results_path}")

    # -- Save Model ----------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Model saved -> {OUTPUT_DIR}")

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LeadSense AI -- MiniLM Trainer v2")
    parser.add_argument("--freeze_layers",   type=int,   default=0)
    parser.add_argument("--epochs",           type=int,   default=5)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--label_smoothing",  type=float, default=0.1,
                        help="Label smoothing for CrossEntropy (0=off, 0.1=recommended)")
    args = parser.parse_args()

    train(
        freeze_layers=args.freeze_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        label_smoothing=args.label_smoothing,
    )
