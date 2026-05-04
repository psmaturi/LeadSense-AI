"""
Baseline Comparison: TF-IDF + Logistic Regression vs MiniLM.

Why classical ML sometimes wins:
  - Very small datasets: simpler model avoids overfitting
  - Short, keyword-heavy text: TF-IDF captures features well
  - Speed-critical inference pipelines
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ID2LABEL = {0: "Hot", 1: "Warm", 2: "Cold"}


def train_baseline() -> dict:
    """Train TF-IDF + LR and return evaluation metrics."""
    # -- Load splits ----------------------------
    train_path = DATA_DIR / "train.csv"
    val_path   = DATA_DIR / "val.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "train.csv / val.csv not found. Run preprocessing/preprocess.py first."
        )

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    X_train, y_train = train_df["text"].tolist(), train_df["label_id"].tolist()
    X_val,   y_val   = val_df["text"].tolist(),   val_df["label_id"].tolist()

    # -- Pipeline -------------------------------
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            multi_class="multinomial",
        )),
    ])

    print("Training TF-IDF + Logistic Regression baseline ?")
    pipeline.fit(X_train, y_train)

    # -- Evaluate -------------------------------
    y_pred = pipeline.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    print(f"\nBaseline Results:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_val, y_pred, target_names=["Hot", "Warm", "Cold"]
    ))

    # -- Confusion Matrix -----------------------
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=["Hot", "Warm", "Cold"],
        yticklabels=["Hot", "Warm", "Cold"],
        ax=ax,
    )
    ax.set_title("Baseline (TF-IDF + LR) -- Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "baseline_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix -> {cm_path}")

    # -- Save model -----------------------------
    model_path = MODELS_DIR / "baseline_tfidf_lr.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Baseline model saved -> {model_path}")

    # -- Save results ---------------------------
    report = classification_report(
        y_val, y_pred, target_names=["Hot", "Warm", "Cold"], output_dict=True
    )
    results = {
        "model":     "TF-IDF + Logistic Regression",
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "classification_report": report,
    }
    results_path = RESULTS_DIR / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def compare_results():
    """Print a side-by-side comparison of baseline vs. MiniLM."""
    baseline_path = RESULTS_DIR / "baseline_results.json"
    minilm_path   = RESULTS_DIR / "minilm_results.json"

    results = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            results["baseline"] = json.load(f)
    if minilm_path.exists():
        with open(minilm_path) as f:
            results["minilm"] = json.load(f)

    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(f"{'Metric':<20} {'TF-IDF+LR':>15} {'MiniLM':>15}")
    print("-" * 55)

    b = results.get("baseline", {})
    m = results.get("minilm",   {})

    print(f"{'Accuracy':<20} {b.get('accuracy', 'N/A'):>15.4f} {m.get('eval_accuracy', 'N/A'):>15.4f}")
    print(f"{'Macro F1':<20} {b.get('macro_f1', 'N/A'):>15.4f} {m.get('eval_macro_f1', 'N/A'):>15.4f}")
    print("=" * 55)
    print("\nWhen TF-IDF+LR outperforms transformers:")
    print("  -> Very small datasets (<500 samples)")
    print("  -> Short, keyword-dominated text")
    print("  -> Inference speed is a hard constraint")
    print("  -> No GPU available for fine-tuning")


if __name__ == "__main__":
    train_baseline()
    compare_results()
