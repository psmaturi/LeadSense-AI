import sys
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.hybrid_predictor import predict

def run_fresh_audit():
    print("=" * 60)
    print("  LEADSENSE AI -- FRESH OOD AUDIT (FINAL VERIFICATION)")
    print("=" * 60)

    test_path = ROOT / "data" / "fresh_test_data" / "fresh_test_data.csv"
    if not test_path.exists():
        print(f"Error: Fresh test file {test_path} not found.")
        return

    df = pd.read_csv(test_path)
    
    true_labels = []
    pred_labels = []
    failures = []

    print(f"Running inference on {len(df)} samples...")
    
    for _, row in df.iterrows():
        text = row['text']
        expected = row['label']
        
        result = predict(text)
        predicted = result['label']
        
        true_labels.append(expected)
        pred_labels.append(predicted)
        
        if predicted != expected:
            failures.append({
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "method": result['method'],
                "confidence": result['confidence']
            })

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    
    print(f"\nRESULTS:")
    print(f"  Accuracy : {acc:.4f} ({int(acc*len(df))}/{len(df)})")
    print(f"  Macro F1 : {f1:.4f}")
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(true_labels, pred_labels, labels=["Hot", "Warm", "Cold"]))
    
    if failures:
        print(f"\nFAILED PREDICTIONS ({len(failures)}):")
        for fail in failures:
            print(f"  [{fail['expected']} -> {fail['predicted']}] (via {fail['method']}, conf={fail['confidence']:.2f})")
            print(f"  Text: {fail['text']}")
            print("-" * 30)

    # Save results
    audit_results = {
        "accuracy": acc,
        "macro_f1": f1,
        "failures": failures
    }
    
    with open(ROOT / "evaluation" / "final_results" / "fresh_audit_results.json", "w") as f:
        json.dump(audit_results, f, indent=2)
    
    print(f"\nAudit results saved to evaluation/final_results/fresh_audit_results.json")

if __name__ == "__main__":
    run_fresh_audit()
