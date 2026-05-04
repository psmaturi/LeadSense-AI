"""
Hybrid Predictor for LeadSense AI.

Architecture:
  1. Rule-Based Layer (high-confidence keyword overrides)
  2. MiniLM Classifier (neural fallback)
  3. Low-confidence guard -> baseline TF-IDF+LR as secondary fallback

This prevents confident misclassifications for strongly-signaled text.
"""

import re
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ID2LABEL = {0: "Hot", 1: "Warm", 2: "Cold"}
LABEL2ID = {"Hot": 0, "Warm": 1, "Cold": 2}


# ---------------------------------------------
# Rule-Based Layer
# ---------------------------------------------
RULES: dict[str, list[str]] = {
    "Cold": [
        # Strong rejection / disengagement signals
        r"\bnot\s+interested\b",
        r"\bno longer interested\b",
        r"\bplease\s+remove\s+me\b",
        r"\bunsubscribe\b",
        r"\bopt[\s-]?out\b",
        r"\bdo not\s+contact\b",
        r"\bstop\s+emailing\b",
        r"\bnot\s+the\s+right\s+(time|fit|moment)\b",
        r"\balready\s+(have|using|signed|selected)\b",
        r"\bgo(ing)?\s+with\s+(a\s+)?(another|different|competitor|someone\s+else)\b",
        r"\b(selected|chose|signed\s+with)\s+(a\s+)?competitor\b",
        r"\bclosed[\s-]lost\b",
        r"\bno\s+budget\b",
        r"\bbudget\s+freeze\b",
        r"\bpurchase\s+freeze\b",
        r"\brestructuring\b",
        r"\bnever\s+(activated|logged in|responded)\b",
        r"\b90[\s-]day[s]?\s+silent\b",
        r"\bmark.*archive\b",
        # Informal rejection signals
        r"\bpricing.*(too\s+high|expensive|out\s+of\s+reach)\b",
        r"\b(not|won'?t|not\s+be)\s+(moving|going)\s+forward\b",
        r"\bselected\s+(another|a\s+different)\s+vendor\b",
        r"\bno\s+longer\s+(interested|needed)\b",
        r"\bnot\s+proceeding\b",
        r"\bdecided\s+not\s+to\s+proceed\b",
        # Cancel / account closure signals
        r"\bcancel\b",
        r"\bdelete\s+(my\s+)?account\b",
        r"\bclose\s+(my\s+)?account\b",
        # Build in-house / alternative decisions
        r"\bbuild\s+(it\s+)?(in-house|internally|ourselves)\b",
        r"\b(project|initiative)\s+(has\s+been\s+)?(shelved|cancelled|paused|killed)\b",
        r"\bwon'?t\s+be\s+(buying|purchasing|proceeding|moving)\b",
        r"\bdon'?t\s+(need|want)\s+(this|it|your)\b",
        r"\b(went|going)\s+with\s+(someone|somebody)\s+else\b",
        r"\bno\s+(longer|more)\s+(need|evaluating|considering)\b",
        r"\bghosting\b",
        r"\bdead\s+(lead|end|account)\b",
        r"\bclosed[\s-]?(lost|dead)\b",
        r"\bwent\s+(dark|silent|cold)\b",
        r"\bstop\s+(contacting|reaching|following)\b",
        r"\bdo\s+not\s+(call|email|contact)\b",
        r"\bremove\s+(us|me|from)\b",
    ],
    "Hot": [
        # Strong buying / closing signals
        r"\bready\s+to\s+(sign|finalize|commit|buy|purchase|onboard)\b",
        r"\bsend\s+(the\s+)?(contract|sow|proposal|po|invoice)\b",
        r"\bpurchase\s+order\s+is\s+ready\b",
        r"\bbudget\s+(approved|signed off|unlocked|cleared|greenlit)\b",
        r"\bboard\s+approval\b",
        r"\bfinal\s+(decision|step|review|approval)\b",
        r"\bexecutive\s+(signed|approved|greenlit)\b",
        r"\bclose\s+(this\s+)?(deal|contract|agreement)\b",
        r"\b(pricing|quote|contract|agreement|proposal|po|invoice|sow).*asap\b",
        r"\basap.*(pricing|quote|contract|agreement|proposal|po|invoice|sow)\b",
        r"\b(quote|pricing|agreement|contract).*urgent\b",
        r"\burgent.*(quote|pricing|agreement|contract)\b",
        r"\bboss\s+asking\b",
        r"\bmanager\s+approval\b",
        r"\bimmediately?\b.*\bstart\b",
        r"\bstart\s+immediately\b",
        r"\bpo\s+ready\b",
        r"\bcfo\s+(approved|greenlit|signed)\b",
        r"\blegal\s+(team\s+)?(is\s+)?(review(ing)?|cleared|approved|signed)\b",
        r"\bprocurement\s+(cleared|ready|last stage)\b",
        r"\bpo\s+(link|ready|sent|raised)\b",
        r"\bneed\s+to\s+buy\b",
        r"\bwant\s+to\s+buy\b",
        r"\bbuy\s+this\b",
        r"\bbuy\s+now\b",
        r"\bbuy\b",
    ],
    "Warm": [
        # Evaluation / research signals
        r"\b(still\s+)?(evaluating|comparing|exploring|researching|looking\s+at)\b",
        r"\bnext\s+(quarter|month|year|q[1-4]|fy)\b",
        r"\b(internal|team|board)\s+(approval|review|alignment|discussion|buy-in|sign-off)\b",
        r"\bteam\s+(needs|alignment|review|discussion)\b",
        r"\bstill\s+(deciding|considering|evaluating)\b",
        r"\bnot\s+ready\s+to\s+(commit|decide|buy|sign)\b",
        r"\b(discovery|demo|pilot|trial)\b",
        r"\bfollow\s+up\b",
        r"\brevisit\b",
        r"\bcheck\s+(back|in)\s+(later|next|soon)\b",
        r"\breconnect\b",
        r"\bstay\s+in\s+touch\b",
        r"\bmoving\s+forward\b",
        r"\binterested\s+but\b",
        r"\bnot\s+fully\s+convinced\b",
        r"\bquick\s+demo\b",
    ],
}

_compiled_rules: dict[str, list[re.Pattern]] = {
    label: [re.compile(p, re.IGNORECASE) for p in patterns]
    for label, patterns in RULES.items()
}


def rule_based_classify(text: str) -> Optional[str]:
    """
    Return label if any high-confidence rule matches, else None.
    Priority: Cold > Warm > Hot (so negative signals always win).
    """
    for label in ("Cold", "Hot", "Warm"):
        for pattern in _compiled_rules[label]:
            if pattern.search(text):
                return label
    return None


# ---------------------------------------------
# Model Loader (lazy singleton)
# ---------------------------------------------
_model_cache: dict = {}


def _load_model():
    if "model" not in _model_cache:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = ROOT / "models" / "minilm_lead_classifier"
        if not model_dir.exists():
            raise RuntimeError(
                f"Model not found at {model_dir}. Run training/train_minilm.py first."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        _model_cache["model"]     = model
        _model_cache["tokenizer"] = tokenizer
        _model_cache["device"]    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(_model_cache["device"])
        print("MiniLM model loaded.")
    return _model_cache["model"], _model_cache["tokenizer"], _model_cache["device"]


def _load_baseline():
    if "baseline" not in _model_cache:
        path = ROOT / "models" / "baseline_tfidf_lr.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _model_cache["baseline"] = pickle.load(f)
            print("Baseline model loaded as secondary fallback.")
    return _model_cache.get("baseline")


# ---------------------------------------------
# Neural Prediction
# ---------------------------------------------
def neural_predict(text: str) -> tuple[str, float]:
    """Run MiniLM and return (label, confidence)."""
    model, tokenizer, device = _load_model()
    inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs  = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    top_id = int(np.argmax(probs))
    return ID2LABEL[top_id], float(probs[top_id])


# ---------------------------------------------
# Hybrid Predict (Public API)
# ---------------------------------------------
def predict(text: str, confidence_threshold: float = 0.55) -> dict:
    """Legacy wrapper for simple label/conf prediction."""
    res = predict_full(text, confidence_threshold)
    return {
        "label": res["label"],
        "confidence": res["confidence"],
        "method": res["method"]
    }

def predict_full(text: str, confidence_threshold: float = 0.55, raw_text: str = None) -> dict:
    """
    Full hybrid prediction pipeline returning probabilities.
    
    Args:
        text: The text to classify (may include RAG context -- used for logging only).
        confidence_threshold: Min confidence for neural model to be trusted.
        raw_text: The original user input WITHOUT RAG context.
                  ALL classification layers run on raw_text to avoid
                  false triggers from keywords in historical CRM data.
                  If None, uses `text` (backward compatible).
    """
    classify_text = (raw_text.strip() if raw_text else text).strip()
    
    # -- 1. Rule-based Layer (deterministic overrides) --
    rule_label = rule_based_classify(classify_text)
    if rule_label:
        return {
            "label": rule_label,
            "confidence": 0.99,
            "probabilities": {
                "Hot": 0.99 if rule_label == "Hot" else 0.005, 
                "Warm": 0.99 if rule_label == "Warm" else 0.005, 
                "Cold": 0.99 if rule_label == "Cold" else 0.005
            },
            "method": "rule_based"
        }
    
    # -- 2. Neural model (runs on CLEAN user input) --
    try:
        model, tokenizer, device = _load_model()
        inputs  = tokenizer(classify_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs  = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs_raw  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        top_id = int(np.argmax(probs_raw))
        
        label = ID2LABEL[top_id]
        conf = float(probs_raw[top_id])
        probs_dict = {ID2LABEL[i]: float(probs_raw[i]) for i in range(3)}

        if conf >= confidence_threshold:
            return {
                "label": label,
                "confidence": conf,
                "probabilities": probs_dict,
                "method": "neural"
            }
        
        neural_res = {"label": label, "confidence": conf, "probabilities": probs_dict}
    except Exception as e:
        print(f"[WARN] Neural model failed: {e}")
        neural_res = None

    # -- 3. Baseline fallback ------------------
    baseline = _load_baseline()
    if baseline is not None:
        bl_id    = int(baseline.predict([classify_text])[0])
        bl_label = ID2LABEL[bl_id]
        bl_proba = baseline.predict_proba([classify_text])[0]
        bl_conf  = float(np.max(bl_proba))
        bl_probs_dict = {ID2LABEL[i]: float(bl_proba[i]) for i in range(3)}
        
        if bl_conf >= 0.90:
            return {
                "label": bl_label,
                "confidence": bl_conf,
                "probabilities": bl_probs_dict,
                "method": "fallback"
            }

    # -- 4. Last resort --
    if neural_res:
        return {
            **neural_res,
            "method": "neural_low_conf"
        }

    return {
        "label": "Warm",
        "confidence": 0.0,
        "probabilities": {"Hot": 0.33, "Warm": 0.34, "Cold": 0.33},
        "method": "default"
    }


# ---------------------------------------------
if __name__ == "__main__":
    test_cases = [
        "The client is not interested and asked to be removed from the list.",
        "Budget approved, ready to sign the contract by end of week.",
        "They attended our webinar and are exploring options for next quarter.",
        "No response for 90 days, trial never activated.",
        "The CFO greenlit the investment, send the PO to procurement.",
        "Interested but needs internal alignment before committing.",
        "Explicitly said not the right time and asked us not to reach out again.",
        "Ready to finalize implementation, legal team is reviewing the MSA.",
        "Downloaded our whitepaper, replied positively to the outreach email.",
        "Company is undergoing restructuring, no budget allocated.",
    ]

    print("\n" + "=" * 70)
    print("  LEADSENSE AI -- HYBRID PREDICTION TEST")
    print("=" * 70)
    for i, text in enumerate(test_cases, 1):
        result = predict(text)
        print(f"\nCase {i:02d}: {text[:65]}?")
        print(f"  -> Label: {result['label']:5s} | Conf: {result['confidence']:.3f} | Via: {result['method']}")
