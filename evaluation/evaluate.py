"""
Evaluation Script -- LeadSense AI v2.

Modes:
  hybrid     -- Hybrid predictor on validation set (standard metrics)
  ood        -- 30 real-world OOD examples (messy, indirect, non-template)
  adversarial-- 30 contradictory/tricky inputs designed to break keyword models
  ablation   -- Model-only vs Rule+Model comparison on same test set
  compare    -- Side-by-side model comparison table
  all        -- Run all modes

Usage:
  python evaluation/evaluate.py --mode all
  python evaluation/evaluate.py --mode ood
  python evaluation/evaluate.py --mode adversarial
  python evaluation/evaluate.py --mode ablation
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "final_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"

ID2LABEL = {0: "Hot", 1: "Warm", 2: "Cold"}


# ======================================================================
# OOD TEST SET -- 30 Real-World Style Examples
# 10 per class: messy, indirect, non-template, realistic
# ======================================================================
OOD_TEST_CASES = [
    # -- OOD HOT (10) ---------------------------------------------------
    ("pls send quote asap, mgmt pushing hard, need docs by fri latest",                "Hot",  "Informal urgent CRM note"),
    ("3 calls this wk from their ops lead. wants impl plan. think we won this",        "Hot",  "Abbreviated sales note"),
    ("cfo greenlit it in the meeting i was told. send the damn contract pls",           "Hot",  "Casual chain message with profanity"),
    ("They're not walking away from this -- their CEO literally told our AE 'get it done'", "Hot", "Indirect strong buy signal"),
    ("sign-off came through!! kickoff in 2wks. loop in impl team now",                 "Hot",  "Excited abbreviated note"),
    ("we burning cash on the manual process, ur solution is exactly what we need rn",  "Hot",  "Metaphorical urgency, casual tone"),
    ("Legal reviewed. minor clauses. can close weds if u agree to 99.9 uptime SLA",   "Hot",  "Messy email fragment"),
    ("their procurement dude called ME. thats a first. he wants po details today",     "Hot",  "Reversed outreach = strong signal"),
    ("This is the one. pilot results blew them away. they want full rollout next mo",  "Hot",  "Sales comment, informal"),
    ("CEO overrode the slow procurement. said start now sort paperwork later",          "Hot",  "Impatient executive signal"),

    # -- OOD WARM (10) --------------------------------------------------
    ("they kinda interested but no decision yet. good vibes tho",                      "Warm", "Informal vague interest"),
    ("opened every email we sent in 3 months but never replied. classic lurker",       "Warm", "Behavioral signal, indirect"),
    ("spoke 2 their guy, he likes it but gotta convince the boss first",               "Warm", "Abbreviated sales note"),
    ("attended demo, asked good questions, comparing w/ 2 others. no timeline",        "Warm", "Abbreviated, genuine evaluator"),
    ("just browsing options for now but ur product came highly recmd",                 "Warm", "Casual early-stage browsing"),
    ("waiting for internal approval before we can even start a trial",                 "Warm", "Blocked but interested"),
    ("good prospect but slow movers. check in every 3wks, dont push too hard",         "Warm", "Messy nurture note"),
    ("dm'd us on twitter asking about enterprise pricing. too early to call hot",       "Warm", "Social media early signal"),
    ("new contact replied to nurture seq after 6 weeks silence. keep them warm",       "Warm", "Re-engaged warm lead"),
    ("she said 'on my list for q3' which means interested but not now",               "Warm", "Indirect timeline signal"),

    # -- OOD COLD (10) --------------------------------------------------
    ("nope not interested rn. stop emailing pls",                                      "Cold", "Blunt rejection, informal"),
    ("ghosted us 5 months. linkedin msg: 'went w/ someone else'. rip this one",        "Cold", "Indirect rejection via social"),
    ("wrong fit tbh, they need smth we dont do. close this out",                       "Cold", "Sales rep closing bad fit"),
    ("company restructuring. nobody left who cared abt this project. close",           "Cold", "Org change killed deal"),
    ("liked the demo but their board axed the whole initiative. dead end",             "Cold", "Board killed project -- adversarial-ish"),
    ("budget approved for diff project entirely. not us. rip",                         "Cold", "Budget exists but not for us"),
    ("3rd follow up, bounced. linkedin: no profile. company site: down. ghost",        "Cold", "Contact disappeared"),
    ("they said maybe q4 for 4 quarters straight now. this is going nowhere",          "Cold", "Chronic delay = Cold"),
    ("unsubscribed AND marked as spam. dont think we're coming back from that",        "Cold", "Strong disengage"),
    ("great pilot feedback, terrible outcome: they decided to build in-house",         "Cold", "Adversarial: positive feedback -> Cold"),
]


# ======================================================================
# ADVERSARIAL TEST SET -- 30 Contradictory/Tricky Inputs
# Designed to break keyword-reliant models
# ======================================================================
ADVERSARIAL_TEST_CASES = [
    # -- ADVERSARIAL HOT (10) -- look Cold/Warm but are Hot --------------
    ("budget approved but not for our product -- the CFO actually overrode that and approved ours instead", "Hot", "Double twist: budget initially cold, then approved"),
    ("Initially said no. After the pilot they completely changed their mind and want to sign by Friday.", "Hot", "Negation reversal"),
    ("no budget approved yet but legal told us to assume it's happening and prepare the implementation plan", "Hot", "Implied approval without formal confirmation"),
    ("They went quiet for 6 weeks then called today to say 'let's finalize.' Moving to close.", "Hot", "Re-engagement after silence"),
    ("not in a rush -- said that -- but also sent us a calendar invite titled 'Contract Review.' Go figure.", "Hot", "Contradiction: stated no urgency, actions say Hot"),
    ("their champion left the org but the new CTO took over and is even more committed to signing", "Hot", "Personnel change but deal survives"),
    ("said they were going with a competitor. called back 48hrs later to say the competitor failed and they want us.", "Hot", "Competitor failure = rescue deal"),
    ("they told our AE 'we can't make decisions until Q4' -- that was last quarter, Q4 is now, and they called.", "Hot", "Time context shift"),
    ("very difficult internal process -- procurement slow as molasses -- but all stakeholders have now approved.", "Hot", "Process pain ? Cold intent"),
    ("said 'not now' eight months ago. 'now' is now. they want implementation to start in two weeks.", "Hot", "Time-delayed Yes"),

    # -- ADVERSARIAL WARM (10) -- look Hot/Cold but are Warm --------------
    ("budget approved but it's for a different initiative. still interested in us for next cycle.", "Warm", "Budget approved but not for us"),
    ("not ready to finalize yet but this is their #1 priority for H2.", "Warm", "Negation + high priority = still Warm"),
    ("no budget yet but very interested -- described it as 'when not if'", "Warm", "No budget + genuine intent"),
    ("urgently requested a demo -- but this is how they start every vendor evaluation, always.", "Warm", "Urgent demo ? Hot if it's just their process"),
    ("said 'ready to move forward' -- meaning they're ready to enter formal eval, not to buy.", "Warm", "Ambiguous 'ready' language"),
    ("asked for pricing -- for a board presentation to justify the budget request next quarter.", "Warm", "Pricing request ? buying intent"),
    ("sent three emails this week asking questions -- very engaged but explicitly said 12 months minimum before deciding.", "Warm", "High engagement + long horizon"),
    ("their CTO loves it. but the CTO isn't the decision maker and the committee is 6 months from voting.", "Warm", "Executive enthusiasm ? deal readiness"),
    ("trial activated, heavy usage -- but they're also trialing 3 competitors and won't decide until all pilots complete.", "Warm", "High engagement, competitive eval"),
    ("they signed the NDA and sent their technical requirements -- still just early-stage qualification.", "Warm", "Document exchange ? buying signal"),

    # -- ADVERSARIAL COLD (10) -- look Hot/Warm but are Cold --------------
    ("budget approved but not proceeding -- the initiative was cancelled despite having the funds.", "Cold", "Classic adversarial: budget ? purchase"),
    ("loved the demo but said it's too expensive and they won't reconsider the price point.", "Cold", "Positive feedback -> price objection -> Cold"),
    ("signed the NDA, completed 30-day trial, gave great feedback, then went silent for 4 months.", "Cold", "Engagement trail goes cold"),
    ("said we're their top choice but they've been saying that for 8 months with no movement.", "Cold", "Stale verbal commitment"),
    ("executive sponsor greenlit it -- but that executive left and no one else is carrying the initiative.", "Cold", "Champion departed = Cold"),
    ("strong interest 6 months ago. since then: 2 mergers, 3 contact changes, 0 responses.", "Cold", "Org chaos = deal death"),
    ("won't proceed even after the pilot results exceeded their own success criteria.", "Cold", "Negation + good results = mysterious Cold"),
    ("their procurement team is ready to issue the PO -- but the business team has deprioritized the initiative.", "Cold", "Process ready, business cold"),
    ("enthusiastic on every call for a year. no decision. their champion admitted 'it'll never happen here.'", "Cold", "Friendly but no path to purchase"),
    ("requested a proposal. reviewed it. asked for revisions. reviewed again. then said 'let's revisit in 2 years.'", "Cold", "Exhausting engagement -> definitive Cold"),
]


# ======================================================================
# Helpers
# ======================================================================
def _save_confusion_matrix(labels: list, preds: list, title: str, save_path: Path):
    cm = confusion_matrix(labels, preds, labels=["Hot", "Warm", "Cold"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Hot", "Warm", "Cold"],
        yticklabels=["Hot", "Warm", "Cold"],
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix -> {save_path}")


def _print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _run_test_suite(
    cases: list[tuple],
    suite_name: str,
    use_rules: bool = True,
) -> dict:
    """Generic test runner for any list of (text, label, rationale) tuples."""
    from inference.hybrid_predictor import predict, neural_predict, rule_based_classify
    from preprocessing.preprocess import LABEL_MAP

    labels_true, labels_pred, methods = [], [], []
    failures = []

    for text, expected, rationale in cases:
        if use_rules:
            result = predict(text)
            predicted = result["label"]
            method = result["method"]
        else:
            # Model-only: skip rule layer
            rule = rule_based_classify(text)
            try:
                predicted, conf = neural_predict(text)
                method = "neural_only"
            except RuntimeError:
                predicted = "Warm"
                method = "default"

        labels_true.append(expected)
        labels_pred.append(predicted)
        methods.append(method)

        if predicted != expected:
            failures.append({
                "text": text[:80],
                "expected": expected,
                "predicted": predicted,
                "rationale": rationale,
            })

    acc = accuracy_score(labels_true, labels_pred)
    f1  = f1_score(labels_true, labels_pred, average="macro", zero_division=0)

    print(f"\n  Results ({suite_name}):")
    print(f"    Accuracy : {acc:.4f}  ({int(acc*len(cases))}/{len(cases)})")
    print(f"    Macro F1 : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(labels_true, labels_pred, target_names=["Hot", "Warm", "Cold"], zero_division=0))

    if failures:
        print(f"  [FAIL] Failures ({len(failures)}):")
        for fail in failures:
            print(f"    [{fail['expected']}->{fail['predicted']}] {fail['text']}")
            print(f"      Rationale: {fail['rationale']}")

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "n": len(cases),
        "n_correct": int(acc * len(cases)),
        "failures": failures,
        "method_counts": dict(Counter(methods)),
    }


# ======================================================================
# Evaluation Modes
# ======================================================================
def evaluate_hybrid(val_csv: Path | None = None):
    """Standard hybrid predictor evaluation on the validation set."""
    from inference.hybrid_predictor import predict
    from preprocessing.preprocess import LABEL_MAP

    if val_csv is None:
        val_csv = DATA_DIR / "final_dataset" / "final_dataset.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"{val_csv} not found. Run preprocessing first.")

    df = pd.read_csv(val_csv)
    texts  = df["text"].tolist()
    labels_true = [ID2LABEL[i] for i in df["label_id"].tolist()]

    _print_banner("STANDARD EVALUATION -- Hybrid Predictor on Validation Set")
    print(f"  Samples: {len(df)}")

    preds, methods = [], []
    for text in texts:
        result = predict(text)
        preds.append(result["label"])
        methods.append(result["method"])

    acc      = accuracy_score(labels_true, preds)
    macro_f1 = f1_score(labels_true, preds, average="macro", zero_division=0)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(labels_true, preds, target_names=["Hot", "Warm", "Cold"], zero_division=0))
    print(f"  Method breakdown: {dict(Counter(methods))}")

    _save_confusion_matrix(
        labels_true, preds,
        "Hybrid Predictor -- Validation Set",
        RESULTS_DIR / "hybrid_confusion_matrix.png",
    )

    results = {"model": "Hybrid", "accuracy": acc, "macro_f1": macro_f1, "method_counts": dict(Counter(methods))}
    with open(RESULTS_DIR / "hybrid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def evaluate_ood():
    """Run OOD test on 30 real-world messy examples."""
    _print_banner("OOD TEST -- 30 Real-World Messy Examples")
    results = _run_test_suite(OOD_TEST_CASES, "OOD", use_rules=True)

    labels_true = [c[1] for c in OOD_TEST_CASES]
    labels_pred_list = [
        r["predicted"] for r in results["failures"]
    ]
    # Re-run to get all predictions for confusion matrix
    from inference.hybrid_predictor import predict
    all_preds = [predict(c[0])["label"] for c in OOD_TEST_CASES]
    _save_confusion_matrix(
        labels_true, all_preds,
        "OOD Test -- Confusion Matrix",
        RESULTS_DIR / "ood_confusion_matrix.png",
    )

    with open(RESULTS_DIR / "ood_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  OOD results saved -> {RESULTS_DIR / 'ood_results.json'}")
    return results


def evaluate_adversarial():
    """Run adversarial test on 30 contradictory inputs."""
    _print_banner("ADVERSARIAL TEST -- 30 Contradictory/Tricky Inputs")
    print("  These cases are designed to BREAK keyword-reliant models.\n")
    results = _run_test_suite(ADVERSARIAL_TEST_CASES, "Adversarial", use_rules=True)

    from inference.hybrid_predictor import predict
    labels_true = [c[1] for c in ADVERSARIAL_TEST_CASES]
    all_preds = [predict(c[0])["label"] for c in ADVERSARIAL_TEST_CASES]
    _save_confusion_matrix(
        labels_true, all_preds,
        "Adversarial Test -- Confusion Matrix",
        RESULTS_DIR / "adversarial_confusion_matrix.png",
    )

    with open(RESULTS_DIR / "adversarial_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Adversarial results saved -> {RESULTS_DIR / 'adversarial_results.json'}")
    return results


def evaluate_ablation():
    """Compare Model-Only vs Rule+Model on OOD + Adversarial sets."""
    _print_banner("RULE ABLATION -- Model-Only vs Rule+Model")

    all_cases = OOD_TEST_CASES + ADVERSARIAL_TEST_CASES

    print("\n  -- [A] MODEL ONLY (neural, no rule override) --")
    results_model = _run_test_suite(all_cases, "Model-Only", use_rules=False)

    print("\n  -- [B] RULE + MODEL (hybrid) --")
    results_hybrid = _run_test_suite(all_cases, "Rule+Model", use_rules=True)

    delta_acc = results_hybrid["accuracy"] - results_model["accuracy"]
    delta_f1  = results_hybrid["macro_f1"]  - results_model["macro_f1"]

    print("\n  -- ABLATION SUMMARY --")
    print(f"  {'':25s}  {'Accuracy':>10}  {'Macro F1':>10}")
    print(f"  {'Model Only':25s}  {results_model['accuracy']:>10.4f}  {results_model['macro_f1']:>10.4f}")
    print(f"  {'Rule + Model':25s}  {results_hybrid['accuracy']:>10.4f}  {results_hybrid['macro_f1']:>10.4f}")
    print(f"  {'Delta (Rules add)':25s}  {delta_acc:>+10.4f}  {delta_f1:>+10.4f}")

    if delta_acc < 0:
        print("\n  [!]  Rules HURT performance on OOD/Adversarial examples.")
        print("     Consider softening or removing keyword overrides.")
    elif delta_acc > 0.02:
        print("\n  [OK] Rules meaningfully HELP performance.")
    else:
        print("\n  [i]  Rules have negligible effect (<2% delta).")

    ablation = {
        "model_only": {"accuracy": results_model["accuracy"], "macro_f1": results_model["macro_f1"]},
        "rule_plus_model": {"accuracy": results_hybrid["accuracy"], "macro_f1": results_hybrid["macro_f1"]},
        "delta_accuracy": delta_acc,
        "delta_macro_f1": delta_f1,
    }
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"\n  Ablation results saved -> {RESULTS_DIR / 'ablation_results.json'}")
    return ablation


def full_comparison_report():
    """Load all results files and print a unified comparison table."""
    files = {
        "TF-IDF+LR (Baseline)":   RESULTS_DIR / "baseline_results.json",
        "MiniLM Transformer":      RESULTS_DIR / "minilm_results.json",
        "Hybrid (Rule+Neural)":    RESULTS_DIR / "hybrid_results.json",
        "OOD Test":                RESULTS_DIR / "ood_results.json",
        "Adversarial Test":        RESULTS_DIR / "adversarial_results.json",
    }

    _print_banner("LEADSENSE AI -- FULL MODEL COMPARISON REPORT")
    print(f"  {'Model / Test':35s}  {'Accuracy':>10}  {'Macro F1':>10}")
    print("  " + "-" * 60)

    for name, path in files.items():
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            acc = d.get("accuracy") or d.get("eval_accuracy", "N/A")
            f1  = d.get("macro_f1") or d.get("eval_macro_f1", "N/A")
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            f1_str  = f"{f1:.4f}"  if isinstance(f1, float)  else str(f1)
            print(f"  {name:35s}  {acc_str:>10}  {f1_str:>10}")
        else:
            print(f"  {name:35s}  {'N/A':>10}  {'N/A':>10}")

    print("=" * 70)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["hybrid", "ood", "adversarial", "ablation", "compare", "all"],
        default="all",
    )
    args = p.parse_args()

    if args.mode in ("hybrid", "all"):
        evaluate_hybrid()
    if args.mode in ("ood", "all"):
        evaluate_ood()
    if args.mode in ("adversarial", "all"):
        evaluate_adversarial()
    if args.mode in ("ablation", "all"):
        evaluate_ablation()
    if args.mode in ("compare", "all"):
        full_comparison_report()
