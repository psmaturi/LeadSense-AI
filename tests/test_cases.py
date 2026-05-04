"""
Test Suite -- LeadSense AI v2.

30 Test Cases across 3 sub-suites:
  A. Standard (10)    -- Clean, well-formed test cases covering all classes
  B. OOD (10)         -- Messy, informal, indirect real-world inputs
  C. Adversarial (10) -- Contradictory cases that break keyword models

Usage:
  python tests/test_cases.py
  python tests/test_cases.py --suite all
  python tests/test_cases.py --suite ood
  python tests/test_cases.py --suite adversarial
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.hybrid_predictor import predict

# ======================================================================
# SUITE A -- Standard Test Cases (10)
# ======================================================================
STANDARD_CASES = [
    (
        "Legal is reviewing the MSA. PO is being raised. They want kickoff before end of quarter.",
        "Hot",
        "Legal + PO + deadline -> Hot",
    ),
    (
        "The prospect explicitly asked to be removed from all our communications.",
        "Cold",
        "Direct opt-out request -> Cold",
    ),
    (
        "They attended our webinar and downloaded the whitepaper. Exploring options, no timeline.",
        "Warm",
        "Interest + exploration, no urgency -> Warm",
    ),
    (
        "No response to 4 follow-ups. Trial account was activated but never used.",
        "Cold",
        "Disengagement signals -> Cold",
    ),
    (
        "VP of Procurement asked our AE to draft the SOW. Implementation must start before Q3.",
        "Hot",
        "SOW request + urgency -> Hot",
    ),
    (
        "Replied positively to our nurture email. Curious about enterprise features but building a business case still.",
        "Warm",
        "Positive re-engagement + internal process -> Warm",
    ),
    (
        "Company is undergoing restructuring. Budget freeze until at least next fiscal year. Do not contact.",
        "Cold",
        "Budget freeze + do-not-contact -> Cold",
    ),
    (
        "Shortlisted to final two vendors. Executive sponsor involved. Requesting references from similar clients.",
        "Hot",
        "Final shortlist + exec involvement -> Hot",
    ),
    (
        "Signed up for free trial last week and clicked the ROI calculator. Still early-stage research.",
        "Warm",
        "Trial + engagement + early-stage -> Warm",
    ),
    (
        "CFO approved the spend. Purchase order is ready. Escalate to senior AE immediately.",
        "Hot",
        "CFO approval + PO ready -> Hot",
    ),
]

# ======================================================================
# SUITE B -- OOD Test Cases (10)
# Messy, informal, non-template real-world inputs
# ======================================================================
OOD_CASES = [
    (
        "pls send quote asap, mgmt pushing hard this wk",
        "Hot",
        "Informal urgent CRM note -- abbreviations + pressure signal",
    ),
    (
        "they kinda interested but no decision yet. good vibes tho",
        "Warm",
        "Casual vague interest -- no urgency, no commitment",
    ),
    (
        "nope not interested rn. stop emailing pls",
        "Cold",
        "Blunt informal rejection",
    ),
    (
        "we burning cash on this manual process, ur solution is exactly what we need rn",
        "Hot",
        "Metaphorical urgency + need signal, casual language",
    ),
    (
        "good prospect but slow movers. check in every 3wks, dont push too hard",
        "Warm",
        "Sales rep nurture note -- interested but slow",
    ),
    (
        "ghosted us 5 months. linkedin msg: went w/ someone else. rip this one",
        "Cold",
        "Indirect rejection via social media, abbreviated",
    ),
    (
        "cfo greenlit it, send the contract pls. they hv 6wk impl window",
        "Hot",
        "Messy CRM note -- CFO approval + implementation timeline",
    ),
    (
        "dm'd us on twitter asking about enterprise pricing. too early to call hot",
        "Warm",
        "Social media early signal -- engagement without commitment",
    ),
    (
        "liked the demo but their board axed the whole initiative. dead end",
        "Cold",
        "Positive signal -> board cancellation -> Cold",
    ),
    (
        "spoke 2 their guy, he likes it but gotta convince the boss first",
        "Warm",
        "Champion interest but no authority -- Warm",
    ),
]

# ======================================================================
# SUITE C -- Adversarial Test Cases (10)
# Contradictory inputs that fool keyword-based models
# ======================================================================
ADVERSARIAL_CASES = [
    (
        "budget approved but not proceeding -- the initiative was cancelled despite having the funds.",
        "Cold",
        "ADVERSARIAL: 'budget approved' -> keyword trap -> actually Cold",
    ),
    (
        "Initially said no interest. After the pilot they completely reversed -- want to sign by Friday.",
        "Hot",
        "ADVERSARIAL: starts with rejection, ends with commitment -> Hot",
    ),
    (
        "no budget yet but very interested -- described it as 'when not if'",
        "Warm",
        "ADVERSARIAL: no budget keyword -> still Warm due to intent",
    ),
    (
        "loved the demo but too expensive and won't reconsider the price point.",
        "Cold",
        "ADVERSARIAL: positive feedback -> price objection -> Cold",
    ),
    (
        "not ready to finalize yet but this is their #1 priority for H2.",
        "Warm",
        "ADVERSARIAL: negation + high priority = Warm, not Cold",
    ),
    (
        "executive sponsor greenlit it -- but that executive left and no one else is carrying the initiative.",
        "Cold",
        "ADVERSARIAL: executive approval -> champion lost -> Cold",
    ),
    (
        "won't proceed even after the pilot results exceeded their own success criteria.",
        "Cold",
        "ADVERSARIAL: good results, explicit won't proceed -> Cold",
    ),
    (
        "urgently requested a demo -- but this is how they start every vendor eval without buying intent.",
        "Warm",
        "ADVERSARIAL: urgency signal -> process habit, not buying signal",
    ),
    (
        "said they were going with a competitor, then called back 48 hours later -- competitor failed, want us.",
        "Hot",
        "ADVERSARIAL: stated competitor choice -> reversed -> Hot",
    ),
    (
        "very enthusiastic on every call for a year. champion admitted 'it'll never happen here.'",
        "Cold",
        "ADVERSARIAL: sustained enthusiasm -> internal political block -> Cold",
    ),
]


# ======================================================================
# Test Runner
# ======================================================================
def run_suite(cases: list, suite_name: str, verbose: bool = True) -> tuple[int, int]:
    passed = failed = 0
    fail_list = []

    print(f"\n  -- Suite: {suite_name} ({len(cases)} cases) --")

    for i, (text, expected, rationale) in enumerate(cases, 1):
        result    = predict(text)
        predicted = result["label"]
        conf      = result["confidence"]
        method    = result["method"]
        ok        = predicted == expected

        if ok:
            passed += 1
            status = "[OK]"
        else:
            failed += 1
            status = "[FAIL]"
            fail_list.append((text, expected, predicted, rationale))

        if verbose:
            short_text = (text[:72] + "?") if len(text) > 75 else text
            print(f"    {status} [{expected}->{predicted}] (conf={conf:.2f}, via={method})")
            print(f"       {short_text}")

    if fail_list and verbose:
        print(f"\n  -- Failures in {suite_name} --")
        for text, exp, pred, rationale in fail_list:
            print(f"    [{exp}->{pred}] {text[:70]}")
            print(f"      ? {rationale}")

    return passed, failed


def run_tests(suite: str = "all", verbose: bool = True) -> dict:
    suites = {
        "standard":   ("A -- Standard",   STANDARD_CASES),
        "ood":        ("B -- OOD",        OOD_CASES),
        "adversarial":("C -- Adversarial", ADVERSARIAL_CASES),
    }

    print("\n" + "=" * 70)
    print("  LEADSENSE AI -- TEST SUITE v2 (30 Cases, 3 Sub-Suites)")
    print("=" * 70)

    total_pass = total_fail = 0
    report = {}

    for key, (name, cases) in suites.items():
        if suite not in ("all", key):
            continue
        p, f = run_suite(cases, name, verbose=verbose)
        total_pass += p
        total_fail += f
        report[key] = {"passed": p, "failed": f, "total": p + f}

    total = total_pass + total_fail
    pct   = 100 * total_pass / total if total else 0

    print("\n" + "=" * 70)
    print(f"  OVERALL: {total_pass}/{total} PASSED  ({pct:.0f}%)")
    for key, r in report.items():
        bar = "[OK]" if r["failed"] == 0 else "[FAIL]"
        print(f"  {bar} {key.capitalize():12s}: {r['passed']}/{r['total']}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["all", "standard", "ood", "adversarial"], default="all")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_tests(suite=args.suite, verbose=not args.quiet)
