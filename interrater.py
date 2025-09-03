#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List

import numpy as np
from sklearn.metrics import cohen_kappa_score

# ----- Ordered categories (low -> high) -----
ORDER = ["problem_answers", "partially", "acceptable", "perfect"]
ALLOWED = set(ORDER)
TO_INT = {lbl: i for i, lbl in enumerate(ORDER)}

# ----- Helpers -----
def _to_int_labels(labels: List[str]) -> np.ndarray:
    unknown = set(labels) - ALLOWED
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}. Expected {ORDER}")
    return np.array([TO_INT[x] for x in labels])

def kappa_unweighted(llm_labels: List[str], human_labels: List[str]) -> float:
    """Standard (unweighted) Cohen's kappa."""
    y1 = _to_int_labels(llm_labels)
    y2 = _to_int_labels(human_labels)
    return cohen_kappa_score(y1, y2)  # weights=None by default

def kappa_quadratic(llm_labels: List[str], human_labels: List[str]) -> float:
    """Quadratic weighted Cohen's kappa for ordinal labels."""
    y1 = _to_int_labels(llm_labels)
    y2 = _to_int_labels(human_labels)
    return cohen_kappa_score(y1, y2, weights="quadratic")

def kappa_bootstrap_ci(
    llm_labels: List[str],
    human_labels: List[str],
    B: int = 5000,
    seed: int = 7
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap 95% CI for κ_QW.
    B = number of resamples.
    """
    rng = np.random.default_rng(seed)
    pairs = np.array(list(zip(llm_labels, human_labels)), dtype=object)
    n = len(pairs)
    if n < 2:
        raise ValueError("Need at least 2 paired labels to compute CI.")
    samples = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)  # sample n pairs with replacement
        a = [pairs[i][0] for i in idx]
        b_ = [pairs[i][1] for i in idx]
        samples[b] = kappa_quadratic(a, b_)
    lb, ub = np.percentile(samples, [2.5, 97.5])
    return float(lb), float(ub)

def load_all_evals(directory: Path, pattern: str = "evaluation_*.json") -> Dict[str, str]:
    """
    Loads evaluation JSON files and returns {question_id: category}.
    Accepts files that contain either a single object or a list of objects.
    """
    evaluations: Dict[str, str] = {}
    for path in sorted(directory.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        print(path)
        items = data if isinstance(data, list) else [data]
        for item in items:
            qid = str(item["question_id"])
            cat = item["judgement"]["category"]
            if cat not in ALLOWED:
                raise ValueError(f"Unknown category '{cat}' in {path} (qid={qid}); expected {ORDER}")
            evaluations[qid] = cat
    return evaluations

# ----- Main comparison -----
def compare_directories(
    llm_dir: str,
    human_dir: str,
    llm_pattern: str = "evaluation_gpt5-mini*.json",  # adjust to your file naming
    human_pattern: str = "evaluation_*.json",
    adopt_threshold: float = 0.70,
    adopt_ci_lower: float = 0.60,
    B: int = 5000,
    seed: int = 7,
):
    llm_dir_p = Path(llm_dir)
    human_dir_p = Path(human_dir)

    llm_evals = load_all_evals(llm_dir_p, llm_pattern)
    human_evals = load_all_evals(human_dir_p, human_pattern)

    common_ids = sorted(set(llm_evals) & set(human_evals))
    if not common_ids:
        raise ValueError("No overlapping question_id between LLM and human evaluations.")

    llm_labels   = [llm_evals[qid] for qid in common_ids]
    human_labels = [human_evals[qid] for qid in common_ids]

    # κ (unweighted) and κ_QW (quadratic) + CI for κ_QW
    kappa_unw = kappa_unweighted(llm_labels, human_labels)
    kappa_qw  = kappa_quadratic(llm_labels, human_labels)
    ci_low, ci_high = kappa_bootstrap_ci(llm_labels, human_labels, B=B, seed=seed)

    # Decision rule uses κ_QW and its lower CI bound
    decision = "ADOPT" if (kappa_qw >= adopt_threshold and ci_low >= adopt_ci_lower) else "HUMAN-IN-LOOP"

    # Build stats
    stats = {
        "n_common": len(common_ids),
        "llm_counts": dict(Counter(llm_labels)),
        "human_counts": dict(Counter(human_labels)),
        "kappa_unweighted": kappa_unw,
        "kappa_qw": kappa_qw,
        "ci95_qw": (ci_low, ci_high),
        "decision": decision,
        "order": ORDER,
        "thresholds": {"adopt_threshold": adopt_threshold, "adopt_ci_lower": adopt_ci_lower},
    }

    # Console summary
    print(f"Compared {stats['n_common']} items")
    print(f"Unweighted κ       = {kappa_unw:.3f}")
    print(f"Quadratic weighted κ = {kappa_qw:.3f}  CI95 = ({ci_low:.3f}, {ci_high:.3f})  -> {decision}")
    print("LLM distribution :", stats["llm_counts"])
    print("Human distribution:", stats["human_counts"])
    return stats

# ----- Example run -----
if __name__ == "__main__":
    stats = compare_directories(
        llm_dir="evaluations/50-gpt5-mini",
        human_dir="evaluations/human-gpt5-mini",
        llm_pattern="evaluation_gpt5-mini*.json",
        human_pattern="gpt5-mini*.json",
        adopt_threshold=0.70,
        adopt_ci_lower=0.60,
        B=5000,
        seed=7,
    )
