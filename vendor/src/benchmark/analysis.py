"""
Robust Statistical Analysis Engine for EquiCEval Benchmark Evaluation.
Computes:
- False Positive Rate (FPR) on Equivalent Suite (RQ1)
- Mutation Detection Recall & AUROC/AUPRC on Mutant Suite (RQ2)
- Spearman rank correlation rho, Kendall tau, and Clustered Bootstrap CIs (RQ3)
- Model/Prompt Ranking Stability metrics (RQ4)
- Solver Call Complexity & Detection-Cost Trade-offs (RQ5)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats


@dataclass
class EvaluationSummaryStats:
    total_records: int
    equivalent_fpr_baseline: float
    equivalent_fpr_equiceval: float
    mutant_recall_baseline: float
    mutant_recall_equiceval: float
    recall_by_error_family: Dict[str, Tuple[float, float]] # family -> (baseline_recall, equiceval_recall)
    spearman_rho_cons_rmse_vs_gap: float
    spearman_rho_delta_right_vs_gap: float
    kendall_tau_baseline: float
    kendall_tau_equiceval: float
    bootstrap_ci_fpr_equiceval: Tuple[float, float]
    bootstrap_ci_recall_equiceval: Tuple[float, float]
    mean_latency_baseline: float
    mean_latency_equiceval: float


class RobustStatsCalculator:
    """
    Computes rigorous statistical metrics and confidence intervals.
    """

    @staticmethod
    def compute_fpr(predictions_is_fault: List[bool], ground_truth_is_equivalent: List[bool]) -> float:
        """
        False Positive Rate = FP / (FP + TN) on Equivalent Set.
        FP: Equivalent formulation predicted as faulty.
        """
        eq_indices = [i for i, is_eq in enumerate(ground_truth_is_equivalent) if is_eq]
        if not eq_indices:
            return 0.0

        fps = sum(1 for i in eq_indices if predictions_is_fault[i])
        return float(fps / len(eq_indices))

    @staticmethod
    def compute_recall(predictions_is_fault: List[bool], ground_truth_is_mutant: List[bool]) -> float:
        """
        Mutation Detection Recall = TP / (TP + FN) on Mutant Set.
        TP: Mutant formulation correctly predicted as faulty.
        """
        mut_indices = [i for i, is_mut in enumerate(ground_truth_is_mutant) if is_mut]
        if not mut_indices:
            return 0.0

        tps = sum(1 for i in mut_indices if predictions_is_fault[i])
        return float(tps / len(mut_indices))

    @staticmethod
    def clustered_bootstrap_ci(
        data_values: List[float],
        cluster_labels: List[str],
        num_bootstraps: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42
    ) -> Tuple[float, float]:
        """
        Performs Clustered Bootstrap by resampling problem families to compute robust CIs.
        """
        rng = np.random.RandomState(seed)
        unique_clusters = list(set(cluster_labels))
        if not unique_clusters:
            return (0.0, 0.0)

        cluster_map = {c: [data_values[i] for i, label in enumerate(cluster_labels) if label == c] for c in unique_clusters}
        bootstrap_means = []

        for _ in range(num_bootstraps):
            sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            sampled_vals = []
            for c in sampled_clusters:
                sampled_vals.extend(cluster_map[c])
            if sampled_vals:
                bootstrap_means.append(np.mean(sampled_vals))

        if not bootstrap_means:
            return (0.0, 0.0)

        lower_p = (1.0 - ci_level) / 2.0 * 100.0
        upper_p = (1.0 + ci_level) / 2.0 * 100.0
        ci_lower = float(np.percentile(bootstrap_means, lower_p))
        ci_upper = float(np.percentile(bootstrap_means, upper_p))
        return (ci_lower, ci_upper)

    @staticmethod
    def compute_rank_correlations(x: List[float], y: List[float]) -> Tuple[float, float]:
        """
        Computes Spearman rho and Kendall tau rank correlations.
        """
        if len(x) < 3 or len(y) < 3:
            return (0.0, 0.0)

        arr_x = np.array(x, dtype=np.float64)
        arr_y = np.array(y, dtype=np.float64)

        rho, _ = stats.spearmanr(arr_x, arr_y)
        tau, _ = stats.kendalltau(arr_x, arr_y)

        return (float(rho) if not np.isnan(rho) else 0.0,
                float(tau) if not np.isnan(tau) else 0.0)


def leave_one_family_out(corr_fn, rows, family_key: str = "family") -> Dict[str, float]:
    """
    Leave-one-family-out stability: correlation computed on rows without each family.
    """
    families = sorted({r[family_key] for r in rows})
    results: Dict[str, float] = {}
    for fam in families:
        subset = [r for r in rows if r[family_key] != fam]
        results[fam] = float(corr_fn(subset)) if subset else float("nan")
    return results


def leave_one_outlier_out(corr_fn, rows) -> Dict[int, float]:
    """
    Leave-one-outlier-out stability: correlation computed on rows without each data point.
    """
    results: Dict[int, float] = {}
    for i in range(len(rows)):
        subset = rows[:i] + rows[i + 1:]
        results[i] = float(corr_fn(subset)) if subset else float("nan")
    return results


def mixed_effects_model(df, metric_col: str, family_col: str) -> Dict[str, Any]:
    """
    Fit a linear mixed-effects model with a random intercept per family via statsmodels.
    Returns intercept, family variance, number of groups, and number of observations.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("mixed_effects_model requires statsmodels; install it with `pip install statsmodels`")

    formula = f"{metric_col} ~ 1"
    fit = smf.mixedlm(formula, df, groups=df[family_col]).fit()
    family_var = float(np.asarray(fit.cov_re).ravel()[0])
    return {
        "intercept": float(fit.fe_params.iloc[0]),
        "family_var": family_var,
        "n_groups": int(df[family_col].nunique()),
        "n_obs": int(len(df)),
    }


def multiple_comparison_correction(pvalues, method: str = "bonferroni") -> List[float]:
    """
    Adjust p-values for multiple comparisons.
    Supported methods: 'bonferroni', 'holm', 'bh' (Benjamini-Hochberg).
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = len(p)
    if n == 0:
        return []

    if method == "bonferroni":
        corrected = np.minimum(p * n, 1.0)
    elif method == "holm":
        order = np.argsort(p)
        ranked = p[order] * (n - np.arange(n))
        ranked = np.minimum(np.maximum.accumulate(ranked), 1.0)
        corrected = np.empty_like(ranked)
        corrected[order] = ranked
    elif method == "bh":
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1.0)
        ranked = np.minimum.accumulate(np.minimum(ranked, 1.0)[::-1])[::-1]
        corrected = np.empty_like(ranked)
        corrected[order] = ranked
    else:
        raise ValueError(f"Unknown multiple-comparison method: {method!r}")

    return corrected.tolist()


def compute_auroc_auprc(y_true, y_score) -> Tuple[float, float]:
    """
    Compute AUROC and AUPRC for mutation detection. Returns (nan, nan) when the
    true labels contain a single class (metric undefined).
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        raise ImportError("compute_auroc_auprc requires scikit-learn; install it with `pip install scikit-learn`")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2 or np.sum(y_true) == 0:
        return (float("nan"), float("nan"))

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    return (auroc, auprc)


def compute_localization_accuracy(true_list, predicted_list) -> float:
    """
    Top-1 component localization accuracy (RQ2): fraction of records where the top
    predicted component equals the true component. Supports per-record top-1 lists.
    """
    if len(true_list) == 0:
        return 0.0

    def _top(record):
        if isinstance(record, (list, tuple)):
            return record[0] if record else None
        return record

    def _contains(record, comp):
        if isinstance(record, (list, tuple)):
            return comp in record
        return record == comp

    correct = 0
    for true, pred in zip(true_list, predicted_list):
        top = _top(pred)
        if top is not None and _contains(true, top):
            correct += 1
    return correct / len(true_list)


def ranking_stability_ci(scores_by_model, n_boot: int = 100, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap model ranking stability. Returns per-model mean rank and a 95% CI
    (2.5th / 97.5th percentiles) over bootstrap resamples of the score items.
    """
    rng = np.random.RandomState(seed)
    models = list(scores_by_model.keys())
    scores = {m: np.asarray(scores_by_model[m], dtype=np.float64) for m in models}
    n_items = len(scores[models[0]]) if models else 0

    boot_ranks = {m: [] for m in models}
    for _ in range(max(n_boot, 1)):
        if n_items == 0:
            break
        idx = rng.randint(0, n_items, size=n_items)
        means = {m: float(np.mean(scores[m][idx])) for m in models}
        order = sorted(models, key=lambda m: means[m], reverse=True)
        rank_map = {m: i + 1 for i, m in enumerate(order)}
        for m in models:
            boot_ranks[m].append(rank_map[m])

    result: Dict[str, Dict[str, float]] = {}
    for m in models:
        ranks = np.asarray(boot_ranks[m], dtype=np.float64)
        result[m] = {
            "mean_rank": float(np.mean(ranks)),
            "ci_low": float(np.percentile(ranks, 2.5)),
            "ci_high": float(np.percentile(ranks, 97.5)),
        }
    return result


_COST_BINS = ((10, "10"), (50, "50"), (100, "100"), (float("inf"), "500"))


def cost_profiling_by_size(entries) -> Dict[str, Dict[str, float]]:
    """
    Profile solver-call costs by constraint-count bins (10/50/100/500).
    Accepts list of (size, solver_calls) tuples or dicts with 'size' and
    'solver_calls'/'calls' keys.
    """
    buckets = {label: [] for _, label in _COST_BINS}
    for entry in entries:
        if isinstance(entry, dict):
            size = entry["size"]
            calls = entry.get("solver_calls", entry.get("calls"))
        else:
            size, calls = entry[0], entry[1]
        for limit, label in _COST_BINS:
            if size <= limit:
                buckets[label].append(float(calls))
                break

    result: Dict[str, Dict[str, float]] = {}
    for label, values in buckets.items():
        result[label] = {
            "mean_calls": float(np.mean(values)) if values else None,
            "median_calls": float(np.median(values)) if values else None,
            "n": len(values),
        }
    return result


def paired_difference_ci(
    values_a: List[float],
    values_b: List[float],
    cluster_labels: List[str],
    num_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Clustered-bootstrap CI for ``mean(A) - mean(B)`` on paired observations.

    Paired comparison (same pairs, two tools) is required by paper §8.4;
    resampling clusters keeps related variants together. With fewer than two
    clusters the interval is not defined and ``ci_lower``/``ci_upper`` are None,
    while the observed difference is still reported.
    """
    if not (len(values_a) == len(values_b) == len(cluster_labels)):
        raise ValueError("paired inputs must have equal length")
    values_a = [float(v) for v in values_a]
    values_b = [float(v) for v in values_b]
    diffs = [a - b for a, b in zip(values_a, values_b)]
    observed = float(np.mean(diffs)) if diffs else 0.0

    unique = list(dict.fromkeys(cluster_labels))
    if len(unique) < 2:
        return {"observed": observed, "ci_lower": None, "ci_upper": None, "n": len(diffs)}

    cluster_map = {c: [diffs[i] for i, lab in enumerate(cluster_labels) if lab == c]
                   for c in unique}
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(num_bootstraps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        vals = [v for c in sampled for v in cluster_map[c]]
        if vals:
            means.append(np.mean(vals))
    if not means:
        return {"observed": observed, "ci_lower": None, "ci_upper": None, "n": len(diffs)}

    lower_p = (1.0 - ci_level) / 2.0 * 100.0
    upper_p = (1.0 + ci_level) / 2.0 * 100.0
    return {
        "observed": observed,
        "ci_lower": float(np.percentile(means, lower_p)),
        "ci_upper": float(np.percentile(means, upper_p)),
        "n": len(diffs),
    }


def bonferroni_level(num_tests: int, alpha: float = 0.05) -> float:
    """CI level adjusted for multiple comparisons (Bonferroni, paper §8.4)."""
    if num_tests <= 1:
        return 1.0 - alpha
    return 1.0 - alpha / num_tests


def leave_one_family_out_metrics(labels: List, verdicts: List[str], families: List[str]) -> Dict[str, Dict[str, Any]]:
    """Recompute FPR/recall with each problem family removed in turn (§8.4)."""
    from src.equiceval.metrics import verification_metrics

    out: Dict[str, Dict[str, Any]] = {}
    for family in sorted(set(families)):
        keep = [i for i, f in enumerate(families) if f != family]
        metrics = verification_metrics([labels[i] for i in keep], [verdicts[i] for i in keep])
        out[family] = {
            "n": len(keep),
            "false_alarm_rate": metrics["false_alarm_rate"]["value"],
            "error_recall": metrics["error_recall"]["value"],
        }
    return out
