"""F1-based metrics, report generation, and result archiving."""

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results.json")
RESULTS_DIR = Path("results")


@dataclass
class SystemMetrics:
    """Aggregate metrics for one system (baseline or react)."""

    mean_f1: float
    mean_precision: float
    mean_recall: float
    exact_match_rate: float
    total: int


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successes.
        total: Total trials.
        z: Z-score for confidence level (1.96 = 95%).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = p + z**2 / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return (center - spread) / denom, (center + spread) / denom


def compute_metrics(results: list[dict], system: str) -> SystemMetrics:
    """Compute aggregate F1 metrics for a system from evaluation results.

    Args:
        results: List of result dicts from eval run.
        system: "baseline" or "react".

    Returns:
        SystemMetrics with mean F1, precision, recall.
    """
    total = len(results)
    scores = [r[system] for r in results]

    mean_f1 = sum(s["f1"] for s in scores) / total if total else 0.0
    mean_p = sum(s["precision"] for s in scores) / total if total else 0.0
    mean_r = sum(s["recall"] for s in scores) / total if total else 0.0
    exact = sum(1 for s in scores if s["f1"] == 1.0)

    return SystemMetrics(
        mean_f1=mean_f1,
        mean_precision=mean_p,
        mean_recall=mean_r,
        exact_match_rate=exact / total if total else 0.0,
        total=total,
    )


def per_category_breakdown(results: list[dict]) -> dict[str, dict]:
    """Compute per-category mean F1 for both systems.

    Args:
        results: List of result dicts.

    Returns:
        Dict mapping category to F1 stats.
    """
    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["gold_label"], []).append(r)

    breakdown = {}
    for category in sorted(by_category):
        items = by_category[category]
        n = len(items)
        b_f1 = sum(r["baseline"]["f1"] for r in items) / n
        r_f1 = sum(r["react"]["f1"] for r in items) / n
        breakdown[category] = {"n": n, "baseline_f1": b_f1, "react_f1": r_f1}
    return breakdown


def print_report(results: list[dict]) -> None:
    """Print formatted evaluation report to stdout.

    Args:
        results: List of result dicts.
    """
    baseline = compute_metrics(results, "baseline")
    react = compute_metrics(results, "react")

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Baseline':>12} {'ReAct':>12}")
    print("-" * 54)
    print(f"{'Mean F1':<30} {baseline.mean_f1:>11.3f} {react.mean_f1:>11.3f}")
    print(f"{'Mean Precision':<30} {baseline.mean_precision:>11.3f} {react.mean_precision:>11.3f}")
    print(f"{'Mean Recall':<30} {baseline.mean_recall:>11.3f} {react.mean_recall:>11.3f}")
    print(f"{'Exact Match Rate':<30} {baseline.exact_match_rate:>11.1%} {react.exact_match_rate:>11.1%}")
    print(f"{'Total Samples':<30} {baseline.total:>12} {react.total:>12}")

    delta = react.mean_f1 - baseline.mean_f1
    print(f"\nReAct F1 improvement: {delta:+.3f} ({'PASS' if delta >= 0.10 else 'FAIL'}: target >= +0.10)")

    breakdown = per_category_breakdown(results)
    print(f"\n{'Category':<25} {'N':>4} {'Baseline F1':>12} {'ReAct F1':>10}")
    print("-" * 51)
    for category, stats in breakdown.items():
        print(f"{category:<25} {stats['n']:>4} {stats['baseline_f1']:>11.3f} {stats['react_f1']:>9.3f}")

    print("=" * 60)


def save_results(results: list[dict], path: Path = RESULTS_PATH) -> None:
    """Save results list as JSON (incremental checkpoint)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", path)


def _next_version() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted(RESULTS_DIR.glob("v*.json"))
    if not existing:
        return 1
    return int(existing[-1].stem.lstrip("v")) + 1


def archive_results(results: list[dict]) -> Path:
    """Archive final results to results/vNNN.json with embedded metrics."""
    version = _next_version()
    baseline = compute_metrics(results, "baseline")
    react = compute_metrics(results, "react")
    breakdown = per_category_breakdown(results)

    archive = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": len(results),
        "summary": {
            "baseline_f1": round(baseline.mean_f1, 4),
            "react_f1": round(react.mean_f1, 4),
            "delta_f1": round(react.mean_f1 - baseline.mean_f1, 4),
            "baseline_exact_match": round(baseline.exact_match_rate, 4),
            "react_exact_match": round(react.exact_match_rate, 4),
        },
        "per_category": breakdown,
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"v{version:03d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)

    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    logger.info("Archived results to %s (v%d)", path, version)
    return path
