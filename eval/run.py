"""Evaluation pipeline orchestrator.

Usage:
    python -m eval.run                    # full eval (50 samples)
    EVAL_SAMPLE_SIZE=5 python -m eval.run  # smoke test
"""

import json
import logging
import sys

from src.client import create_client, chat
from src.models import ReActEval
from src.prompts import format_react_messages, format_steps_for_display

from eval.baseline import analyze as baseline_analyze
from eval.data import load_samples
from eval.scorer import score as compute_score
from eval.metrics import archive_results, print_report, save_results, RESULTS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def react_analyze(client, rules_text: str, data_csv: str) -> ReActEval:
    """Run ReAct constraint evaluation."""
    messages = format_react_messages(rules_text, data_csv)
    return chat(client, messages, response_format=ReActEval, temperature=0.0)


def run_eval() -> None:
    """Run the full evaluation pipeline: baseline vs ReAct, scored by exact-match F1."""
    client = create_client()
    samples = load_samples()
    total = len(samples)
    results: list[dict] = []

    existing = 0
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH, encoding="utf-8") as f:
                results = json.load(f)
            existing = len(results)
            logger.info("Resuming from %d existing results", existing)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt results file, starting fresh")
            results = []
            existing = 0

    for i, sample in enumerate(samples):
        if i < existing:
            continue

        logger.info("[%d/%d] %s (gold violations: %d)", i + 1, total, sample.category, len(sample.gold_violations))

        try:
            # Baseline
            baseline_result = baseline_analyze(client, sample.rules_text, sample.data_csv)
            baseline_ids = [v.row_id for v in baseline_result.violations]
            baseline_score = compute_score(baseline_ids, sample.gold_row_ids)
            logger.info("  Baseline: found %d violations | F1=%.3f P=%.3f R=%.3f",
                        len(baseline_ids), baseline_score.f1, baseline_score.precision, baseline_score.recall)

            # ReAct
            react_result = react_analyze(client, sample.rules_text, sample.data_csv)
            react_ids = [v.row_id for v in react_result.violations]
            react_score = compute_score(react_ids, sample.gold_row_ids)
            logger.info("  ReAct:    found %d violations | F1=%.3f P=%.3f R=%.3f (conf: %s)",
                        len(react_ids), react_score.f1, react_score.precision, react_score.recall,
                        react_result.confidence)

            result = {
                "index": i,
                "gold_label": sample.category,
                "gold_violations": list(sample.gold_row_ids),
                "baseline_violations": baseline_ids,
                "react_violations": react_ids,
                "react_confidence": react_result.confidence,
                "react_steps": format_steps_for_display(react_result.steps),
                "baseline": {
                    "tp": baseline_score.tp, "fp": baseline_score.fp, "fn": baseline_score.fn,
                    "precision": round(baseline_score.precision, 4),
                    "recall": round(baseline_score.recall, 4),
                    "f1": round(baseline_score.f1, 4),
                },
                "react": {
                    "tp": react_score.tp, "fp": react_score.fp, "fn": react_score.fn,
                    "precision": round(react_score.precision, 4),
                    "recall": round(react_score.recall, 4),
                    "f1": round(react_score.f1, 4),
                },
            }
            results.append(result)
            save_results(results)

        except Exception:
            logger.exception("  FAILED sample %d (%s)", i, sample.category)
            continue

    if not results:
        logger.error("No results collected")
        sys.exit(1)

    print_report(results)
    archived = archive_results(results)
    logger.info("Evaluation complete: %d/%d samples processed, archived to %s", len(results), total, archived)


if __name__ == "__main__":
    run_eval()
