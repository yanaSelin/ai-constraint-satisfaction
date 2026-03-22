"""Exact-match F1 scorer for constraint violation detection.

No LLM judge needed — answers are deterministic row IDs.
"""

from dataclasses import dataclass


@dataclass
class ViolationScore:
    """Precision, recall, and F1 for a single sample."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    @property
    def exact_match(self) -> bool:
        """True if predicted set exactly equals gold set."""
        return self.fp == 0 and self.fn == 0


def score(predicted_ids: list[str], gold_ids: set[str]) -> ViolationScore:
    """Compute precision, recall, F1 comparing predicted vs gold violation row IDs.

    Args:
        predicted_ids: Row IDs reported as violations by the system.
        gold_ids: Ground-truth set of violating row IDs.

    Returns:
        ViolationScore with tp/fp/fn and derived metrics.
    """
    predicted_set = set(predicted_ids)
    tp = len(predicted_set & gold_ids)
    fp = len(predicted_set - gold_ids)
    fn = len(gold_ids - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gold_ids else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ViolationScore(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)
