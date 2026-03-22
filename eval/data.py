"""Dataset loading and stratified sampling for evaluation."""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "constraints_golden.json"


@dataclass
class GoldViolation:
    """Single gold-standard violation."""

    row_id: str
    rules_violated: list[str]
    difficulty: str  # easy | medium | hard


@dataclass
class Sample:
    """Single evaluation sample."""

    category: str
    rules: list[str]
    data_csv: str
    gold_violations: list[GoldViolation] = field(default_factory=list)

    @property
    def rules_text(self) -> str:
        """Format rules as numbered list for prompts."""
        return "\n".join(self.rules)

    @property
    def gold_row_ids(self) -> set[str]:
        """Set of row IDs that have gold violations."""
        return {v.row_id for v in self.gold_violations}


def load_samples(
    json_path: Path = DATA_PATH,
    n: int | None = None,
    seed: int = 42,
    categories: list[str] | None = None,
) -> list[Sample]:
    """Load and stratify-sample from the golden constraints dataset.

    Args:
        json_path: Path to JSON file with constraint satisfaction entries.
        n: Total samples to draw (stratified across categories).
            Defaults to EVAL_SAMPLE_SIZE env var or 50.
        seed: Random seed for reproducibility.
        categories: Optional list of categories to include.

    Returns:
        List of Sample objects, stratified across categories.
    """
    if n is None:
        n = int(os.environ.get("EVAL_SAMPLE_SIZE", "50"))

    if categories is None:
        cat_env = os.environ.get("EVAL_CATEGORIES", "")
        if cat_env:
            categories = [c.strip() for c in cat_env.split(",") if c.strip()]

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    by_category: dict[str, list[Sample]] = {}
    for entry in raw:
        cat = entry["category"]
        if categories and cat not in categories:
            continue
        gold = [
            GoldViolation(row_id=v["row_id"], rules_violated=v["rules_violated"], difficulty=v["difficulty"])
            for v in entry["gold_violations"]
        ]
        sample = Sample(category=cat, rules=entry["rules"], data_csv=entry["data_csv"], gold_violations=gold)
        by_category.setdefault(cat, []).append(sample)

    num_cats = len(by_category)
    per_cat = max(1, n // num_cats)
    actual_total = min(n, per_cat * num_cats)
    logger.info("Loaded %d categories, sampling %d per category (%d total)", num_cats, per_cat, actual_total)

    rng = random.Random(seed)
    samples: list[Sample] = []
    for cat in sorted(by_category):
        pool = by_category[cat]
        k = min(per_cat, len(pool))
        samples.extend(rng.sample(pool, k))

    if len(samples) > n:
        rng.shuffle(samples)
        samples = samples[:n]

    rng.shuffle(samples)
    return samples
