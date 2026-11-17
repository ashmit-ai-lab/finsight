"""Metrics: page recall at k, reciprocal rank, number accuracy, and bootstrap confidence intervals."""
import random
import re

SCALE = {"units": 1, "ones": 1, "unit": 1, "thousands": 1e3, "thousand": 1e3,
         "millions": 1e6, "million": 1e6, "billions": 1e9, "billion": 1e9}


def parse_number(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-")
    m = re.search(r"\d[\d,]*(?:\.\d+)?|\.\d+", s)
    if not m:
        return None
    v = float(m.group().replace(",", ""))
    return -v if neg else v


def to_absolute(value, scale) -> float:
    v = parse_number(value)
    if v is None:
        return None
    return v * SCALE.get(str(scale or "units").strip().lower(), 1)


def number_correct(pred_value, pred_scale, truth: float, unit: str, rel_tol: float = 0.005) -> bool:
    if unit == "USD/shares":  # per share numbers are never scaled
        pred = parse_number(pred_value)
    else:
        pred = to_absolute(pred_value, pred_scale)
    if pred is None:
        return False
    return abs(abs(pred) - abs(truth)) <= rel_tol * max(abs(truth), 1e-9)


def recall_at_k(pages: list, gold: set, k: int) -> float:
    return float(bool(set(pages[:k]) & gold))


def reciprocal_rank(pages: list, gold: set) -> float:
    for i, p in enumerate(pages):
        if p in gold:
            return 1.0 / (i + 1)
    return 0.0


def bootstrap_ci(values: list, n: int = 2000, seed: int = 0):
    """95% interval for the mean, so small differences between methods are not over read."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]
