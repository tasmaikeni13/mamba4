"""Trial-level uncertainty estimates; observations within a trial are clustered."""

import numpy as np
from scipy.stats import norm, t


def mean_interval(values, confidence=0.95):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("At least two finite, independent trials required")
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    margin = float(t.ppf((1 + confidence) / 2, len(x) - 1) * se)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "se": se,
        "low": float(x.mean() - margin),
        "high": float(x.mean() + margin),
    }


def paired_bootstrap(a, b, rng, draws=2000, family_size=1):
    """Positive differences mean a has higher loss than b.

    Return both ordinary and Bonferroni simultaneous percentile intervals.
    A paired t interval is also reported for tail-resolution at large family_size.
    """
    difference = np.asarray(a) - np.asarray(b)
    estimates = difference[
        rng.integers(0, len(difference), (draws, len(difference)))
    ].mean(axis=1)
    adjusted = 0.05 / family_size
    result = mean_interval(difference, confidence=1 - adjusted)
    result["bootstrap_low"], result["bootstrap_high"] = map(
        float, np.quantile(estimates, [0.025, 0.975])
    )
    result["simultaneous_bootstrap_low"], result["simultaneous_bootstrap_high"] = map(
        float, np.quantile(estimates, [adjusted / 2, 1 - adjusted / 2])
    )
    return result


def wilson(successes, n, confidence=0.95):
    z = norm.ppf((1 + confidence) / 2)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "successes": int(successes),
        "rate": p,
        "low": center - half,
        "high": center + half,
    }


def holm(p_values):
    """Holm step-down adjusted p-values, valid under arbitrary dependence."""
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1)
    return out
