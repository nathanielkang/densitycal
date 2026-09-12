"""Conformal prediction evaluation metrics for imbalanced regression."""

from __future__ import annotations

from typing import Any

import numpy as np


def assign_predicted_value_strata(
    yhat: np.ndarray,
    yhat_min_tr: float,
    yhat_max_tr: float,
    K: int = 5,
) -> np.ndarray:
    """Assign each test point to equal-width predicted-value bin 1..K.

    Uses the same bin boundaries as Algorithm 1 (Eq. 2) so that
    evaluation strata match the theoretical conditional Cov(B_k).
    Bin 1 corresponds to the lowest predicted-value range.
    """
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    n = yhat.size
    delta = (yhat_max_tr - yhat_min_tr) / max(K, 1)
    if delta < 1e-12:
        return np.ones(n, dtype=np.int32)
    clipped = np.clip(yhat, yhat_min_tr, yhat_max_tr)
    idx = np.floor((clipped - yhat_min_tr) / delta).astype(np.int32)
    idx = np.clip(idx, 0, K - 1)
    return idx + 1


def compute_strat_coverage(
    y_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y_density_strata: np.ndarray,
    K: int = 5,
) -> dict[str, float]:
    """Per-stratum coverage. Returns dict {stratum_k: coverage_float}."""
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    s = np.asarray(y_density_strata, dtype=np.int32).ravel()
    out: dict[str, float] = {}
    for k in range(1, K + 1):
        m = s == k
        if not np.any(m):
            out[f"stratum_{k}"] = float("nan")
        else:
            inside = (y_test[m] >= lower[m]) & (y_test[m] <= upper[m])
            out[f"stratum_{k}"] = float(np.mean(inside))
    return out


def compute_covgap(strat_coverages: dict[str, float], nominal: float = 0.9) -> float:
    """Mean |StratCov(Bk) - nominal| across strata."""

    def _key(item: tuple[str, float]) -> int:
        k = item[0]
        if k.startswith("stratum_"):
            try:
                return int(k.split("_", 1)[1])
            except ValueError:
                return 0
        return 0

    vals: list[float] = []
    for _, v in sorted(strat_coverages.items(), key=_key):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        vals.append(abs(float(v) - nominal))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def compute_avg_width(
    lower: np.ndarray,
    upper: np.ndarray,
    y_density_strata: np.ndarray | None = None,
    K: int = 5,
) -> float | dict[str, float]:
    """Average interval width, optionally per stratum."""
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    w = upper - lower
    if y_density_strata is None:
        return float(np.mean(w))
    s = np.asarray(y_density_strata, dtype=np.int32).ravel()
    out: dict[str, float] = {}
    for k in range(1, K + 1):
        m = s == k
        out[f"stratum_{k}"] = float(np.mean(w[m])) if np.any(m) else float("nan")
    return out


def compute_width_ratio(
    lower: np.ndarray,
    upper: np.ndarray,
    y_density_strata: np.ndarray,
    K: int = 5,
) -> float:
    """Width(rare) / Width(common)."""
    per = compute_avg_width(lower, upper, y_density_strata, K=K)
    if not isinstance(per, dict):
        return float("nan")
    w1 = per.get("stratum_1", float("nan"))
    wk = per.get(f"stratum_{K}", float("nan"))
    if np.isnan(w1) or np.isnan(wk) or wk < 1e-12:
        return float("nan")
    return float(w1 / wk)


def compute_marginal_coverage(
    y_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Overall coverage = fraction of y in [lower, upper]."""
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    inside = (y_test >= lower) & (y_test <= upper)
    return float(np.mean(inside))


def compute_mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width."""
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    return float(np.mean(upper - lower))


def evaluate_all(
    y_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    yhat_test: np.ndarray,
    yhat_min_tr: float,
    yhat_max_tr: float,
    K: int = 5,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Returns a dict with all metrics.

    Strata are defined by equal-width predicted-value bins (matching the
    algorithm's bin boundaries) so the measured per-stratum coverage
    corresponds to the theoretical Cov(B_k) = P(Y in C(X) | yhat(X) in B_k).
    """
    nominal = 1.0 - alpha
    strata = assign_predicted_value_strata(yhat_test, yhat_min_tr, yhat_max_tr, K)
    strat_cov = compute_strat_coverage(y_test, lower, upper, strata, K=K)
    per_w = compute_avg_width(lower, upper, strata, K=K)
    if isinstance(per_w, dict):
        w_rare = float(per_w.get("stratum_1", float("nan")))
        w_common = float(per_w.get(f"stratum_{K}", float("nan")))
    else:
        w_rare = float("nan")
        w_common = float("nan")
    return {
        "strat_coverage": strat_cov,
        "covgap": compute_covgap(strat_cov, nominal=nominal),
        "marginal_coverage": compute_marginal_coverage(y_test, lower, upper),
        "mpiw": compute_mpiw(lower, upper),
        "avg_width_rare": w_rare,
        "avg_width_common": w_common,
        "width_ratio": compute_width_ratio(lower, upper, strata, K=K),
        "predicted_value_strata": strata,
    }
