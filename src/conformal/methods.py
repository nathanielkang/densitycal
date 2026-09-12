"""Conformal methods for regression: split, CQR, local, Mondrian, DensityCal, mixup."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

from src.models.regressors import predict_regressor


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conservative conformal residual quantile level."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n == 0:
        return 0.0
    level = min(1.0, float(np.ceil((n + 1) * (1 - alpha)) / n))
    return float(np.quantile(s, level, method="higher"))


def _split_widths(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.abs(y_cal.ravel() - yhat_cal.ravel())
    q = _finite_sample_quantile(r, alpha)
    lo = yhat_test.ravel() - q
    hi = yhat_test.ravel() + q
    return lo, hi


def _cqr_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    lo_m = GradientBoostingRegressor(
        loss="quantile",
        alpha=float(alpha / 2.0),
        random_state=seed,
        max_depth=4,
        n_estimators=200,
    )
    hi_m = GradientBoostingRegressor(
        loss="quantile",
        alpha=float(1.0 - alpha / 2.0),
        random_state=seed + 17,
        max_depth=4,
        n_estimators=200,
    )
    lo_m.fit(X_train, y_train.ravel())
    hi_m.fit(X_train, y_train.ravel())
    ql = lo_m.predict(X_cal)
    qh = hi_m.predict(X_cal)
    q_test_lo = lo_m.predict(X_test)
    q_test_hi = hi_m.predict(X_test)
    e = np.maximum(ql - y_cal.ravel(), y_cal.ravel() - qh)
    q = _finite_sample_quantile(e, alpha)
    lo = q_test_lo - q
    hi = q_test_hi + q
    return lo, hi


def _locally_adaptive_intervals(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    X_test: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Guan-style localized CP: kNN MAD of neighbor residuals (k=20)."""
    del seed
    X_cal = np.asarray(X_cal, dtype=np.float64)
    if X_cal.ndim == 1:
        X_cal = X_cal.reshape(-1, 1)
    X_test = np.asarray(X_test, dtype=np.float64)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    signed = y_cal.ravel() - yhat_cal.ravel()
    r = np.abs(signed)
    n = X_cal.shape[0]
    k = max(1, min(20, n))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_cal)
    floor = max(float(np.median(r)) * 0.05, 1e-6)

    def _mad_scales(query: np.ndarray) -> np.ndarray:
        idx = nn.kneighbors(query, return_distance=False)
        neigh = signed[idx]
        med = np.median(neigh, axis=1, keepdims=True)
        mad = np.median(np.abs(neigh - med), axis=1)
        return np.maximum(mad, floor)

    sig_cal = _mad_scales(X_cal)
    scores = r / sig_cal
    q = _finite_sample_quantile(scores, alpha)
    sig_test = _mad_scales(X_test)
    lo = yhat_test.ravel() - q * sig_test
    hi = yhat_test.ravel() + q * sig_test
    return lo, hi


def _mondrian_intervals(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    X_test: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float,
    K: int,
    n0: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_cal = X_cal.shape[0]
    k_clust = int(max(2, min(K, n_cal // max(2, n0))))
    km = KMeans(n_clusters=k_clust, random_state=seed, n_init=10)
    lab_cal = km.fit_predict(X_cal)
    lab_test = km.predict(X_test)
    r = np.abs(y_cal.ravel() - yhat_cal.ravel())
    global_q = _finite_sample_quantile(r, alpha)
    q_by: dict[int, float] = {}
    for c in range(k_clust):
        m = lab_cal == c
        if int(np.sum(m)) >= n0:
            q_by[c] = _finite_sample_quantile(r[m], alpha)
        else:
            q_by[c] = global_q
    lo = np.empty(X_test.shape[0], dtype=np.float64)
    hi = np.empty_like(lo)
    for i in range(X_test.shape[0]):
        c = int(lab_test[i])
        q = q_by.get(c, global_q)
        yh = float(yhat_test.ravel()[i])
        lo[i] = yh - q
        hi[i] = yh + q
    return lo, hi


def _assign_predicted_value_bin(
    yhat: np.ndarray, yhat_min_tr: float, yhat_max_tr: float, K: int,
) -> np.ndarray:
    """Assign each prediction to an equal-width bin 1..K on the training range (Eq. 2)."""
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    delta = (yhat_max_tr - yhat_min_tr) / max(K, 1)
    if delta < 1e-12:
        return np.ones(yhat.size, dtype=np.int32)
    clipped = np.clip(yhat, yhat_min_tr, yhat_max_tr)
    idx = np.floor((clipped - yhat_min_tr) / delta).astype(np.int32)
    idx = np.clip(idx, 0, K - 1)
    return idx + 1


def _densitycal_intervals(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float,
    K: int,
    n0: int,
    yhat_min_tr: float,
    yhat_max_tr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1: equal-width predicted-value bins + floored local--global blend."""
    y_cal = y_cal.ravel()
    yhat_cal = yhat_cal.ravel()
    r = np.abs(y_cal - yhat_cal)
    global_q = _finite_sample_quantile(r, alpha)

    s_cal = _assign_predicted_value_bin(yhat_cal, yhat_min_tr, yhat_max_tr, K)

    q_tilde: dict[int, float] = {}
    for k in range(1, K + 1):
        m = s_cal == k
        n_k = int(np.sum(m))
        if n_k >= 1:
            q_k = _finite_sample_quantile(r[m], alpha)
        else:
            q_k = global_q
        w_k = min(1.0, n_k / max(n0, 1))
        blend = w_k * q_k + (1.0 - w_k) * global_q
        q_tilde[k] = max(global_q, blend)

    s_test = _assign_predicted_value_bin(yhat_test, yhat_min_tr, yhat_max_tr, K)
    yht = yhat_test.ravel()
    half = np.array([q_tilde.get(int(s_test[i]), global_q) for i in range(yht.size)])
    lo = yht - half
    hi = yht + half
    return lo, hi


def _credo_intervals(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float,
    model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """CREDO (Cabezas et al., 2026): epistemic-aware conformal envelopes.

    Estimates epistemic uncertainty via ensemble disagreement (tree models)
    or bootstrap residual variance (MLP), then normalizes residuals by
    local epistemic scale so intervals widen in data-sparse regions.
    """
    y_cal = y_cal.ravel()
    yhat_cal = yhat_cal.ravel()
    r = np.abs(y_cal - yhat_cal)

    sigma_cal = _epistemic_sigma(model, X_cal)
    sigma_test = _epistemic_sigma(model, X_test)

    scores = r / sigma_cal
    q = _finite_sample_quantile(scores, alpha)

    yht = yhat_test.ravel()
    half = q * sigma_test
    return yht - half, yht + half


def _epistemic_sigma(model: Any, X: np.ndarray) -> np.ndarray:
    """Estimate per-point epistemic uncertainty from the fitted model."""
    X = np.asarray(X, dtype=np.float64)
    inner = getattr(model, "_model", model)

    if hasattr(inner, "estimators_") and hasattr(inner.estimators_[0], "predict"):
        preds = np.column_stack([t.predict(X) for t in inner.estimators_])
        sigma = np.std(preds, axis=1)
        return np.clip(sigma, 1e-6, None)

    if hasattr(inner, "get_booster"):
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix(X)
            booster = inner.get_booster()
            n_trees = int(booster.num_boosted_rounds())
            step = max(1, n_trees // 10)
            preds_at = []
            for t in range(step, n_trees + 1, step):
                preds_at.append(booster.predict(dmat, iteration_range=(0, t)))
            if len(preds_at) >= 2:
                preds_mat = np.column_stack(preds_at)
                sigma = np.std(preds_mat, axis=1)
                return np.clip(sigma, 1e-6, None)
        except Exception:
            pass

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(20, X.shape[0]), algorithm="auto").fit(X)
    yhat = np.asarray(model.predict(X), dtype=np.float64).ravel()
    _, idx = nn.kneighbors(X)
    sigma = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        neighbor_preds = yhat[idx[i]]
        sigma[i] = max(np.std(neighbor_preds), 1e-6)
    return sigma


def _calibrated_mixup_intervals(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float,
    seed: int,
    n_aug: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.abs(y_cal.ravel() - yhat_cal.ravel())
    n = r.size
    rng = np.random.default_rng(seed)
    pool = list(r)
    for _ in range(n_aug):
        a, b = rng.integers(0, n, size=2)
        lam = float(rng.random())
        pool.append(float(lam * r[a] + (1.0 - lam) * r[b]))
    q = _finite_sample_quantile(np.asarray(pool, dtype=np.float64), alpha)
    yht = yhat_test.ravel()
    return yht - q, yht + q


def predict_intervals(
    method: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    K: int = 5,
    n0: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch conformal method; returns (lower, upper) on test."""
    method = method.lower().strip()
    yhat_cal = predict_regressor(model, X_cal)
    yhat_test = predict_regressor(model, X_test)

    if method in ("split", "splitconformal", "split_conformal"):
        return _split_widths(y_cal, yhat_cal, yhat_test, alpha)
    if method == "cqr":
        return _cqr_intervals(X_train, y_train, X_cal, y_cal, X_test, alpha, seed)
    if method in ("locally_adaptive", "local", "localized"):
        return _locally_adaptive_intervals(
            X_cal, y_cal, yhat_cal, X_test, yhat_test, alpha, seed
        )
    if method in ("mondrian",):
        return _mondrian_intervals(
            X_cal, y_cal, yhat_cal, X_test, yhat_test, alpha, K, n0, seed
        )
    if method in ("densitycal", "density_cal", "density"):
        yhat_train = predict_regressor(model, X_train)
        yhat_min_tr = float(np.min(yhat_train))
        yhat_max_tr = float(np.max(yhat_train))
        return _densitycal_intervals(
            y_cal, yhat_cal, yhat_test, alpha, K, n0,
            yhat_min_tr, yhat_max_tr,
        )
    if method in ("credo",):
        return _credo_intervals(
            y_cal, yhat_cal, X_cal, X_test, yhat_test, alpha, model,
        )
    if method in ("calibrated_mixup", "mixup"):
        return _calibrated_mixup_intervals(y_cal, yhat_cal, yhat_test, alpha, seed)
    raise ValueError(f"Unknown conformal method: {method}")
