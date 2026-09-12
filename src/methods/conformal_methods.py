"""
Conformal prediction methods for regression with a common interface.

Dependencies: numpy, scipy, scikit-learn only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde, median_abs_deviation
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal (1-alpha) quantile with finite-sample correction."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = int(scores.size)
    if n <= 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = int(np.clip(k, 1, n))
    sorted_s = np.sort(scores)
    return float(sorted_s[k - 1])


def _quantile_compat(a: np.ndarray, qs) -> np.ndarray:
    """np.quantile with NumPy 1.x / 2.x compatible keyword."""
    a = np.asarray(a, dtype=float)
    try:
        return np.quantile(a, qs, method="linear")
    except TypeError:
        return np.quantile(a, qs, interpolation="linear")


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _assign_predicted_value_bin(
    yhat: np.ndarray, yhat_min: float, yhat_max: float, K: int,
) -> np.ndarray:
    yhat = np.asarray(yhat, dtype=float).ravel()
    delta = (yhat_max - yhat_min) / max(K, 1)
    if delta < 1e-12:
        return np.zeros(yhat.size, dtype=int)
    clipped = np.clip(yhat, yhat_min, yhat_max)
    idx = np.floor((clipped - yhat_min) / delta).astype(int)
    return np.clip(idx, 0, K - 1)


class ConformalMethod:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def calibrate(self, y_cal, y_pred_cal):
        """Compute calibration quantile(s) from calibration residuals."""
        raise NotImplementedError

    def predict_intervals(self, y_pred_test):
        """Return (lower, upper) arrays for test predictions."""
        raise NotImplementedError


class SplitConformal(ConformalMethod):
    """Standard split conformal: symmetric interval from global |residual| quantile."""

    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha=alpha)
        self._half_width: float = 0.0

    def calibrate(self, y_cal, y_pred_cal):
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        y_pred_cal = np.asarray(y_pred_cal, dtype=float).ravel()
        abs_res = np.abs(y_cal - y_pred_cal)
        self._half_width = _finite_sample_quantile(abs_res, self.alpha)

    def predict_intervals(self, y_pred_test):
        y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()
        lo = y_pred_test - self._half_width
        hi = y_pred_test + self._half_width
        return lo, hi


class CQR(ConformalMethod):
    """
    Conformalized quantile regression.
    Nonconformity: max(q_low - y, y - q_high).
    If quantile predictions are missing, fall back to residual / MAD normalization.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha=alpha)
        self._conformal_delta: float = 0.0
        self._use_cqr_scores: bool = True
        self._mad: float = 1.0

    def calibrate(
        self,
        y_cal,
        y_pred_cal,
        y_pred_cal_lower=None,
        y_pred_cal_upper=None,
    ):
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        y_pred_cal = np.asarray(y_pred_cal, dtype=float).ravel()
        if y_pred_cal_lower is not None and y_pred_cal_upper is not None:
            q_lo = np.asarray(y_pred_cal_lower, dtype=float).ravel()
            q_hi = np.asarray(y_pred_cal_upper, dtype=float).ravel()
            scores = np.maximum(q_lo - y_cal, y_cal - q_hi)
            scores = np.maximum(scores, 0.0)
            self._conformal_delta = _finite_sample_quantile(scores, self.alpha)
            self._use_cqr_scores = True
        else:
            resid = np.abs(y_cal - y_pred_cal)
            mad = float(median_abs_deviation(resid, scale=1.0, nan_policy="omit"))
            mad = max(mad, 1e-12)
            scores = resid / mad
            self._conformal_delta = _finite_sample_quantile(scores, self.alpha)
            self._mad = mad
            self._use_cqr_scores = False

    def predict_intervals(
        self,
        y_pred_test,
        y_pred_test_lower=None,
        y_pred_test_upper=None,
    ):
        y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()
        if self._use_cqr_scores:
            if y_pred_test_lower is None or y_pred_test_upper is None:
                raise ValueError(
                    "CQR was calibrated with quantile heads; provide "
                    "y_pred_test_lower and y_pred_test_upper."
                )
            q_lo = np.asarray(y_pred_test_lower, dtype=float).ravel()
            q_hi = np.asarray(y_pred_test_upper, dtype=float).ravel()
            lo = q_lo - self._conformal_delta
            hi = q_hi + self._conformal_delta
            return lo, hi
        half = self._conformal_delta * self._mad
        return y_pred_test - half, y_pred_test + half


class LocallyAdaptiveCP(ConformalMethod):
    """Scale |residual| by local MAD of neighbor residuals in feature space (k=20)."""

    def __init__(self, alpha: float = 0.1, k_neighbors: int = 20):
        super().__init__(alpha=alpha)
        self.k_neighbors = int(k_neighbors)
        self._Q: float = 0.0
        self._X_cal: np.ndarray | None = None
        self._residuals_cal: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None

    def calibrate(self, y_cal, y_pred_cal, X_cal):
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        y_pred_cal = np.asarray(y_pred_cal, dtype=float).ravel()
        X_cal = _as_2d(X_cal)
        n = X_cal.shape[0]
        k = max(1, min(self.k_neighbors, n))
        self._X_cal = X_cal
        self._residuals_cal = y_cal - y_pred_cal
        self._nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_cal)
        neigh_idx = self._nn.kneighbors(X_cal, return_distance=False)
        scales = np.empty(n, dtype=float)
        for i in range(n):
            neigh_res = self._residuals_cal[neigh_idx[i]]
            med = float(np.median(neigh_res))
            mad = float(np.median(np.abs(neigh_res - med)))
            scales[i] = max(mad, 1e-12)
        scores = np.abs(self._residuals_cal) / scales
        self._Q = _finite_sample_quantile(scores, self.alpha)

    def predict_intervals(self, y_pred_test, X_test):
        if self._nn is None or self._X_cal is None or self._residuals_cal is None:
            raise RuntimeError("Call calibrate before predict_intervals.")
        y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()
        X_test = _as_2d(X_test)
        k = self._nn.n_neighbors
        neigh_idx = self._nn.kneighbors(X_test, return_distance=False)
        m = X_test.shape[0]
        half_widths = np.empty(m, dtype=float)
        for j in range(m):
            neigh_res = self._residuals_cal[neigh_idx[j]]
            med = float(np.median(neigh_res))
            mad = float(np.median(np.abs(neigh_res - med)))
            mad = max(mad, 1e-12)
            half_widths[j] = self._Q * mad
        lo = y_pred_test - half_widths
        hi = y_pred_test + half_widths
        return lo, hi


class MondrianCP(ConformalMethod):
    """Feature-space K-means clusters; per-cluster split conformal on |residuals|."""

    def __init__(self, alpha: float = 0.1, K: int = 5):
        super().__init__(alpha=alpha)
        self.K = int(K)
        self._kmeans: KMeans | None = None
        self._cluster_half_width: np.ndarray | None = None
        self._global_half_width: float = 0.0

    def calibrate(self, y_cal, y_pred_cal, X_cal, K: int | None = None):
        if K is not None:
            self.K = int(K)
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        y_pred_cal = np.asarray(y_pred_cal, dtype=float).ravel()
        X_cal = _as_2d(X_cal)
        n = X_cal.shape[0]
        abs_res = np.abs(y_cal - y_pred_cal)
        self._global_half_width = _finite_sample_quantile(abs_res, self.alpha)
        n_clusters = max(1, min(self.K, n))
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=0,
        ).fit(X_cal)
        labels = self._kmeans.predict(X_cal)
        self._cluster_half_width = np.full(n_clusters, self._global_half_width, dtype=float)
        for c in range(n_clusters):
            mask = labels == c
            if not np.any(mask):
                continue
            q_c = _finite_sample_quantile(abs_res[mask], self.alpha)
            self._cluster_half_width[c] = q_c

    def predict_intervals(self, y_pred_test, X_test):
        if self._kmeans is None or self._cluster_half_width is None:
            raise RuntimeError("Call calibrate before predict_intervals.")
        y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()
        X_test = _as_2d(X_test)
        lab = self._kmeans.predict(X_test)
        half = self._cluster_half_width[lab]
        lo = y_pred_test - half
        hi = y_pred_test + half
        return lo, hi


class DensityCal(ConformalMethod):
    """Predicted-value bins plus a one-sided floored local--global blend."""

    def __init__(self, alpha: float = 0.1, K: int = 5, n0: int = 30):
        super().__init__(alpha=alpha)
        self.K = int(K)
        self.n0 = int(n0)
        self._yhat_min: float = 0.0
        self._yhat_max: float = 1.0
        self._q_tilde: np.ndarray | None = None
        self._q_global: float = 0.0

    def calibrate(self, y_cal, y_pred_cal, yhat_min_tr=None, yhat_max_tr=None):
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        y_pred_cal = np.asarray(y_pred_cal, dtype=float).ravel()
        n = y_cal.size
        abs_res = np.abs(y_cal - y_pred_cal)
        self._q_global = _finite_sample_quantile(abs_res, self.alpha)
        self._yhat_min = float(yhat_min_tr) if yhat_min_tr is not None else float(np.min(y_pred_cal))
        self._yhat_max = float(yhat_max_tr) if yhat_max_tr is not None else float(np.max(y_pred_cal))
        K = max(1, min(self.K, max(n, 1)))
        if n < 2:
            self._q_tilde = np.array([self._q_global], dtype=float)
            return
        s_cal = _assign_predicted_value_bin(
            y_pred_cal, self._yhat_min, self._yhat_max, K
        )
        self._q_tilde = np.empty(K, dtype=float)
        for k in range(K):
            mask = s_cal == k
            n_k = int(np.sum(mask))
            q_loc = _finite_sample_quantile(abs_res[mask], self.alpha) if n_k > 0 else self._q_global
            w = min(1.0, n_k / max(self.n0, 1))
            blend = w * q_loc + (1.0 - w) * self._q_global
            self._q_tilde[k] = max(self._q_global, blend)

    def predict_intervals(self, y_pred_test):
        if self._q_tilde is None:
            raise RuntimeError("Call calibrate before predict_intervals.")
        y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()
        K = int(self._q_tilde.size)
        strata = _assign_predicted_value_bin(
            y_pred_test, self._yhat_min, self._yhat_max, K
        )
        half = self._q_tilde[strata]
        return y_pred_test - half, y_pred_test + half


class OracleLocalCP(ConformalMethod):
    """Same stratification as DensityCal but assigns test stratum using true y_test (oracle)."""

    def __init__(self, alpha: float = 0.1, K: int = 5, n0: int = 30):
        super().__init__(alpha=alpha)
        self.K = int(K)
        self.n0 = int(n0)
        self._kde: gaussian_kde | None = None
        self._density_edges: np.ndarray | None = None
        self._q_tilde: np.ndarray | None = None
        self._q_global: float = 0.0

    def calibrate(self, y_cal, y_pred_cal):
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        y_pred_cal = np.asarray(y_pred_cal, dtype=float).ravel()
        n = y_cal.size
        abs_res = np.abs(y_cal - y_pred_cal)
        self._q_global = _finite_sample_quantile(abs_res, self.alpha)
        if n < 2:
            self._kde = None
            self._density_edges = np.array([-np.inf, np.inf], dtype=float)
            self._q_tilde = np.array([self._q_global], dtype=float)
            return
        self._kde = gaussian_kde(y_cal, bw_method="silverman")
        dens_cal = np.asarray(self._kde(y_cal), dtype=float)
        K = max(1, min(self.K, n))
        self._density_edges = _quantile_compat(
            dens_cal, np.linspace(0.0, 1.0, K + 1)
        )
        self._q_tilde = np.empty(K, dtype=float)
        for k in range(K):
            low, high = self._density_edges[k], self._density_edges[k + 1]
            if k == K - 1:
                mask = (dens_cal >= low) & (dens_cal <= high)
            else:
                mask = (dens_cal >= low) & (dens_cal < high)
            n_k = int(np.sum(mask))
            q_loc = _finite_sample_quantile(abs_res[mask], self.alpha) if n_k > 0 else self._q_global
            w = min(1.0, n_k / max(self.n0, 1))
            blend = w * q_loc + (1.0 - w) * self._q_global
            self._q_tilde[k] = max(self._q_global, blend)

    def _stratum_from_density(self, d: np.ndarray) -> np.ndarray:
        K = int(self._q_tilde.size)
        edges = self._density_edges
        if edges is None or self._q_tilde is None:
            raise RuntimeError("Call calibrate before predict_intervals.")
        d = np.asarray(d, dtype=float).ravel()
        strata = np.full(d.shape[0], K - 1, dtype=int)
        for kk in range(K - 1):
            low, high = float(edges[kk]), float(edges[kk + 1])
            m = (d >= low) & (d < high)
            strata[m] = kk
        low, high = float(edges[K - 1]), float(edges[K])
        m = (d >= low) & (d <= high)
        strata[m] = K - 1
        return strata

    def predict_intervals(self, y_pred_test, y_test):
        if self._q_tilde is None:
            raise RuntimeError("Call calibrate before predict_intervals.")
        y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()
        y_test = np.asarray(y_test, dtype=float).ravel()
        if self._kde is None:
            half = np.full(y_pred_test.shape[0], self._q_global, dtype=float)
            return y_pred_test - half, y_pred_test + half
        d_oracle = np.asarray(self._kde(y_test), dtype=float)
        strata = self._stratum_from_density(d_oracle)
        half = self._q_tilde[strata]
        return y_pred_test - half, y_pred_test + half


class CalibratedMixupCP(ConformalMethod):
    """
    Lightweight mixup oversampling on the calibration set (rare quantile bins),
    then standard split conformal on augmented (y, y_pred) pairs.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        K_bins: int = 5,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(alpha=alpha)
        self.K_bins = int(K_bins)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._half_width: float = 0.0

    def calibrate(self, y_cal, y_pred_cal, X_cal=None, K_bins: int | None = None):
        if K_bins is not None:
            self.K_bins = int(K_bins)
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        y_pred_cal = np.asarray(y_pred_cal, dtype=float).ravel()
        n = y_cal.size
        if n == 0:
            self._half_width = 0.0
            return
        edges = _quantile_compat(y_cal, np.linspace(0.0, 1.0, self.K_bins + 1))
        bin_id = np.searchsorted(edges, y_cal, side="right") - 1
        bin_id = np.clip(bin_id, 0, self.K_bins - 1)
        counts = np.bincount(bin_id, minlength=self.K_bins)
        mean_c = float(np.mean(counts[counts > 0])) if np.any(counts > 0) else float(n)
        under = counts < mean_c
        y_aug = list(y_cal)
        p_aug = list(y_pred_cal)
        X_cal_arr = _as_2d(X_cal) if X_cal is not None else None
        for b in range(self.K_bins):
            if not under[b]:
                continue
            idx_b = np.where(bin_id == b)[0]
            if idx_b.size < 2:
                continue
            need = int(max(1, round(mean_c - counts[b])))
            nn = None
            X_b = None
            if X_cal_arr is not None:
                X_b = X_cal_arr[idx_b]
                nn = NearestNeighbors(
                    n_neighbors=min(20, idx_b.size), algorithm="auto"
                ).fit(X_b)
            for _ in range(need):
                i1 = int(self._rng.choice(idx_b))
                if X_cal_arr is not None and nn is not None and X_b is not None:
                    _, neigh = nn.kneighbors(X_cal_arr[i1 : i1 + 1], return_distance=True)
                    neigh_local = neigh.ravel()
                    i2 = i1
                    for jl in neigh_local:
                        cand = int(idx_b[int(jl)])
                        if cand != i1:
                            i2 = cand
                            break
                    if i2 == i1:
                        choices = idx_b[idx_b != i1]
                        i2 = int(self._rng.choice(choices))
                else:
                    i2 = int(self._rng.choice(idx_b))
                    if i2 == i1 and idx_b.size > 1:
                        choices = idx_b[idx_b != i1]
                        i2 = int(self._rng.choice(choices))
                lam = float(self._rng.beta(0.4, 0.4))
                y_new = lam * y_cal[i1] + (1.0 - lam) * y_cal[i2]
                p_new = lam * y_pred_cal[i1] + (1.0 - lam) * y_pred_cal[i2]
                y_aug.append(y_new)
                p_aug.append(p_new)
        y_aug = np.asarray(y_aug, dtype=float)
        p_aug = np.asarray(p_aug, dtype=float)
        abs_res = np.abs(y_aug - p_aug)
        self._half_width = _finite_sample_quantile(abs_res, self.alpha)

    def predict_intervals(self, y_pred_test):
        y_pred_test = np.asarray(y_pred_test, dtype=float).ravel()
        lo = y_pred_test - self._half_width
        hi = y_pred_test + self._half_width
        return lo, hi


def get_method(name: str, alpha: float = 0.1, **kwargs) -> ConformalMethod:
    methods = {
        "split": SplitConformal,
        "cqr": CQR,
        "locally_adaptive": LocallyAdaptiveCP,
        "mondrian": MondrianCP,
        "densitycal": DensityCal,
        "oracle": OracleLocalCP,
        "calibrated_mixup": CalibratedMixupCP,
    }
    key = name.lower()
    if key not in methods:
        raise ValueError(f"Unknown method name: {name!r}")
    return methods[key](alpha=alpha, **kwargs)
