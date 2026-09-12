"""
Tabular regression datasets with skewed targets (OpenML / sklearn + fallbacks).

Print messages use ASCII only for Windows cp949 consoles.
"""

from __future__ import annotations

import io
import warnings
import zipfile
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ArrayPair = Tuple[np.ndarray, np.ndarray]

DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "abalone": {
        "n": 4177,
        "d": 8,
        "description": "Abalone age (OpenML 183); target: shell ring count (skewed).",
    },
    "california": {
        "n": 20640,
        "d": 8,
        "description": "California housing; target: median house value.",
    },
    "bike": {
        "n": 731,
        "d": 12,
        "description": "Bike sharing demand (OpenML 44063 / UCI); target: daily rental count.",
    },
    "cpu": {
        "n": 8192,
        "d": 21,
        "description": "Computer activity (OpenML 573 cpu_act; 761 is non-numeric on sklearn mirror); target: usr.",
    },
    "wine": {
        "n": 4898,
        "d": 11,
        "description": "Wine quality (OpenML 287); target: sensory quality score.",
    },
    "energy": {
        "n": 768,
        "d": 9,
        "description": "Building energy efficiency (OpenML 1472 UCI); target: Y1 heating load.",
    },
    "airfoil": {
        "n": 1503,
        "d": 5,
        "description": "NASA airfoil noise (OpenML name airfoil_self_noise v1 / id 43919); target: sound pressure.",
    },
    "superconductor": {
        "n": 21263,
        "d": 81,
        "description": "Superconductivity (OpenML 44964); target: critical temperature.",
    },
}

_DATASET_ORDER: List[str] = [
    "abalone",
    "california",
    "bike",
    "cpu",
    "wine",
    "energy",
    "airfoil",
    "superconductor",
]

# Backward-compatible registry for scripts that iterate dataset keys.
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {name: {"name": name} for name in _DATASET_ORDER}


def list_datasets() -> List[str]:
    return list(_DATASET_ORDER)


def _synthetic_log_normal(seed: int = 42, n_samples: int = 2500, n_features: int = 8) -> ArrayPair:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = np.exp(rng.standard_normal(n_samples) * 0.85 + 1.4)
    return X.astype(np.float64), y.astype(np.float64)


def _drop_nan_rows(X: np.ndarray, y: np.ndarray) -> ArrayPair:
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def _frame_to_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    X_work = df.copy()
    for col in X_work.columns:
        if X_work[col].dtype == object or str(X_work[col].dtype).startswith("category"):
            X_work[col] = pd.Categorical(X_work[col]).codes
        else:
            X_work[col] = pd.to_numeric(X_work[col], errors="coerce")
    return X_work.to_numpy(dtype=np.float64)


def _openml_xy(data_id: int, target_names: List[str]) -> ArrayPair:
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")

    df = data.data.copy()
    lower_map = {str(c).lower(): c for c in df.columns}
    target_col = None
    for name in target_names:
        key = name.lower()
        if key in lower_map:
            target_col = lower_map[key]
            break

    if target_col is not None:
        y_series = df[target_col]
        X_df = df.drop(columns=[target_col])
    elif data.target is not None:
        y_series = pd.Series(np.asarray(data.target))
        X_df = df
    else:
        raise ValueError("Could not resolve target column")

    y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=np.float64)
    X = _frame_to_numeric_matrix(X_df)
    return _drop_nan_rows(X, y)


def _try_openml(data_id: int, target_names: List[str]) -> ArrayPair | None:
    try:
        X, y = _openml_xy(data_id, target_names)
        if X.shape[0] < 3:
            return None
        return X, y
    except Exception:
        return None


def _try_openml_named(
    name: str,
    version: int,
    target_names: List[str],
) -> ArrayPair | None:
    from sklearn.datasets import fetch_openml

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
        df = data.data.copy()
        lower_map = {str(c).lower(): c for c in df.columns}
        target_col = None
        for tn in target_names:
            if tn.lower() in lower_map:
                target_col = lower_map[tn.lower()]
                break
        if target_col is not None:
            y_series = df[target_col]
            X_df = df.drop(columns=[target_col])
        elif data.target is not None:
            y_series = pd.Series(np.asarray(data.target))
            X_df = df
        else:
            return None
        y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=np.float64)
        X = _frame_to_numeric_matrix(X_df)
        X, y = _drop_nan_rows(X, y)
        if X.shape[0] < 3:
            return None
        return X, y
    except Exception:
        return None


def _load_abalone() -> ArrayPair:
    Xy = _try_openml(183, ["Rings", "rings"])
    if Xy is not None:
        return Xy
    try:
        from sklearn.datasets import fetch_openml

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = fetch_openml(name="abalone", version=1, as_frame=True, parser="auto")
        df = data.data.copy()
        if data.target is not None:
            y = pd.to_numeric(data.target, errors="coerce").to_numpy(dtype=np.float64)
            X = _frame_to_numeric_matrix(df)
            return _drop_nan_rows(X, y)
    except Exception:
        pass
    print("[datasets] abalone: OpenML/sklearn failed, using synthetic fallback.")
    return _synthetic_log_normal(seed=183, n_features=8)


def _load_california() -> ArrayPair:
    try:
        from sklearn.datasets import fetch_california_housing

        bunch = fetch_california_housing()
        X = np.asarray(bunch.data, dtype=np.float64)
        y = np.asarray(bunch.target, dtype=np.float64).ravel()
        return _drop_nan_rows(X, y)
    except Exception:
        print("[datasets] california: sklearn failed, using synthetic fallback.")
        return _synthetic_log_normal(seed=440, n_features=8)


def _load_bike() -> ArrayPair:
    Xy = _try_openml(44063, ["cnt", "Count"])
    if Xy is not None:
        return Xy
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    try:
        import urllib.request

        raw = urllib.request.urlopen(url, timeout=120).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        day_name = next((n for n in zf.namelist() if n.lower().endswith("day.csv")), None)
        if day_name is not None:
            with zf.open(day_name) as f:
                df = pd.read_csv(f)
            if "cnt" in df.columns:
                y = pd.to_numeric(df["cnt"], errors="coerce").to_numpy(dtype=np.float64)
                X_df = df.drop(columns=["cnt"])
                X = _frame_to_numeric_matrix(X_df)
                return _drop_nan_rows(X, y)
    except Exception:
        pass
    print("[datasets] bike: OpenML and URL failed, using synthetic fallback.")
    return _synthetic_log_normal(seed=44063, n_features=8)


def _load_cpu() -> ArrayPair:
    # OpenML 761 is not the UCI cpu_act regression task (target is categorical on the mirror).
    Xy = _try_openml(761, ["usr", "user"])
    if Xy is not None and Xy[0].shape[0] >= 100:
        return Xy
    print(
        "[datasets] cpu: OpenML 761 is not numeric usr regression here; using OpenML 573 (cpu_act)."
    )
    Xy = _try_openml(573, ["usr"])
    if Xy is not None:
        return Xy
    Xy = _try_openml(197, ["usr"])
    if Xy is not None:
        print("[datasets] cpu: OpenML 573 failed, using OpenML 197 fallback.")
        return Xy
    print("[datasets] cpu: OpenML failed, using synthetic fallback.")
    return _synthetic_log_normal(seed=761, n_features=8)


def _load_wine() -> ArrayPair:
    Xy = _try_openml(287, ["quality", "Quality"])
    if Xy is not None:
        return Xy
    print("[datasets] wine: OpenML 287 failed, using synthetic fallback.")
    return _synthetic_log_normal(seed=287, n_features=8)


def _load_energy() -> ArrayPair:
    # OpenML 44956 maps to unrelated tabular data on the public mirror (not UCI energy).
    Xy = _try_openml(1472, ["Y1", "y1"])
    if Xy is not None:
        return Xy
    Xy = _try_openml(213, ["Y1", "y1"])
    if Xy is not None:
        print("[datasets] energy: OpenML 1472 failed, using OpenML 213 fallback.")
        return Xy
    print("[datasets] energy: OpenML failed, using synthetic fallback.")
    return _synthetic_log_normal(seed=44956, n_features=8)


def _load_airfoil() -> ArrayPair:
    # Stable NASA airfoil noise: OpenML name (numeric id 1503 on mirror is a different huge set).
    Xy = _try_openml_named("airfoil_self_noise", 1, [])
    if Xy is not None:
        return Xy
    Xy = _try_openml(
        296,
        [
            "sound_pressure",
            "sound-pressure",
            "sound_pressure_level",
            "scaled_sound_pressure",
        ],
    )
    if Xy is not None and Xy[0].shape[0] <= 2500 and Xy[0].shape[1] <= 10:
        return Xy
    if Xy is not None:
        print("[datasets] airfoil: OpenML 296 is not NASA airfoil noise; trying numeric id 43919.")
    Xy = _try_openml(43919, [])
    if Xy is not None:
        return Xy
    print("[datasets] airfoil: OpenML failed, using synthetic fallback.")
    return _synthetic_log_normal(seed=296, n_features=8)


def _load_superconductor() -> ArrayPair:
    Xy = _try_openml(44964, ["critical_temp", "critical temperature"])
    if Xy is not None:
        return Xy
    Xy = _try_openml(43443, ["critical_temp"])
    if Xy is not None:
        print("[datasets] superconductor: OpenML 44964 failed, using OpenML 43443 fallback.")
        return Xy
    print("[datasets] superconductor: OpenML failed, using synthetic fallback.")
    return _synthetic_log_normal(seed=44964, n_features=8)


_LOADERS: Dict[str, Callable[[], ArrayPair]] = {
    "abalone": _load_abalone,
    "california": _load_california,
    "bike": _load_bike,
    "cpu": _load_cpu,
    "wine": _load_wine,
    "energy": _load_energy,
    "airfoil": _load_airfoil,
    "superconductor": _load_superconductor,
}


def load_dataset(name: str) -> ArrayPair:
    key = name.strip().lower()
    if key not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {list_datasets()}.")
    X, y = _LOADERS[key]()
    return X.astype(np.float64, copy=False), y.astype(np.float64, copy=False)


def get_splits(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    60/20/20 train/cal/test. StandardScaler on X (fit on train); y unchanged.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    n = X.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 samples after dropping NaN.")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    i1 = n_train
    i2 = i1 + n_cal
    tr_idx = perm[:i1]
    cal_idx = perm[i1:i2]
    te_idx = perm[i2:]

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_test, y_test = X[te_idx], y[te_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float64, copy=False)
    X_cal = scaler.transform(X_cal).astype(np.float64, copy=False)
    X_test = scaler.transform(X_test).astype(np.float64, copy=False)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
    }


def refresh_dataset_info() -> None:
    for name in _DATASET_ORDER:
        try:
            X, _y = load_dataset(name)
            DATASET_INFO[name]["n"] = int(X.shape[0])
            DATASET_INFO[name]["d"] = int(X.shape[1])
        except Exception:
            continue


def load_dataset_splits(
    name: str,
    seed: int,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    Legacy helper: load + 60/20/20 splits with X scaled, y unscaled (same as get_splits).
    Custom fractions are accepted only if they match 0.6/0.2/0.2; otherwise raises.
    """
    if abs(train_frac + cal_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + cal_frac + test_frac must sum to 1.")
    if (
        abs(train_frac - 0.6) > 1e-6
        or abs(cal_frac - 0.2) > 1e-6
        or abs(test_frac - 0.2) > 1e-6
    ):
        raise ValueError("Only 60/20/20 splits are supported; use load_dataset + get_splits to customize.")
    X, y = load_dataset(name)
    return get_splits(X, y, seed=seed)
