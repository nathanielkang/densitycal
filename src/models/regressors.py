"""
Regressor wrappers with a common fit/predict interface.

If xgboost is not installed, XGBRegressor uses sklearn GradientBoostingRegressor.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor

try:
    from xgboost import XGBRegressor as XGBoostNative

    _HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    XGBoostNative = None
    _HAS_XGB = False

ArrayLike = Union[np.ndarray, List[List[float]]]


class MLPRegressor:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model = SklearnMLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            early_stopping=True,
            max_iter=500,
            random_state=seed,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._model.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._model.predict(X), dtype=np.float64)


class XGBRegressor:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        if _HAS_XGB:
            self._model = XGBoostNative(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                random_state=seed,
                verbosity=0,
            )
        else:
            self._model = GradientBoostingRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                random_state=seed,
            )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "XGBRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._model.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._model.predict(X), dtype=np.float64)


class RFRegressor:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model = RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RFRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._model.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._model.predict(X), dtype=np.float64)


def list_regressors() -> List[str]:
    return ["mlp", "xgboost", "rf"]


def get_regressor(name: str, seed: int = 42):
    key = name.strip().lower()
    if key == "mlp":
        return MLPRegressor(seed=seed)
    if key in ("xgboost", "xgb"):
        return XGBRegressor(seed=seed)
    if key in ("rf", "random_forest", "randomforest"):
        return RFRegressor(seed=seed)
    raise ValueError(f"Unknown regressor '{name}'. Expected one of {list_regressors()}.")


def fit_regressor(name: str, X: ArrayLike, y: ArrayLike, seed: int = 42):
    """Fit regressor by name; returns a model with predict()."""
    model = get_regressor(name, seed=seed)
    return model.fit(X, y)


def predict_regressor(model, X: ArrayLike) -> np.ndarray:
    """Predict with a model returned by fit_regressor/get_regressor."""
    return np.asarray(model.predict(X), dtype=np.float64).ravel()
