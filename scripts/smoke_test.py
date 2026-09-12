"""Quick local validation before VM deployment — all 7 methods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split

from src.conformal.methods import predict_intervals
from src.evaluation.metrics import evaluate_all
from src.models.regressors import fit_regressor, predict_regressor


ALL_METHODS = [
    "split",
    "cqr",
    "locally_adaptive",
    "mondrian",
    "credo",
    "densitycal",
    "calibrated_mixup",
]


def _ascii(s: str) -> str:
    return s.encode("ascii", errors="replace").decode("ascii")


def main() -> int:
    rng = np.random.default_rng(0)
    n, p = 200, 5
    X = rng.standard_normal((n, p))
    coef = rng.standard_normal(p)
    y = np.exp(X @ coef * 0.15 + 0.05 * rng.standard_normal(n))
    y = np.log(y + 1e-6)

    X_trv, X_test, y_trv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trv, y_trv, test_size=0.25, random_state=0
    )

    alpha = 0.1
    K = 5

    ok = True
    msgs: list[str] = []

    for reg_name in ("mlp", "xgboost", "rf"):
        model = fit_regressor(reg_name, X_train, y_train, seed=42)
        yhat_tr = predict_regressor(model, X_train)
        yhat_te = predict_regressor(model, X_test)
        yhat_min_tr = float(np.min(yhat_tr))
        yhat_max_tr = float(np.max(yhat_tr))

        for method in ALL_METHODS:
            try:
                lo, hi = predict_intervals(
                    method, model,
                    X_train, y_train, X_cal, y_cal, X_test,
                    alpha=alpha, K=K, n0=30, seed=42,
                )
                m = evaluate_all(
                    y_test, lo, hi, yhat_te,
                    yhat_min_tr, yhat_max_tr,
                    K=K, alpha=alpha,
                )
                mc = float(m["marginal_coverage"])
                cg = float(m["covgap"])
                mpiw = float(m["mpiw"])
                print(
                    _ascii(
                        f"  {reg_name:8s} {method:20s} "
                        f"marg_cov={mc:.4f} covgap={cg:.4f} mpiw={mpiw:.4f}"
                    )
                )
                if mc < 0.5:
                    ok = False
                    msgs.append(f"{reg_name}/{method}: marginal_coverage {mc:.4f} < 0.5")
                if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
                    ok = False
                    msgs.append(f"{reg_name}/{method}: NaN in intervals")
                if lo.shape != hi.shape or lo.shape[0] != X_test.shape[0]:
                    ok = False
                    msgs.append(f"{reg_name}/{method}: shape mismatch")
            except Exception as e:
                ok = False
                msgs.append(f"{reg_name}/{method}: EXCEPTION {e}")
                import traceback
                traceback.print_exc()

    if ok:
        print(_ascii("PASS smoke_test (all 7 methods x 3 regressors)"))
        return 0
    print(_ascii("FAIL smoke_test"))
    for line in msgs:
        print(_ascii(line))
    return 1


if __name__ == "__main__":
    sys.exit(main())
