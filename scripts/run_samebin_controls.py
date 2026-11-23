"""Same-bin component controls for DensityCal.

Every variant uses the same grouping as DensityCal: the clipped predicted value
assigned to K equal-width bins on the training predicted range. Only the
half-width rule changes:

    mondrian_pv  local quantile in each bin (Mondrian CP on the DensityCal bins);
                 an empty bin uses the global quantile. No blend, no floor.
    no_blend     local quantile floored at the global quantile (w_k = 1).
    no_floor     occupancy blend of local and global quantiles, no floor.
    densitycal   blend + floor (Algorithm 1).

Split and DensityCal rows are refit so the run can be checked against
``results_v3``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal.methods import (  # noqa: E402
    _assign_predicted_value_bin,
    _finite_sample_quantile,
    predict_intervals,
)
from src.data.datasets import get_splits, list_datasets, load_dataset  # noqa: E402
from src.evaluation.metrics import evaluate_all  # noqa: E402
from src.models.regressors import fit_regressor, predict_regressor  # noqa: E402

VARIANTS = ("mondrian_pv", "no_blend", "no_floor", "densitycal_re")


def _row(ds, seed, reg, method, m, K):
    row = {"dataset": ds, "seed": seed, "regressor": reg, "method": method}
    sc = m.get("strat_coverage", {})
    for k in range(1, K + 1):
        row[f"strat_cov_{k}"] = sc.get(f"stratum_{k}", np.nan)
    for key, name in [("covgap", "covgap"), ("marginal_coverage", "marg_cov"),
                      ("mpiw", "mpiw"), ("avg_width_rare", "avg_width_rare"),
                      ("avg_width_common", "avg_width_common"), ("width_ratio", "width_ratio")]:
        row[name] = m.get(key, np.nan)
    return row


def samebin_halfwidths(r, s_cal, K, alpha, n0):
    g = _finite_sample_quantile(r, alpha)
    out = {v: {} for v in VARIANTS}
    for k in range(1, K + 1):
        m = s_cal == k
        n_k = int(m.sum())
        q_k = _finite_sample_quantile(r[m], alpha) if n_k >= 1 else g
        w_k = min(1.0, n_k / max(n0, 1))
        blend = w_k * q_k + (1.0 - w_k) * g
        out["mondrian_pv"][k] = q_k
        out["no_blend"][k] = max(g, q_k)
        out["no_floor"][k] = blend
        out["densitycal_re"][k] = max(g, blend)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="puma32h,california,bike,cpu,grid_stability,energy,airfoil,superconductor")
    p.add_argument("--regressors", default="mlp,xgboost,rf")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--n0", type=int, default=30)
    p.add_argument("--output_dir", default="results_samebin")
    a = p.parse_args()

    out = ROOT / a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "experiment_results.csv"
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    unknown = set(datasets) - set(list_datasets())
    if unknown:
        raise SystemExit(f"unknown datasets: {sorted(unknown)}")

    rows = []
    for ds in datasets:
        X_raw, y_raw = load_dataset(ds)
        for seed in [42 + i for i in range(a.seeds)]:
            s = get_splits(X_raw, y_raw, seed=seed)
            for reg in [r.strip() for r in a.regressors.split(",") if r.strip()]:
                model = fit_regressor(reg, s["X_train"], s["y_train"], seed=seed)
                yhat_tr = predict_regressor(model, s["X_train"])
                yhat_ca = np.asarray(predict_regressor(model, s["X_cal"]), float).ravel()
                yhat_te = np.asarray(predict_regressor(model, s["X_test"]), float).ravel()
                ymin, ymax = float(np.min(yhat_tr)), float(np.max(yhat_tr))

                def ev(lo, hi):
                    return evaluate_all(s["y_test"], lo, hi, yhat_te, ymin, ymax, K=a.K, alpha=a.alpha)

                for method in ("split", "densitycal"):
                    lo, hi = predict_intervals(method, model, s["X_train"], s["y_train"],
                                               s["X_cal"], s["y_cal"], s["X_test"],
                                               alpha=a.alpha, K=a.K, n0=a.n0, seed=seed)
                    rows.append(_row(ds, seed, reg, method, ev(np.asarray(lo, float), np.asarray(hi, float)), a.K))

                r = np.abs(np.asarray(s["y_cal"], float).ravel() - yhat_ca)
                s_cal = _assign_predicted_value_bin(yhat_ca, ymin, ymax, a.K)
                s_te = _assign_predicted_value_bin(yhat_te, ymin, ymax, a.K)
                hw = samebin_halfwidths(r, s_cal, a.K, a.alpha, a.n0)
                for v in VARIANTS:
                    half = np.array([hw[v][int(k)] for k in s_te])
                    m = ev(yhat_te - half, yhat_te + half)
                    rows.append(_row(ds, seed, reg, v, m, a.K))
                print(f"[ok] {ds} seed={seed} reg={reg}", flush=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[saved] {ds} rows={len(rows)}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
