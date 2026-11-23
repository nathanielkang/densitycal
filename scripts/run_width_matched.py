"""Width-matched Split control.

For every dataset-seed-regressor triple of the main suite, refit the regressor,
recompute Split and DensityCal, and add ``split_wm``: the Split interval with its
single global half-width inflated so that its MPIW equals DensityCal's MPIW on
the same run. Split and DensityCal rows are written too so the refit can be
checked against ``results_v3``.
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

from src.conformal.methods import predict_intervals
from src.data.datasets import get_splits, list_datasets, load_dataset
from src.evaluation.metrics import evaluate_all
from src.models.regressors import fit_regressor, predict_regressor


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="puma32h,california,bike,cpu,grid_stability,energy,airfoil,superconductor")
    p.add_argument("--regressors", default="mlp,xgboost,rf")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--n0", type=int, default=30)
    p.add_argument("--output_dir", default="results_width_matched")
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
                yhat_te = predict_regressor(model, s["X_test"])
                ymin, ymax = float(np.min(yhat_tr)), float(np.max(yhat_tr))

                def ev(lo, hi):
                    return evaluate_all(s["y_test"], lo, hi, yhat_te, ymin, ymax, K=a.K, alpha=a.alpha)

                ints = {}
                for method in ["split", "densitycal"]:
                    lo, hi = predict_intervals(method, model, s["X_train"], s["y_train"],
                                               s["X_cal"], s["y_cal"], s["X_test"],
                                               alpha=a.alpha, K=a.K, n0=a.n0, seed=seed)
                    ints[method] = (np.asarray(lo, float), np.asarray(hi, float))
                    rows.append(_row(ds, seed, reg, method, ev(*ints[method]), a.K))

                lo_s, hi_s = ints["split"]
                lo_d, hi_d = ints["densitycal"]
                center = 0.5 * (lo_s + hi_s)
                half = 0.5 * float(np.mean(hi_d - lo_d))
                m = ev(center - half, center + half)
                rows.append(_row(ds, seed, reg, "split_wm", m, a.K))
                print(f"[ok] {ds} seed={seed} reg={reg} split_wm covgap={m['covgap']:.4f} "
                      f"mpiw={m['mpiw']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[saved] {ds} rows={len(rows)}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
