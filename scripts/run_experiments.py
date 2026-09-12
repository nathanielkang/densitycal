"""Main experiment runner for DensityCal / conformal calibration benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal.methods import predict_intervals
from src.data.datasets import get_splits, list_datasets, load_dataset
from src.evaluation.metrics import evaluate_all
from src.models.regressors import fit_regressor, predict_regressor


ALL_DATASETS = list_datasets()
ALL_REGRESSORS = ["mlp", "xgboost", "rf"]
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


class TeeStream:
    """Tee stdout to a UTF-8 log file (prints should stay ASCII for cp949 consoles)."""

    def __init__(self, stream: TextIO, log_path: Path) -> None:
        self._stream = stream
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "a", encoding="utf-8")

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def close_log(self) -> None:
        self._log.close()


def _parse_csv_or_all(raw: str, universe: list[str]) -> list[str]:
    t = raw.strip().lower()
    if t == "all":
        return list(universe)
    out = [x.strip() for x in raw.split(",") if x.strip()]
    for x in out:
        if x not in universe and t != "all":
            pass
    return out


def _method_aliases(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    if n == "split":
        return "split"
    return n


def _build_row(
    dataset: str,
    seed: int,
    regressor: str,
    method: str,
    metrics: dict[str, Any],
    K: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "regressor": regressor,
        "method": method,
    }
    sc = metrics.get("strat_coverage", {})
    for k in range(1, K + 1):
        key = f"stratum_{k}"
        row[f"strat_cov_{k}"] = sc.get(key, np.nan)
    row["covgap"] = metrics.get("covgap", np.nan)
    row["marg_cov"] = metrics.get("marginal_coverage", np.nan)
    row["mpiw"] = metrics.get("mpiw", np.nan)
    row["avg_width_rare"] = metrics.get("avg_width_rare", np.nan)
    row["avg_width_common"] = metrics.get("avg_width_common", np.nan)
    row["width_ratio"] = metrics.get("width_ratio", np.nan)
    return row


def _write_status(path: Path, completed: list[str], current: str | None, total: int) -> None:
    pct = 100.0 * len(completed) / max(total, 1)
    payload = {"completed": completed, "current": current, "pct": round(pct, 4)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    cfg_path = ROOT / "configs" / "default.yaml"
    defaults: dict[str, Any] = {}
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}

    p = argparse.ArgumentParser(description="DensityCal conformal experiments")
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(defaults.get("datasets", ALL_DATASETS)),
        help='Comma-separated dataset names or "all"',
    )
    p.add_argument(
        "--regressors",
        type=str,
        default=",".join(defaults.get("regressors", ALL_REGRESSORS)),
        help='Comma-separated regressors or "all"',
    )
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(defaults.get("methods", ALL_METHODS)),
        help='Comma-separated conformal methods or "all"',
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=int(defaults.get("seeds", 5)),
        help="Number of random seeds (42, 43, ...)",
    )
    p.add_argument("--alpha", type=float, default=float(defaults.get("alpha", 0.1)))
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--K", type=int, default=int(defaults.get("K", 5)))
    p.add_argument("--n0", type=int, default=int(defaults.get("n0", 30)))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    tee = TeeStream(sys.stdout, log_path)
    old_out = sys.stdout
    sys.stdout = tee  # type: ignore[assignment]

    try:
        print(
            _ascii(
                f"[start] {datetime.now(timezone.utc).isoformat()} "
                f"output_dir={out_dir}"
            )
        )
        datasets = _parse_csv_or_all(args.datasets, ALL_DATASETS)
        regressors = _parse_csv_or_all(args.regressors, ALL_REGRESSORS)
        methods = [_method_aliases(m) for m in _parse_csv_or_all(args.methods, ALL_METHODS)]

        seeds = [42 + i for i in range(max(1, int(args.seeds)))]
        csv_path = out_dir / "experiment_results.csv"
        status_path = out_dir / "status.json"
        completed: list[str] = []

        for dsi, ds in enumerate(datasets):
            _write_status(status_path, completed, ds, len(datasets))
            print(_ascii(f"[dataset] ({dsi+1}/{len(datasets)}) {ds}"))

            rows: list[dict[str, Any]] = []
            for seed in seeds:
                X_raw, y_raw = load_dataset(ds)
                splits = get_splits(X_raw, y_raw, seed=seed)
                X_tr, y_tr = splits["X_train"], splits["y_train"]
                X_ca, y_ca = splits["X_cal"], splits["y_cal"]
                X_te, y_te = splits["X_test"], splits["y_test"]

                for reg_name in regressors:
                    try:
                        model = fit_regressor(reg_name, X_tr, y_tr, seed=seed)
                    except Exception as e:
                        print(_ascii(f"[error] fit {reg_name} ds={ds} seed={seed}: {e}"))
                        traceback.print_exc()
                        continue

                    yhat_tr = predict_regressor(model, X_tr)
                    yhat_te = predict_regressor(model, X_te)
                    yhat_min_tr = float(np.min(yhat_tr))
                    yhat_max_tr = float(np.max(yhat_tr))

                    for method in methods:
                        try:
                            lo, hi = predict_intervals(
                                method,
                                model,
                                X_tr,
                                y_tr,
                                X_ca,
                                y_ca,
                                X_te,
                                alpha=float(args.alpha),
                                K=int(args.K),
                                n0=int(args.n0),
                                seed=int(seed),
                            )
                            m = evaluate_all(
                                y_te,
                                lo,
                                hi,
                                yhat_te,
                                yhat_min_tr,
                                yhat_max_tr,
                                K=int(args.K),
                                alpha=float(args.alpha),
                            )
                            rows.append(
                                _build_row(ds, seed, reg_name, method, m, int(args.K))
                            )
                            print(
                                _ascii(
                                    f"[ok] ds={ds} seed={seed} reg={reg_name} "
                                    f"method={method} marg_cov={m['marginal_coverage']:.4f} "
                                    f"covgap={m['covgap']:.4f}"
                                )
                            )
                        except Exception as e:
                            print(
                                _ascii(
                                    f"[error] method={method} ds={ds} seed={seed} "
                                    f"reg={reg_name}: {e}"
                                )
                            )
                            traceback.print_exc()

            if rows:
                df_new = pd.DataFrame(rows)
                if csv_path.is_file():
                    df_old = pd.read_csv(csv_path)
                    pd.concat([df_old, df_new], ignore_index=True).to_csv(
                        csv_path, index=False
                    )
                else:
                    df_new.to_csv(csv_path, index=False)
                print(_ascii(f"[saved] {csv_path} rows_added={len(rows)}"))

            completed.append(ds)
            _write_status(status_path, completed, None, len(datasets))

        print(_ascii(f"[done] completed={len(completed)} datasets"))
    finally:
        sys.stdout = old_out
        tee.close_log()


if __name__ == "__main__":
    main()
