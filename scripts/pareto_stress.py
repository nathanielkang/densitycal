"""Synthetic heteroscedastic + Pareto-tail DGP for Appendix stress test.

Generator for the construction in the manuscript:
    X ~ Uniform(0, 1)
    base mean 2X
    Gaussian noise scale (1 + 3X)
    optional Pareto right-tail mixture on Y with index beta.

Usage (from 2_Code/):
    python scripts/pareto_stress.py --n 5000 --beta 2.0 --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np


def sample_dgp(n: int, beta: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 1.0, size=n) * (1.0 + 3.0 * x)
    y = 2.0 * x + noise
    if np.isfinite(beta) and beta > 0:
        # Mix a Pareto tail onto the upper quantile of the design.
        mask = rng.random(n) < 0.15
        y = y.copy()
        y[mask] += rng.pareto(beta, size=int(np.sum(mask))) * (1.0 + 3.0 * x[mask])
    return x.reshape(-1, 1), y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    x, y = sample_dgp(args.n, args.beta, args.seed)
    print(f"n={args.n} beta={args.beta} seed={args.seed}")
    print(f"X[{x.min():.3f},{x.max():.3f}] Y mean={y.mean():.3f} p90={np.quantile(y, 0.9):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
