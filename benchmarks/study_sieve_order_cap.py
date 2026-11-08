"""Does the SieveAR order cap ``(n - 2) // 3`` earn its place against a small-sample criterion?

``select_ar_order`` evaluates every candidate order on the same sample truncated to ``upper``
lags and caps ``upper`` at ``(n - 2) // 3`` so the largest candidate keeps at least two
effective observations per parameter. The alternative is no cap with the corrected AIC
(AICc), which charges a high order for the lost degrees of freedom itself.

For each series length the script draws replicate AR series, selects an order under four
rules and records, for the fitted model at the selected order:

- the share whose fitted AR polynomial is non-stationary (the failure the cap exists for);
- the mean selected order and the share of selections equal to the true order;
- the mean absolute error of the selected order's one-step in-sample innovation variance
  against the true innovation variance (a fit-quality proxy).

Rules: ``bic`` with the cap (the shipped default), ``aic`` with the cap, ``aicc`` with the cap,
and ``aicc`` with the largest order the sample can fit at all (``n_eff - k - 2 > 0``).

Run::

    python benchmarks/study_sieve_order_cap.py --json benchmarks/results/study/sieve_cap.json

Only numpy is used; the selection is re-derived here by least squares so the study does not
depend on the library's selector.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

TRUE_PHI = {"ar1": [0.6], "ar2": [0.5, -0.3], "ar1_near_unit": [0.9]}


def _simulate(phi: list[float], n: int, rng: np.random.Generator) -> np.ndarray:
    p = len(phi)
    burn = 200
    e = rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(p, n + burn):
        x[t] = sum(phi[j] * x[t - 1 - j] for j in range(p)) + e[t]
    return x[burn:]


def _fit(series: np.ndarray, upper: int, k: int) -> tuple[float, np.ndarray, int]:
    """OLS of order ``k`` on the sample truncated to ``upper`` lags: (rss, coefs, n_eff)."""
    n = series.shape[0]
    target = series[upper:]
    n_eff = target.shape[0]
    design = np.ones((n_eff, k + 1))
    for j in range(1, k + 1):
        design[:, j] = series[upper - j : n - j]
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    return float(resid @ resid), beta[1:], n_eff


def _criterion(name: str, rss: float, n_eff: int, k: int) -> float:
    params = k + 1
    base = n_eff * np.log(max(rss / n_eff, 1e-300))
    if name == "aic":
        return base + 2 * params
    if name == "bic":
        return base + np.log(n_eff) * params
    denom = n_eff - params - 1
    return base + 2 * params + 2 * params * (params + 1) / denom if denom > 0 else np.inf


def _stationary(coefs: np.ndarray) -> bool:
    k = coefs.shape[0]
    if k == 0:
        return True
    comp = np.zeros((k, k))
    comp[0] = coefs
    comp[1:, :-1] = np.eye(k - 1)
    return bool(np.max(np.abs(np.linalg.eigvals(comp))) < 1.0)


def _select(series: np.ndarray, rule: str) -> tuple[int, np.ndarray, float]:
    criterion, capped = rule.split("_")
    n = series.shape[0]
    upper = int(min(10 * np.log10(n), n - 1))
    if capped == "cap":
        upper = min(upper, (n - 2) // 3)
    else:  # the largest order whose AICc is finite: n_eff - (k + 1) - 1 > 0 at n_eff = n - k
        upper = max(1, (n - 3) // 2)
    best: tuple[float, int] = (np.inf, 1)
    for k in range(1, upper + 1):
        rss, _, n_eff = _fit(series, upper, k)
        value = _criterion(criterion, rss, n_eff, k)
        if value < best[0]:
            best = (value, k)
    rss, coefs, n_eff = _fit(series, upper, best[1])
    return best[1], coefs, rss / n_eff


RULES = ("bic_cap", "aic_cap", "aicc_cap", "aicc_nocap")


def _cell(model: str, n: int, reps: int, seed: int) -> dict[str, Any]:
    phi = TRUE_PHI[model]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    stats = {r: {"orders": [], "nonstat": 0, "hit": 0, "var_err": []} for r in RULES}
    for _ in range(reps):
        x = _simulate(phi, n, rng)
        x = x - x.mean()
        for rule in RULES:
            k, coefs, sigma2 = _select(x, rule)
            s = stats[rule]
            s["orders"].append(k)
            s["nonstat"] += int(not _stationary(coefs))
            s["hit"] += int(k == len(phi))
            s["var_err"].append(abs(sigma2 - 1.0))
    for rule, s in stats.items():
        out[rule] = {
            "mean_order": float(np.mean(s["orders"])),
            "nonstationary_share": s["nonstat"] / reps,
            "true_order_share": s["hit"] / reps,
            "mean_abs_variance_error": float(np.mean(s["var_err"])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    result: dict[str, Any] = {"reps": args.reps, "cells": {}}
    for model in TRUE_PHI:
        for n in (20, 30, 40, 60, 100, 200):
            cell = _cell(model, n, args.reps, seed=1000 + n)
            result["cells"][f"{model}_n{n}"] = cell
            row = " ".join(
                f"{r}:ord={cell[r]['mean_order']:.2f},ns={cell[r]['nonstationary_share']:.3f}"
                for r in RULES
            )
            print(f"{model} n={n} {row}", flush=True)
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=1)
    print("WROTE", args.json)


if __name__ == "__main__":
    main()
