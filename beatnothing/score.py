"""
Scoring: Net Edge with honest uncertainty.

Net Edge = net Sharpe(strategy) minus net Sharpe(do nothing), both through the same
engine on the same days. Because both series live on the same calendar, the paired
difference is what carries the evidence, and a stationary block bootstrap of that
paired series gives a confidence interval that respects volatility clustering.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, skew, kurtosis

from .engine import TRADING_DAYS


def sharpe(daily_returns) -> float:
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(equity) -> float:
    e = np.asarray(equity, dtype=float)
    return float((e / np.maximum.accumulate(e) - 1.0).min())


def sharpe_standard_error(daily_returns) -> float:
    """Lo (2002) large sample standard error of an annualized Sharpe ratio (iid case)."""
    r = np.asarray(daily_returns, dtype=float)
    n = len(r)
    sr_daily = r.mean() / r.std() if r.std() > 0 else 0.0
    return float(np.sqrt((1 + 0.5 * sr_daily ** 2) / n) * np.sqrt(TRADING_DAYS))


def probabilistic_sharpe(daily_returns, benchmark_sharpe_annual: float = 0.0) -> float:
    """
    Bailey and Lopez de Prado (2012): probability that the true Sharpe exceeds a
    benchmark, accounting for sample length, skew and kurtosis. The benchmark is
    given annualized; the computation runs in daily units.
    """
    r = np.asarray(daily_returns, dtype=float)
    n = len(r)
    if n < 3 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    sr_b = benchmark_sharpe_annual / np.sqrt(TRADING_DAYS)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def _stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis and Romano stationary bootstrap: random starts, geometric block lengths."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        take = min(int(rng.geometric(p)), n - i)
        idx[i:i + take] = (start + np.arange(take)) % n
        i += take
    return idx


def bootstrap_sharpe_diff(strategy_returns, bar_returns, n_boot: int = 2000,
                          mean_block: float = 10.0, seed: int = 0, alpha: float = 0.05) -> dict:
    """
    Paired stationary block bootstrap of Sharpe(strategy) minus Sharpe(bar).
    Returns the point estimate, the central (1 - alpha) interval and the bootstrap
    probability that the difference is positive.
    """
    s = np.asarray(strategy_returns, dtype=float)
    b = np.asarray(bar_returns, dtype=float)
    if len(s) != len(b):
        raise ValueError("paired series must have the same length")
    rng = np.random.default_rng(seed)
    point = sharpe(s) - sharpe(b)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = _stationary_bootstrap_indices(len(s), mean_block, rng)
        diffs[k] = sharpe(s[idx]) - sharpe(b[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"net_edge": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_positive": float((diffs > 0).mean()), "n_boot": n_boot, "mean_block": mean_block}


def net_edge(strategy_returns, bar_returns, **kw) -> dict:
    """Net Edge with its bootstrap interval and the verdict: a contestant clears the
    bar only if the whole interval sits above zero."""
    out = bootstrap_sharpe_diff(strategy_returns, bar_returns, **kw)
    out["clears_bar"] = bool(out["ci_low"] > 0)
    out["strategy_sharpe"] = sharpe(strategy_returns)
    out["bar_sharpe"] = sharpe(bar_returns)
    out["strategy_sharpe_se"] = sharpe_standard_error(strategy_returns)
    out["psr_vs_zero"] = probabilistic_sharpe(strategy_returns, 0.0)
    return out


def cost_grid(actual, predictions=None, weights=None, grid=(0, 5, 10, 15, 20, 25)) -> dict:
    """Net Sharpe of one contestant across a grid of cost assumptions (bps)."""
    from .engine import Backtest
    return {int(c): Backtest(actual, predictions=predictions, weights=weights, cost_bps=c).stats()["net_sharpe"]
            for c in grid}
