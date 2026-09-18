"""
Scoring: Net Edge with honest uncertainty.

Net Edge = net Sharpe(strategy) minus net Sharpe(the bar), both through the same engine
on the same days. Because both series live on the same calendar, the paired difference
is what carries the evidence.

Two ways to put an interval on it. The default is the studentized circular block
bootstrap of Ledoit and Wolf, which recomputes a heteroskedasticity and autocorrelation
robust standard error inside every resample; it lives in `beatnothing.stats` together
with the familywise correction that a leaderboard needs. The older percentile bootstrap
is kept for comparison. Both were measured rather than assumed: see
`leaderboard/stats_validation.json`, which reports how often each one claims an edge
when there is none, and how often it finds one that is really there.
"""
from __future__ import annotations

import numpy as np

from .engine import TRADING_DAYS
from .stats import (deflated_sharpe, expected_max_sharpe, joint_sharpe_tests, ledoit_wolf_test,
                    pbo_cscv, politis_white_block_size, probabilistic_sharpe, sharpe_diff_and_se)

__all__ = ["sharpe", "max_drawdown", "sharpe_standard_error", "probabilistic_sharpe", "net_edge",
           "bootstrap_sharpe_diff", "cost_grid", "joint_sharpe_tests", "ledoit_wolf_test",
           "sharpe_diff_and_se", "pbo_cscv", "deflated_sharpe", "expected_max_sharpe",
           "politis_white_block_size"]


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
    """Lo (2002) large sample standard error of an annualized Sharpe ratio, iid case."""
    r = np.asarray(daily_returns, dtype=float)
    n = len(r)
    if n < 2 or r.std() == 0:
        return 0.0
    sr_daily = r.mean() / r.std()
    return float(np.sqrt((1 + 0.5 * sr_daily ** 2) / n) * np.sqrt(TRADING_DAYS))


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
    Paired stationary block bootstrap of Sharpe(strategy) minus Sharpe(bar), percentile
    interval. Kept for comparison; `net_edge` uses the studentized test by default.
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


def net_edge(strategy_returns, bar_returns, method: str = "ledoit_wolf", n_boot: int = 2000,
             alpha: float = 0.05, seed: int = 0, **kw) -> dict:
    """
    Net Edge with an interval and the verdict. A contestant clears its bar only when the
    whole interval sits above zero.

    method="ledoit_wolf" (the default) studentizes a circular block bootstrap with a HAC
    standard error and also reports a one sided p value. method="percentile" is the older
    stationary bootstrap. Neither corrects for a board of many contestants; that is what
    `beatnothing.stats.joint_sharpe_tests` does, and what the leaderboard runs.
    """
    if method == "ledoit_wolf":
        out = ledoit_wolf_test(strategy_returns, bar_returns, n_boot=n_boot, alpha=alpha, seed=seed, **kw)
    elif method == "percentile":
        out = bootstrap_sharpe_diff(strategy_returns, bar_returns, n_boot=n_boot, alpha=alpha, seed=seed, **kw)
        out["clears_bar"] = bool(out["ci_low"] > 0)
    else:
        raise ValueError("method must be 'ledoit_wolf' or 'percentile'")
    out["method"] = method
    out["strategy_sharpe"] = sharpe(strategy_returns)
    out["bar_sharpe"] = sharpe(bar_returns)
    out["strategy_sharpe_se"] = sharpe_standard_error(strategy_returns)
    out["psr_vs_zero"] = probabilistic_sharpe(strategy_returns, 0.0)
    return out


def cost_grid(actual, predictions=None, weights=None, grid=(0, 5, 10, 15, 20, 25), **engine_kwargs) -> dict:
    """Net Sharpe of one contestant across a grid of cost assumptions (bps)."""
    from .engine import Backtest
    return {int(c): Backtest(actual, predictions=predictions, weights=weights, cost_bps=c,
                             **engine_kwargs).stats()["net_sharpe"] for c in grid}
