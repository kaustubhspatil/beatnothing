"""
Verify the verifier: does the leaderboard's own statistics machinery behave?

A benchmark that ships canaries for leakage should ship them for its statistics too.
This script simulates return series with the features that break naive tests, fat tails
and volatility clustering, and a strategy that is highly correlated with its bar because
real contestants are, and then measures four things.

1. **Size.** When the strategy and the bar have the same true Sharpe, how often does each
   method claim an edge at five percent? It should be five percent.
2. **Power.** When the strategy really is better by a third of a Sharpe, how often is that
   found?
3. **Familywise error.** With fifteen worthless contestants on one board, how often does
   at least one get called a winner, with and without the stepdown?
4. **Overfitting.** What does the combinatorially symmetric cross validation say about a
   submitter who tried twelve variants of noise, and about one whose best variant is real?

Writes leaderboard/stats_validation.json and leaderboard/figures/stats_validation.png.

    python scripts/validate_stats.py [--sims 300] [--boot 200] [--fwe-sims 100]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from beatnothing.score import bootstrap_sharpe_diff
from beatnothing.stats import joint_sharpe_tests, ledoit_wolf_test, pbo_cscv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "leaderboard"
TRADING_DAYS = 252


def _volatility_path(T: int, rng: np.random.Generator, vol: float, persistence: float) -> np.ndarray:
    h = np.empty(T)
    h[0] = vol
    for t in range(1, T):                                   # persistent vol process
        h[t] = np.sqrt(persistence * h[t - 1] ** 2 + (1 - persistence) * vol ** 2 * rng.gamma(2.0, 0.5))
    return h


def _innovations(T: int, rng: np.random.Generator, nu: float) -> np.ndarray:
    return rng.standard_t(nu, T) / np.sqrt(nu / (nu - 2))    # unit variance, fat tails


def simulate_board(T: int, n_contestants: int, rng: np.random.Generator, edge: float = 0.0, rho: float = 0.9,
                   mu: float = 0.0004, vol: float = 0.011, nu: float = 5.0, persistence: float = 0.94):
    """
    One bar and `n_contestants` strategies, all with fat tails and clustered volatility.

    Every strategy shares the bar's shocks with correlation rho, which is what makes a
    leaderboard hard: contestants are nearly the bar. The construction gives each strategy
    the same mean and the same unconditional variance as the bar, so when `edge` is zero
    every contestant has exactly the bar's true Sharpe and any win is a false one. Getting
    this wrong is easy and silent: adding zero mean noise to the bar leaves the mean alone
    but raises the variance, which makes every contestant genuinely worse than the bar and
    a familywise experiment that can never produce a false winner.
    """
    h = _volatility_path(T, rng, vol, persistence)
    z_bar = _innovations(T, rng, nu)
    bar = mu + h * z_bar
    strategies = []
    for _ in range(n_contestants):
        z = _innovations(T, rng, nu)
        strategies.append(mu + edge + h * (rho * z_bar + np.sqrt(1 - rho ** 2) * z))
    return strategies, bar


def simulate(T: int, rng: np.random.Generator, **kw):
    """One strategy and its bar."""
    strategies, bar = simulate_board(T, 1, rng, **kw)
    return strategies[0], bar


def rejection_rates(sims: int, boot: int, T: int, edge: float, seed: int) -> dict:
    """How often each method claims an edge, at five percent, one sided."""
    rng = np.random.default_rng(seed)
    lw = pc = 0
    t0 = time.time()
    for i in range(sims):
        s, b = simulate(T, rng, edge=edge)
        lw += ledoit_wolf_test(s, b, n_boot=boot, seed=i)["p_value"] <= 0.05
        pc += bootstrap_sharpe_diff(s, b, n_boot=boot, seed=i)["ci_low"] > 0
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{sims} edge={edge:.5f} lw={lw / (i + 1):.3f} pc={pc / (i + 1):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return {"sims": sims, "edge_per_day": edge, "ledoit_wolf": lw / sims, "percentile": pc / sims}


def size_by_length(sims: int, boot: int, lengths, seed: int) -> list:
    """Size as the record lengthens: a finite sample effect shrinks, a bug does not."""
    rows = []
    for T in lengths:
        rng = np.random.default_rng(seed)
        lw = pc = 0
        for i in range(sims):
            s, b = simulate(T, rng)
            lw += ledoit_wolf_test(s, b, n_boot=boot, seed=i)["p_value"] <= 0.05
            pc += bootstrap_sharpe_diff(s, b, n_boot=boot, seed=i)["ci_low"] > 0
        se = np.sqrt(0.05 * 0.95 / sims)
        rows.append({"days": T, "years": round(T / TRADING_DAYS, 1), "ledoit_wolf": lw / sims,
                     "percentile": pc / sims, "monte_carlo_se": round(float(se), 4)})
        print(f"  T={T:<5} lw={lw / sims:.3f} pct={pc / sims:.3f} (+/- {1.96 * se:.3f})", flush=True)
    return rows


def familywise(sims: int, boot: int, T: int, n_contestants: int, seed: int) -> dict:
    """With a board of worthless contestants, how often is at least one called a winner?"""
    rng = np.random.default_rng(seed)
    any_raw = any_adj = 0
    t0 = time.time()
    for i in range(sims):
        board, bar = simulate_board(T, n_contestants, rng)
        strategies = {f"c{k}": s for k, s in enumerate(board)}
        bars = {f"c{k}": bar for k in range(n_contestants)}
        res = joint_sharpe_tests(strategies, bars, n_boot=boot, seed=i)["contestants"]
        any_raw += any(r["p_value"] <= 0.05 for r in res.values())
        any_adj += any(r["clears_bar_fwe"] for r in res.values())
        if (i + 1) % 20 == 0:
            print(f"  fwe {i + 1}/{sims} raw={any_raw / (i + 1):.3f} adj={any_adj / (i + 1):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return {"sims": sims, "n_contestants": n_contestants, "uncorrected": any_raw / sims,
            "stepdown": any_adj / sims, "bonferroni_expectation": 1 - 0.95 ** n_contestants}


def overfitting(T: int, n_variants: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    noise = np.column_stack(simulate_board(T, n_variants, rng)[0])
    real = noise.copy()
    real[:, n_variants // 3] += 0.0009                      # one variant is better
    return {"noise": pbo_cscv(noise, n_splits=10)["pbo"], "with_a_real_edge": pbo_cscv(real, n_splits=10)["pbo"],
            "n_variants": n_variants, "n_observations": T}


def figure(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, BLUE, ORANGE, GOOD, BAD, GRID = "#0b0b0b", "#898781", "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b", "#e1e0d9"
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 12,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "savefig.dpi": 150,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    ax = axes[0]
    size, power = report["size"], report["power"]
    x = np.arange(2)
    ax.bar(x - 0.2, [size["percentile"], power["percentile"]], 0.4, color=MUTED, label="percentile bootstrap")
    ax.bar(x + 0.2, [size["ledoit_wolf"], power["ledoit_wolf"]], 0.4, color=BLUE, label="studentized, HAC")
    for xi, vals in zip(x, [(size["percentile"], size["ledoit_wolf"]), (power["percentile"], power["ledoit_wolf"])]):
        for dx, v in zip((-0.2, 0.2), vals):
            ax.text(xi + dx, v + 0.012, f"{v:.3f}".lstrip("0"), ha="center", fontsize=8.5, color=INK)
    ax.axhline(0.05, color=BAD, ls="--", lw=1.2)
    ax.text(-0.45, 0.062, "five percent", color=BAD, fontsize=8.5, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(["no real edge\n(should be 0.05)", "a real edge\n(higher is better)"])
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("share of simulations claiming an edge")
    ax.set_title("Does the test keep its word?")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.92))

    ax = axes[1]
    fw = report["familywise"]
    bars = [fw["uncorrected"], fw["stepdown"]]
    ax.bar([0, 1], bars, 0.55, color=[BAD, GOOD])
    for i, v in enumerate(bars):
        ax.text(i, v + 0.015, f"{v:.0%}", ha="center", fontweight="bold", color=INK)
    ax.axhline(0.05, color=MUTED, ls="--", lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["tested separately", "after the stepdown"])
    ax.set_ylim(0, max(bars) * 1.3 + 0.05)
    ax.set_ylabel("share of boards with a false winner")
    ax.set_title(f"{fw['n_contestants']} worthless contestants")

    ax = axes[2]
    ov = report["overfitting"]
    vals = [ov["noise"], ov["with_a_real_edge"]]
    ax.bar([0, 1], vals, 0.55, color=[BAD, GOOD])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.0%}", ha="center", fontweight="bold", color=INK)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1.2)
    ax.text(1.42, 0.53, "one half, where noise belongs", color=MUTED, fontsize=8.5, ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{ov['n_variants']} variants\nof noise", f"{ov['n_variants']} variants,\none of them real"])
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("probability of backtest overfitting")
    ax.set_title("A submitter who tried many things")

    fig.suptitle("The statistics, measured on simulated markets with fat tails and clustered volatility",
                 x=0.006, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=3.0)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "stats_validation.png")
    print("figure written")


def main(sims: int, boot: int, fwe_sims: int, T: int) -> None:
    t0 = time.time()
    print("size, no real edge")
    size = rejection_rates(sims, boot, T, edge=0.0, seed=101)
    print("power, a real edge of about a third of a Sharpe")
    daily_edge = 0.33 * 0.011 / np.sqrt(TRADING_DAYS)      # 1/3 Sharpe as daily mean
    power = rejection_rates(sims, boot, T, edge=daily_edge, seed=202)
    print("size as the record lengthens")
    by_len = size_by_length(max(sims // 2, 80), boot, (500, 1000, 2000, 4000), seed=505)
    print("familywise error across a board")
    fw = familywise(fwe_sims, boot, T, n_contestants=15, seed=303)
    print("overfitting")
    ov = overfitting(T=2000, n_variants=12, seed=404)
    report = {"built": time.strftime("%Y-%m-%d"), "observations_per_simulation": T, "n_boot": boot,
              "size": size, "power": power, "size_by_length": by_len, "familywise": fw, "overfitting": ov,
              "minutes": round((time.time() - t0) / 60, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats_validation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "overfitting"}, indent=1))
    figure(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--fwe-sims", type=int, default=100)
    ap.add_argument("--days", type=int, default=1000)
    a = ap.parse_args()
    main(a.sims, a.boot, a.fwe_sims, a.days)
