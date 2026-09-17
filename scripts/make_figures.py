"""
Figures for the leaderboard page. Run after beatnothing.leaderboard.main().

    net_edge.png          the signature chart: Net Edge with 95% intervals, both windows
    cost_inversion.png    gross versus net Sharpe on the sealed window, one bar pair each
    forward_2026.png      equity curves of every contestant against the bar, 2026 to date
    survivorship_gap.png  the point in time universe a free price source can no longer supply
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beatnothing import Backtest, Membership, coverage_report
from beatnothing.leaderboard import LEADERBOARD, ROOT, SUBMISSIONS, WINDOWS, load_actual, load_submission

FIG = LEADERBOARD / "figures"
BLUE, ORANGE, AQUA, YELLOW, MUTED, INK, INK2, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#898781", "#0b0b0b", "#52514e", "#e1e0d9")
GOOD, BAD = "#0ca30c", "#d03b3b"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 13,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})

LABELS = {
    "always_long": "Always long (the bar)", "linear_incumbent": "Linear (incumbent)",
    "ffnn_128_64_32": "Feedforward NN", "lstm_60d": "LSTM 60d", "cnn1d_60d": "1D CNN 60d",
    "lightgbm_mse": "LightGBM (MSE)", "kronos_small_zero_shot": "Kronos small, zero shot",
    "costaware_net_lambda0bps_3seed_mean": "Cost aware net, no cost term",
    "costaware_net_lambda10bps_3seed_mean": "Cost aware net, 10 bps term",
}


def label(name):
    return LABELS.get(name, name)


def fig_net_edge(board):
    windows = list(board["windows"].items())
    first = windows[0][0]
    # one fixed row order (by the first window) so both panels share labels honestly
    order = sorted([en["meta"]["name"] for en in board["entries"]
                    if first in en["windows"] and en["meta"]["name"] != "always_long"],
                   key=lambda n: next(en["windows"][first]["net_edge"] for en in board["entries"] if en["meta"]["name"] == n))
    by_name = {en["meta"]["name"]: en["windows"] for en in board["entries"]}
    has_inv = board.get("investable_bar") is not None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for ax, (wname, (s, e)) in zip(axes, windows):
        for i, name in enumerate(order):
            r = by_name[name].get(wname)
            if r is None:
                continue
            c = GOOD if r["clears_bar"] else (BAD if r["ci_high"] < 0 else MUTED)
            ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(r["net_edge"], i, "o", color=c, ms=7, zorder=3)
            if has_inv and "edge_vs_investable" in r:      # hollow marker: the same contestant against RSP
                y = i - 0.28
                ax.plot([r["ci_low_vs_investable"], r["ci_high_vs_investable"]], [y, y], color=ORANGE, lw=1.2, alpha=0.9)
                ax.plot(r["edge_vs_investable"], y, "o", mfc="white", mec=ORANGE, mew=1.4, ms=6, zorder=3)
        ax.axvline(0, color=INK, lw=1.0)
        ax.set_title(f"{wname.replace('_', ' ')}: {s} to {e}", pad=8)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([label(n) for n in order])
    note = "Filled: against the universe bar (always long the same 48 names). Green would mean the whole interval clears zero; red, entirely below; grey straddles zero. Nothing is green."
    if has_inv:
        note += "\nHollow orange: the same contestant against RSP, the investable equal weight index that still holds the names that died."
    fig.text(0.01, 0.995, note, fontsize=9.2, color=INK2, va="top")
    fig.supxlabel("Net Edge = net Sharpe minus the bar's net Sharpe on the same days, 95% paired stationary bootstrap", fontsize=10, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.12)
    fig.savefig(FIG / "net_edge.png")
    plt.close(fig)


def fig_bars(actual, bars):
    """The two bars side by side: the survivor universe against the investable index."""
    if bars is None or "RSP" not in bars.columns:
        return
    s, e = "2022-01-01", "2026-12-31"
    a = actual.loc[(actual.index >= s) & (actual.index <= e)]
    uni = Backtest(a, predictions=pd.DataFrame(1.0, index=a.index, columns=a.columns)).daily_returns
    rsp = bars["RSP"].reindex(uni.index).fillna(0.0)
    spy = bars["SPY"].reindex(uni.index).fillna(0.0) if "SPY" in bars.columns else None
    fig, ax = plt.subplots(figsize=(11, 5))
    for series, name, color, ls in [(uni, "48 survivor names, equal weight, the universe bar", INK, "--"),
                                     (rsp, "RSP: every index member, dead ones included", ORANGE, "-"),
                                     (spy, "SPY: cap weighted index", MUTED, ":")]:
        if series is None:
            continue
        eq = (1 + series).cumprod()
        ax.plot(eq.index, eq.values, color=color, lw=2.0 if name.startswith("48") else 1.6, ls=ls)
        ax.annotate(f"{name}  (Sharpe {series.mean() / series.std() * np.sqrt(252):+.2f})", xy=(eq.index[-1], eq.values[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.8, va="center", color=color)
    ax.axvline(pd.Timestamp("2026-01-01"), color=GRID, lw=1.2)
    ax.text(pd.Timestamp("2026-01-05"), ax.get_ylim()[1] * 0.98, "2026, post cutoff", fontsize=8.5, color=INK2, va="top")
    ax.set_xlim(uni.index[0], uni.index[-1] + pd.Timedelta(days=430))
    ax.set_ylabel("growth of $1")
    ax.set_title("Two bars: the universe chosen with hindsight against the index that could not choose")
    fig.savefig(FIG / "two_bars.png")
    plt.close(fig)


def fig_cost_inversion(board):
    wname = "sealed_2022_2025"
    rows = [(label(en["meta"]["name"]), en["windows"][wname]) for en in board["entries"] if wname in en["windows"]]
    rows.sort(key=lambda t: -t[1]["gross_sharpe"])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["gross_sharpe"] for _, r in rows], 0.4, color=MUTED, label="gross (0 bps)")
    ax.bar(x + 0.2, [r["net_sharpe"] for _, r in rows], 0.4, color=BLUE, label="net (10 bps)")
    bar = next(r["net_sharpe"] for nm, r in rows if nm.startswith("Always long"))
    ax.axhline(bar, color=ORANGE, lw=1.2, ls="--")
    ax.text(len(rows) - 0.5, bar + 0.03, "do nothing", color=ORANGE, ha="right", fontsize=9, fontweight="bold")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([nm for nm, _ in rows], rotation=25, ha="right")
    ax.set_ylabel("annualized Sharpe, sealed 2022 to 2025")
    ax.set_title("Costs reorder the field: gross rank is not net rank")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIG / "cost_inversion.png")
    plt.close(fig)


def fig_forward(board, actual):
    s, e = WINDOWS["post_cutoff_2026"]
    a = actual.loc[(actual.index >= s) & (actual.index <= e)]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    curves = {}
    for folder in sorted(SUBMISSIONS.iterdir()):
        if not (folder / "meta.json").exists():
            continue
        meta, wide = load_submission(folder)
        w = wide.loc[(wide.index >= s) & (wide.index <= e)]
        kw = {"predictions": w} if meta["kind"] == "predictions" else {"weights": w}
        curves[meta["name"]] = Backtest(a, **kw).equity / 1e6
    order = sorted(curves, key=lambda k: -curves[k].iloc[-1])
    palette = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#4a3aa7", "#e34948", "#008300", MUTED]
    # spread end labels so converging curves stay readable (minimum vertical gap in data units)
    ends = np.array([curves[n].iloc[-1] for n in order])
    span = max(np.nanmax([c.max() for c in curves.values()]) - np.nanmin([c.min() for c in curves.values()]), 1e-6)
    gap = 0.045 * span
    pos = ends.copy()
    for i in range(1, len(pos)):               # order is descending, so push each label below the previous one
        pos[i] = min(pos[i], pos[i - 1] - gap)
    for i, name in enumerate(order):
        c = curves[name]
        is_bar = name == "always_long"
        color = INK if is_bar else palette[i % len(palette)]
        ax.plot(c.index, c.values, color=color, lw=2.4 if is_bar else 1.3, ls="--" if is_bar else "-", zorder=5 if is_bar else 2)
        ax.annotate(label(name), xy=(c.index[-1], c.values[-1]), xytext=(c.index[-1] + pd.Timedelta(days=4), pos[i]),
                    textcoords="data", fontsize=8.5, va="center", color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.6) if abs(pos[i] - ends[i]) > 1e-9 else None)
    ax.set_title("2026 to date, frozen models, net of 10 bps: the field hugs the bar or falls below it")
    ax.set_ylabel("growth of $1 (net)")
    ax.set_xlim(c.index[0], c.index[-1] + pd.Timedelta(days=60))
    fig.savefig(FIG / "forward_2026.png")
    plt.close(fig)


def fig_survivorship(available):
    mem = Membership()
    rep = coverage_report(mem, available, "2021-12-31", "2026-09-17")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    n_start, n_left, n_gone = rep["members_at_start"], rep["left_during_window"], rep["left_and_unavailable"]
    ax.barh(["Members on 2021 12 31", "Left the index 2022 to 2026", "Left and no longer downloadable"],
            [n_start, n_left, n_gone], color=[MUTED, ORANGE, BAD])
    for i, v in enumerate([n_start, n_left, n_gone]):
        ax.text(v + 3, i, f"{v}", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.invert_yaxis()
    ax.set_title("What a survivor universe silently drops from the sealed test window")
    names = rep["left_and_unavailable_tickers"]
    ax.text(0, -0.28, "No longer on Yahoo Finance: " + ", ".join(names), transform=ax.transAxes,
            fontsize=8, color=INK2, wrap=True)
    ax.set_xlim(0, n_start * 1.15)
    fig.savefig(FIG / "survivorship_gap.png")
    plt.close(fig)
    (LEADERBOARD / "coverage_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    board = json.loads((LEADERBOARD / "leaderboard.json").read_text(encoding="utf-8"))
    actual = load_actual(ROOT / "data" / "actual_returns.parquet")
    from beatnothing.leaderboard import load_bars
    bars = load_bars(ROOT / "data" / "benchmark_returns.parquet")
    fig_net_edge(board)
    fig_cost_inversion(board)
    fig_forward(board, actual)
    fig_bars(actual, bars)
    from beatnothing.universe import PIT_DIR
    gone = fig_survivorship(available=set(actual.columns) | {
        # names that left the index but still trade under the same ticker (checked 2026 09 17)
        t for t in json.loads((PIT_DIR / "still_listed_leavers.json").read_text())})
    print("figures written; survivorship gap:", gone["left_and_unavailable"], "of", gone["members_at_start"])
