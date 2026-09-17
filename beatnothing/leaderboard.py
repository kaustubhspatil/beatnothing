"""
Build the leaderboard from the submissions folder.

A submission is a folder under submissions/<name>/ holding
    meta.json     name, kind ("predictions" or "weights"), registered (date),
                  training_cutoff (date), description, and optional sha256 of the
                  frozen model file(s)
    signal.parquet  long format: Date, Ticker, value

Every submission is scored on every window in WINDOWS through the same engine
against the same do nothing bar (the always long contestant on the same days).

The benchmark root is, in order: the BEATNOTHING_ROOT environment variable, the
repository this file lives in when it has a submissions folder, else the current
working directory. An installed copy of the package therefore scores whatever
folder you run it from.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import Backtest, COST_BPS
from .score import net_edge, cost_grid

WINDOWS = {
    "sealed_2022_2025": ("2022-01-01", "2025-12-31"),
    "post_cutoff_2026": ("2026-01-01", "2026-12-31"),
}


def find_root() -> Path:
    env = os.environ.get("BEATNOTHING_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parents[1]
    if (here / "submissions").exists():
        return here
    return Path.cwd()


ROOT = find_root()
SUBMISSIONS = ROOT / "submissions"
LEADERBOARD = ROOT / "leaderboard"


def load_actual(path: Path) -> pd.DataFrame:
    """Realised next day returns, wide (dates x tickers)."""
    df = pd.read_parquet(path)
    return df.pivot(index="Date", columns="Ticker", values="target").sort_index()


def load_signal(path: Path) -> pd.DataFrame:
    """A long format signal file (Date, Ticker, value) as a wide frame."""
    sig = pd.read_parquet(path)
    return sig.pivot(index="Date", columns="Ticker", values="value").sort_index()


def load_submission(folder: Path) -> tuple[dict, pd.DataFrame]:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return meta, load_signal(folder / "signal.parquet")


def bar_returns(actual: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    """The do nothing bar on exactly these days: always long, same costs."""
    always = pd.DataFrame(1.0, index=actual.index, columns=actual.columns)
    return Backtest(actual, predictions=always, cost_bps=cost_bps).daily_returns


def score_window(actual: pd.DataFrame, wide: pd.DataFrame, kind: str, start, end,
                 bar: pd.Series, n_boot: int = 2000, cost_bps: float = COST_BPS) -> dict | None:
    a = actual.loc[(actual.index >= start) & (actual.index <= end)]
    w = wide.loc[(wide.index >= start) & (wide.index <= end)]
    if len(a) == 0 or len(w) == 0:
        return None
    kw = {"predictions": w} if kind == "predictions" else {"weights": w}
    bt = Backtest(a, cost_bps=cost_bps, **kw)
    stats = bt.stats()
    r = bt.daily_returns
    b = bar.reindex(r.index).fillna(0.0)
    edge = net_edge(r.values, b.values, n_boot=n_boot)
    grid = cost_grid(a, grid=(0, 10, 20), **kw)
    stats.update({"net_edge": edge["net_edge"], "ci_low": edge["ci_low"], "ci_high": edge["ci_high"],
                  "p_positive": edge["p_positive"], "clears_bar": edge["clears_bar"],
                  "bar_sharpe": edge["bar_sharpe"], "sharpe_se": edge["strategy_sharpe_se"],
                  "psr_vs_zero": edge["psr_vs_zero"],
                  "gross_sharpe": grid[0], "net_sharpe_20bps": grid[20]})
    return stats


def score_signal(signal_path: Path, actual_path: Path, kind: str = "predictions", start=None, end=None,
                 n_boot: int = 2000, cost_bps: float = COST_BPS) -> dict:
    """Score one signal file against one realised returns file. Used by the CLI."""
    actual = load_actual(actual_path)
    wide = load_signal(signal_path)
    start = pd.Timestamp(start) if start else actual.index.min()
    end = pd.Timestamp(end) if end else actual.index.max()
    a = actual.loc[(actual.index >= start) & (actual.index <= end)]
    res = score_window(actual, wide, kind, start, end, bar_returns(a, cost_bps), n_boot=n_boot, cost_bps=cost_bps)
    if res is None:
        raise ValueError("no overlapping days between the signal and the realised returns in that window")
    res["window"] = [str(start.date()), str(end.date())]
    return res


def build(actual_path: Path, n_boot: int = 2000, root: Path | None = None) -> dict:
    root = root or ROOT
    submissions = root / "submissions"
    actual = load_actual(actual_path)
    subs = sorted(p for p in submissions.iterdir() if (p / "meta.json").exists())
    bars = {}
    for name, (s, e) in WINDOWS.items():
        a = actual.loc[(actual.index >= s) & (actual.index <= e)]
        bars[name] = bar_returns(a)
    board = {"windows": WINDOWS, "cost_bps": COST_BPS, "bar": "always long, equal weight, same universe, same costs",
             "entries": []}
    for folder in subs:
        meta, wide = load_submission(folder)
        entry = {"meta": meta, "windows": {}}
        for name, (s, e) in WINDOWS.items():
            res = score_window(actual, wide, meta["kind"], s, e, bars[name], n_boot=n_boot)
            if res is not None:
                entry["windows"][name] = res
        board["entries"].append(entry)
    return board


def _fmt(x, kind="f2"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return {"f2": f"{x:+.2f}", "pct": f"{x:+.1%}", "pct1": f"{x:.1%}", "int": f"{x:,.0f}", "t": f"{x:.1f}x",
            "usd": f"{x:+,.0f}"}[kind]


def to_markdown(board: dict) -> str:
    """Leaderboard as HTML tables (no markdown table syntax), one per window."""
    out = ["# Leaderboard", "",
           f"Bar: {board['bar']}. Costs: {board['cost_bps']:.0f} bps per unit of turnover. "
           "Net Edge = net Sharpe minus the bar's net Sharpe on the same days, with a 95% paired "
           "stationary bootstrap interval (2,000 resamples, mean block 10 days). A contestant "
           "clears the bar only if the whole interval is above zero.", ""]
    for wname, (s, e) in board["windows"].items():
        rows = [(en["meta"], en["windows"][wname]) for en in board["entries"] if wname in en["windows"]]
        rows.sort(key=lambda t: -t[1]["net_edge"])
        out += [f"## {wname.replace('_', ' ')}  ({s} to {e})", "", "<table>",
                "<tr><th>Contestant</th><th>Net Edge</th><th>95% CI</th><th>Clears bar</th><th>Net Sharpe</th>"
                "<th>Gross Sharpe</th><th>Sharpe at 20 bps</th><th>Max DD</th><th>Turnover/yr</th>"
                "<th>Avg exposure</th><th>P&amp;L on $1M</th></tr>"]
        for meta, r in rows:
            out.append(
                f"<tr><td>{meta['name']}</td><td>{_fmt(r['net_edge'])}</td>"
                f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
                f"<td>{'yes' if r['clears_bar'] else 'no'}</td><td>{_fmt(r['net_sharpe'])}</td>"
                f"<td>{_fmt(r['gross_sharpe'])}</td><td>{_fmt(r['net_sharpe_20bps'])}</td>"
                f"<td>{_fmt(r['max_drawdown'], 'pct')}</td><td>{_fmt(r['annual_turnover'], 't')}</td>"
                f"<td>{_fmt(r['avg_exposure'], 'pct1')}</td><td>{_fmt(r['dollar_pnl'], 'usd')}</td></tr>")
        out += ["</table>", ""]
    out += ["## Contestants", ""]
    for en in board["entries"]:
        m = en["meta"]
        out.append(f"* **{m['name']}** ({m['kind']}, training cutoff {m.get('training_cutoff', 'n/a')}, "
                   f"registered {m.get('registered', 'n/a')}): {m.get('description', '')}")
    return "\n".join(out) + "\n"


def main(actual_path: str | Path | None = None, n_boot: int = 2000, root: str | Path | None = None):
    root = Path(root) if root else ROOT
    actual_path = Path(actual_path) if actual_path else root / "data" / "actual_returns.parquet"
    board = build(actual_path, n_boot=n_boot, root=root)
    out_dir = root / "leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    (out_dir / "LEADERBOARD.md").write_text(to_markdown(board), encoding="utf-8")
    return board


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
