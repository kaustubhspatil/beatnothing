"""
Build the leaderboard from the submissions folder.

A submission is a folder under submissions/<name>/ holding
    meta.json     name, kind ("predictions" or "weights"), registered (date),
                  training_cutoff (date), description, and optional sha256 of the
                  frozen model file(s)
    signal.parquet  long format: Date, Ticker, value

Every submission is scored on every window in WINDOWS through the same engine
against two bars:

    the universe bar   always long every name in the contestant's universe, equal
                       weight, same days, same costs (the primary score, Net Edge)
    the investable bar RSP, the equal weight S&P 500 ETF, which holds every index
                       member by construction, dead ones included, and needs no
                       cost charge because it is a single buy and hold position

The gap between the two bars is the survivorship and selection inflation of the
contestant's universe, reported rather than hidden.

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
INVESTABLE_BAR = "RSP"


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

# Two tracks share one engine and one scoring rule and differ only in the universe.
TRACKS = {
    "survivor48": {"actual": "data/actual_returns.parquet", "submissions": "submissions", "out": "leaderboard",
                   "universe": "48 large cap names chosen in August 2026 (a survivor universe)"},
    "pit": {"actual": "data/pit/actual_returns_pit.parquet", "submissions": "submissions_pit", "out": "leaderboard/pit",
            "universe": "every S&P 500 member on each day, dead names included (point in time)"},
}


def load_actual(path: Path) -> pd.DataFrame:
    """Realised next day returns, wide (dates x tickers)."""
    df = pd.read_parquet(path)
    return df.pivot(index="Date", columns="Ticker", values="target").sort_index()


def load_signal(path: Path) -> pd.DataFrame:
    """A long format signal file (Date, Ticker, value) as a wide frame."""
    sig = pd.read_parquet(path)
    return sig.pivot(index="Date", columns="Ticker", values="value").sort_index()


def load_bars(path: Path | None) -> pd.DataFrame | None:
    """Investable bar returns (Date index, one column per ETF), or None if absent."""
    if path is None or not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    return df.set_index("Date").sort_index()


def load_submission(folder: Path) -> tuple[dict, pd.DataFrame]:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return meta, load_signal(folder / "signal.parquet")


def bar_returns(actual: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    """The universe bar on exactly these days: always long, same costs."""
    always = pd.DataFrame(1.0, index=actual.index, columns=actual.columns)
    return Backtest(actual, predictions=always, cost_bps=cost_bps).daily_returns


def score_window(actual: pd.DataFrame, wide: pd.DataFrame, kind: str, start, end,
                 bar: pd.Series, n_boot: int = 2000, cost_bps: float = COST_BPS,
                 investable: pd.Series | None = None, rule: str = "long_flat", quantile: float = 0.1) -> dict | None:
    a = actual.loc[(actual.index >= start) & (actual.index <= end)]
    w = wide.loc[(wide.index >= start) & (wide.index <= end)]
    if len(a) == 0 or len(w) == 0:
        return None
    kw = {"predictions": w, "rule": rule, "quantile": quantile} if kind == "predictions" else {"weights": w, "allow_short": True}
    bt = Backtest(a, cost_bps=cost_bps, **kw)
    stats = bt.stats()
    r = bt.daily_returns
    # a dollar neutral book competes with cash, not with a long only bar
    dollar_neutral = rule == "long_short" or (kind == "weights" and abs(stats["avg_exposure"]) < 0.1 and stats["avg_short_exposure"] > 0.05)
    stats["bar_used"] = "cash" if dollar_neutral else "universe"
    b = pd.Series(0.0, index=r.index) if dollar_neutral else bar.reindex(r.index).fillna(0.0)
    edge = net_edge(r.values, b.values, n_boot=n_boot)
    grid = cost_grid(a, grid=(0, 10, 20), **kw)
    stats.update({"net_edge": edge["net_edge"], "ci_low": edge["ci_low"], "ci_high": edge["ci_high"],
                  "p_positive": edge["p_positive"], "clears_bar": edge["clears_bar"],
                  "bar_sharpe": edge["bar_sharpe"], "sharpe_se": edge["strategy_sharpe_se"],
                  "psr_vs_zero": edge["psr_vs_zero"],
                  "gross_sharpe": grid[0], "net_sharpe_20bps": grid[20]})
    if investable is not None:
        inv = investable.reindex(r.index)
        ok = inv.notna().values
        if ok.sum() >= 20:
            e2 = net_edge(r.values[ok], inv.values[ok], n_boot=n_boot, seed=1)
            stats.update({"edge_vs_investable": e2["net_edge"], "ci_low_vs_investable": e2["ci_low"],
                          "ci_high_vs_investable": e2["ci_high"], "clears_investable": e2["clears_bar"],
                          "investable_sharpe": e2["bar_sharpe"], "investable_days": int(ok.sum())})
    return stats


def score_signal(signal_path: Path, actual_path: Path, kind: str = "predictions", start=None, end=None,
                 n_boot: int = 2000, cost_bps: float = COST_BPS, bars_path: Path | None = None) -> dict:
    """Score one signal file against one realised returns file. Used by the CLI."""
    actual = load_actual(actual_path)
    wide = load_signal(signal_path)
    start = pd.Timestamp(start) if start else actual.index.min()
    end = pd.Timestamp(end) if end else actual.index.max()
    a = actual.loc[(actual.index >= start) & (actual.index <= end)]
    bars = load_bars(bars_path)
    inv = bars[INVESTABLE_BAR] if bars is not None and INVESTABLE_BAR in bars.columns else None
    res = score_window(actual, wide, kind, start, end, bar_returns(a, cost_bps), n_boot=n_boot,
                       cost_bps=cost_bps, investable=inv)
    if res is None:
        raise ValueError("no overlapping days between the signal and the realised returns in that window")
    res["window"] = [str(start.date()), str(end.date())]
    return res


def build(actual_path: Path, n_boot: int = 2000, root: Path | None = None, submissions: Path | None = None,
          universe: str | None = None) -> dict:
    root = root or ROOT
    submissions = submissions or root / "submissions"
    actual = load_actual(actual_path)
    bars = load_bars(root / "data" / "benchmark_returns.parquet")
    inv_all = bars[INVESTABLE_BAR] if bars is not None and INVESTABLE_BAR in bars.columns else None
    subs = sorted(p for p in submissions.iterdir() if (p / "meta.json").exists())
    ubars = {}
    for name, (s, e) in WINDOWS.items():
        a = actual.loc[(actual.index >= s) & (actual.index <= e)]
        ubars[name] = bar_returns(a)
    board = {"windows": WINDOWS, "cost_bps": COST_BPS, "universe": universe,
             "bar": "always long, equal weight, same universe, same costs",
             "investable_bar": f"{INVESTABLE_BAR}, the equal weight S&P 500 ETF, buy and hold" if inv_all is not None else None,
             "entries": []}
    for folder in subs:
        meta, wide = load_submission(folder)
        entry = {"meta": meta, "windows": {}}
        for name, (s, e) in WINDOWS.items():
            res = score_window(actual, wide, meta["kind"], s, e, ubars[name], n_boot=n_boot, investable=inv_all,
                               rule=meta.get("rule", "long_flat"), quantile=float(meta.get("quantile", 0.1)))
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
    has_inv = board.get("investable_bar") is not None
    out = ["# Leaderboard", ""]
    if board.get("universe"):
        out += [f"Universe: {board['universe']}.", ""]
    out += [f"Universe bar: {board['bar']}. Costs: {board['cost_bps']:.0f} bps per unit of turnover. "
           "Net Edge = net Sharpe minus the universe bar's net Sharpe on the same days, with a 95% paired "
           "stationary bootstrap interval (2,000 resamples, mean block 10 days). A contestant "
           "clears a bar only if the whole interval is above zero."]
    if has_inv:
        out.append(f" Investable bar: {board['investable_bar']}, no cost charged because it is one position held "
                   "throughout; it holds every index member by construction, dead ones included. The gap between "
                   "the two bars is the survivorship and selection inflation of the contestant's universe.")
    out.append("")
    for wname, (s, e) in board["windows"].items():
        rows = [(en["meta"], en["windows"][wname]) for en in board["entries"] if wname in en["windows"]]
        rows.sort(key=lambda t: -t[1]["net_edge"])
        head = ("<tr><th>Contestant</th><th>Rule</th><th>Net Edge</th><th>95% CI</th><th>Clears bar</th>"
                + ("<th>Edge vs RSP</th><th>95% CI</th><th>Clears RSP</th>" if has_inv else "")
                + "<th>Net Sharpe</th><th>Gross Sharpe</th><th>Sharpe at 20 bps</th><th>Max DD</th>"
                  "<th>Turnover/yr</th><th>Avg exposure</th><th>P&amp;L on $1M</th></tr>")
        out += [f"## {wname.replace('_', ' ')}  ({s} to {e})", "", "<table>", head]
        for meta, r in rows:
            inv_cells = ""
            if has_inv:
                if "edge_vs_investable" in r:
                    inv_cells = (f"<td>{_fmt(r['edge_vs_investable'])}</td>"
                                 f"<td>[{_fmt(r['ci_low_vs_investable'])}, {_fmt(r['ci_high_vs_investable'])}]</td>"
                                 f"<td>{'yes' if r['clears_investable'] else 'no'}</td>")
                else:
                    inv_cells = "<td></td><td></td><td></td>"
            rule_cell = r.get("rule", "long_flat") + (" vs cash" if r.get("bar_used") == "cash" else "")
            out.append(
                f"<tr><td>{meta['name']}</td><td>{rule_cell}</td><td>{_fmt(r['net_edge'])}</td>"
                f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
                f"<td>{'yes' if r['clears_bar'] else 'no'}</td>{inv_cells}<td>{_fmt(r['net_sharpe'])}</td>"
                f"<td>{_fmt(r['gross_sharpe'])}</td><td>{_fmt(r['net_sharpe_20bps'])}</td>"
                f"<td>{_fmt(r['max_drawdown'], 'pct')}</td><td>{_fmt(r['annual_turnover'], 't')}</td>"
                f"<td>{_fmt(r['avg_exposure'], 'pct1')}</td><td>{_fmt(r['dollar_pnl'], 'usd')}</td></tr>")
        out += ["</table>", ""]
        if has_inv and rows:
            bar_row = next((r for m, r in rows if m["name"] == "always_long"), None)
            if bar_row and "investable_sharpe" in bar_row:
                out += [f"Universe bar net Sharpe {_fmt(bar_row['bar_sharpe'])} against RSP {_fmt(bar_row['investable_sharpe'])}: "
                        f"the universe bar's edge over the investable index is {_fmt(bar_row['edge_vs_investable'])} "
                        f"[{_fmt(bar_row['ci_low_vs_investable'])}, {_fmt(bar_row['ci_high_vs_investable'])}]. "
                        "That is how much choosing the universe with hindsight was worth on this window.", ""]
    out += ["## Contestants", ""]
    for en in board["entries"]:
        m = en["meta"]
        out.append(f"* **{m['name']}** ({m['kind']}, training cutoff {m.get('training_cutoff', 'n/a')}, "
                   f"registered {m.get('registered', 'n/a')}): {m.get('description', '')}")
    return "\n".join(out) + "\n"


def main(actual_path: str | Path | None = None, n_boot: int = 2000, root: str | Path | None = None,
         track: str = "survivor48"):
    root = Path(root) if root else ROOT
    cfg = TRACKS[track]
    actual_path = Path(actual_path) if actual_path else root / cfg["actual"]
    board = build(actual_path, n_boot=n_boot, root=root, submissions=root / cfg["submissions"], universe=cfg["universe"])
    board["track"] = track
    out_dir = root / cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    (out_dir / "LEADERBOARD.md").write_text(to_markdown(board), encoding="utf-8")
    return board


if __name__ == "__main__":
    import sys
    main(track=sys.argv[1] if len(sys.argv) > 1 else "survivor48")
