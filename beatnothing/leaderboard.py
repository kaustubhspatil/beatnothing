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
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .engine import Backtest, COST_BPS
from .io import load_actual, load_bars, load_submission, load_signal
from .score import cost_grid, net_edge, sharpe
from .stats import joint_sharpe_tests

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


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    # Flat when silent: a day with no signal is a day in cash, not a day removed from the
    # record. Without this a contestant could submit only on the days it liked the look of
    # and have its Sharpe computed on that subset. It also puts every contestant on one
    # calendar, which the joint test across the board requires.
    w = w.reindex(index=a.index, columns=a.columns)
    if kind != "predictions":
        w = w.fillna(0.0)
    kw = {"predictions": w, "rule": rule, "quantile": quantile} if kind == "predictions" else {"weights": w, "allow_short": True}
    bt = Backtest(a, cost_bps=cost_bps, **kw)
    stats = bt.stats()
    r = bt.daily_returns
    # a dollar neutral book competes with cash, not with a long only bar
    dollar_neutral = rule == "long_short" or (kind == "weights" and abs(stats["avg_exposure"]) < 0.1 and stats["avg_short_exposure"] > 0.05)
    stats["bar_used"] = "cash" if dollar_neutral else "universe"
    b = pd.Series(0.0, index=r.index) if dollar_neutral else bar.reindex(r.index).fillna(0.0)
    grid = cost_grid(a, grid=(0, 10, 20), **kw)
    stats.update({"bar_sharpe": sharpe(b.values), "gross_sharpe": grid[0], "net_sharpe_20bps": grid[20]})
    if stats["avg_short_exposure"] > 0.01:
        # The bottom decile of almost any screen is where hard to borrow names live, so a
        # flat fifty basis points a year is the optimistic case. Price the pessimistic one.
        stats["net_sharpe_borrow_500bps"] = Backtest(a, cost_bps=cost_bps, borrow_bps_annual=500.0, **kw
                                                     ).stats()["net_sharpe"]
    # A dollar neutral book competes with cash, so measuring it against a long index says
    # nothing about it; the comparison is only meaningful for a contestant that is long.
    if investable is not None and not dollar_neutral:
        inv = investable.reindex(r.index)
        ok = inv.notna().values
        if ok.sum() >= 20:
            e2 = net_edge(r.values[ok], inv.values[ok], n_boot=n_boot, seed=1)
            stats.update({"edge_vs_investable": e2["net_edge"], "ci_low_vs_investable": e2["ci_low"],
                          "ci_high_vs_investable": e2["ci_high"], "clears_investable": e2["clears_bar"],
                          "investable_sharpe": e2["bar_sharpe"], "investable_days": int(ok.sum())})
    return stats, r, b


def score_signal(signal_path: Path, actual_path: Path, kind: str = "predictions", start=None, end=None,
                 n_boot: int = 2000, cost_bps: float = COST_BPS, bars_path: Path | None = None,
                 rule: str = "long_flat", quantile: float = 0.1) -> dict:
    """Score one signal file against one realised returns file. Used by the CLI."""
    actual = load_actual(actual_path)
    wide = load_signal(signal_path)
    start = pd.Timestamp(start) if start else actual.index.min()
    end = pd.Timestamp(end) if end else actual.index.max()
    a = actual.loc[(actual.index >= start) & (actual.index <= end)]
    bars = load_bars(bars_path)
    inv = bars[INVESTABLE_BAR] if bars is not None and INVESTABLE_BAR in bars.columns else None
    scored = score_window(actual, wide, kind, start, end, bar_returns(a, cost_bps), n_boot=n_boot,
                          cost_bps=cost_bps, investable=inv, rule=rule, quantile=quantile)
    if scored is None:
        raise ValueError("no overlapping days between the signal and the realised returns in that window")
    res, r, b = scored
    res.update(net_edge(r.values, b.values, n_boot=n_boot))
    res["window"] = [str(start.date()), str(end.date())]
    return res


def build(actual_path: Path, n_boot: int = 2000, root: Path | None = None, submissions: Path | None = None,
          universe: str | None = None, seed: int = 0, track: str | None = None) -> dict:
    root = root or ROOT
    submissions = submissions or root / "submissions"
    actual = load_actual(actual_path)
    bars_path = root / "data" / "benchmark_returns.parquet"
    bars = load_bars(bars_path)
    inv_all = bars[INVESTABLE_BAR] if bars is not None and INVESTABLE_BAR in bars.columns else None
    subs = sorted(p for p in submissions.iterdir() if (p / "meta.json").exists())
    ubars = {}
    for name, (s, e) in WINDOWS.items():
        a = actual.loc[(actual.index >= s) & (actual.index <= e)]
        ubars[name] = bar_returns(a)
    board = {"windows": WINDOWS, "cost_bps": COST_BPS, "universe": universe, "track": track,
             "bar": "always long, equal weight, same universe, same costs",
             "investable_bar": f"{INVESTABLE_BAR}, the equal weight S&P 500 ETF, buy and hold" if inv_all is not None else None,
             "entries": [], "seed": int(seed),
             "provenance": {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "actual_path": str(actual_path), "actual_sha256": _sha256(actual_path),
                            "benchmark_path": str(bars_path), "benchmark_sha256": _sha256(bars_path),
                            "submissions_path": str(submissions)}}
    series = {w: {} for w in WINDOWS}
    bar_series = {w: {} for w in WINDOWS}
    for folder in subs:
        meta, wide = load_submission(folder)
        entry = {"meta": meta, "windows": {}}
        for name, (s, e) in WINDOWS.items():
            scored = score_window(actual, wide, meta["kind"], s, e, ubars[name], n_boot=n_boot, investable=inv_all,
                                  rule=meta.get("rule", "long_flat"), quantile=float(meta.get("quantile", 0.1)))
            if scored is not None:
                res, r, b = scored
                entry["windows"][name] = res
                series[name][meta["name"]] = r.values
                bar_series[name][meta["name"]] = b.values
        board["entries"].append(entry)

    # One joint bootstrap per window: it gives every contestant its studentized interval
    # and p value, and the stepdown that controls the chance of even one false claim.
    by_name = {en["meta"]["name"]: en for en in board["entries"]}
    board["multiple_testing"] = {}
    for wname in WINDOWS:
        if not series[wname]:
            continue
        joint = joint_sharpe_tests(series[wname], bar_series[wname], n_boot=n_boot, seed=seed)
        board["multiple_testing"][wname] = {k: v for k, v in joint.items() if k != "contestants"}
        for cname, stats in joint["contestants"].items():
            by_name[cname]["windows"][wname].update(stats)
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
    mt = board.get("multiple_testing", {})
    block = next((v.get("block_size") for v in mt.values()), None)
    k = next((v.get("n_contestants") for v in mt.values()), len(board["entries"]))
    live = next((v.get("n_live_hypotheses") for v in mt.values()), None)
    q95 = next((v.get("bootstrap_max_t_q95") for v in mt.values()), None)
    out += [f"Universe bar: {board['bar']}. Costs: {board['cost_bps']:.0f} bps per unit of turnover. "
            "Net Edge = net Sharpe minus the bar's net Sharpe on the same days. The interval and the "
            "p value come from a studentized circular block bootstrap (Ledoit and Wolf), which recomputes "
            "a heteroskedasticity and autocorrelation robust standard error inside every resample"
            + (f"; block length {block} days, chosen by the Politis and White rule" if block else "")
            + ". **p adj** is the Romano and Wolf stepdown p value across all "
            + f"{k} contestants on this board: it is the one that decides. Testing {k} contestants "
            f"separately at five percent would produce a false winner {1 - 0.95 ** k:.0%} of the time, "
            "so a contestant clears its bar only when the adjusted p value is at or below five percent."]
    if live is not None:
        out.append(f" Joint diagnostics: {live} live hypotheses in the stepdown."
                   + (f" 95th percentile of the bootstrap max t-statistic: {q95:.2f}." if isinstance(q95, (int, float)) else ""))
    if has_inv:
        out.append(f" Investable bar: {board['investable_bar']}, no cost charged because it is one position held "
                   "throughout; it holds every index member by construction, dead ones included. The gap between "
                   "the two bars is the survivorship and selection inflation of the contestant's universe.")
    out.append("")
    for wname, (s, e) in board["windows"].items():
        rows = [(en["meta"], en["windows"][wname]) for en in board["entries"] if wname in en["windows"]]
        rows.sort(key=lambda t: -t[1]["net_edge"])
        head = ("<tr><th>Contestant</th><th>Rule</th><th>Net Edge</th><th>95% CI</th><th>p</th><th>p adj</th>"
                "<th>Clears bar</th>"
                + ("<th>Edge vs RSP</th><th>95% CI</th>" if has_inv else "")
                + "<th>Net Sharpe</th><th>Gross Sharpe</th><th>Sharpe at 20 bps</th>"
                  "<th>Sharpe at 500 bps borrow</th><th>Max DD</th>"
                  "<th>Turnover/yr</th><th>Avg exposure</th><th>P&amp;L on $1M</th></tr>")
        out += [f"## {wname.replace('_', ' ')}  ({s} to {e})", "", "<table>", head]
        for meta, r in rows:
            inv_cells = ""
            if has_inv:
                if "edge_vs_investable" in r:
                    inv_cells = (f"<td>{_fmt(r['edge_vs_investable'])}</td>"
                                 f"<td>[{_fmt(r['ci_low_vs_investable'])}, {_fmt(r['ci_high_vs_investable'])}]</td>")
                else:
                    inv_cells = "<td></td><td></td>"
            rule_cell = r.get("rule", "long_flat") + (" vs cash" if r.get("bar_used") == "cash" else "")
            verdict = "yes" if r.get("clears_bar_fwe") else "no"
            out.append(
                f"<tr><td>{meta['name']}</td><td>{rule_cell}</td><td>{_fmt(r['net_edge'])}</td>"
                f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
                f"<td>{r.get('p_value', float('nan')):.3f}</td><td>{r.get('p_value_fwe', float('nan')):.3f}</td>"
                f"<td>{verdict}</td>{inv_cells}<td>{_fmt(r['net_sharpe'])}</td>"
                f"<td>{_fmt(r['gross_sharpe'])}</td><td>{_fmt(r['net_sharpe_20bps'])}</td>"
                f"<td>{_fmt(r['net_sharpe_borrow_500bps']) if 'net_sharpe_borrow_500bps' in r else ''}</td>"
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
         track: str = "survivor48", seed: int | None = None, output_scope: str = "official"):
    root = Path(root) if root else ROOT
    cfg = TRACKS[track]
    actual_path = Path(actual_path) if actual_path else root / cfg["actual"]
    resolved_seed = int(seed if seed is not None else os.environ.get("BEATNOTHING_SEED", 0))
    board = build(actual_path, n_boot=n_boot, root=root, submissions=root / cfg["submissions"],
                  universe=cfg["universe"], seed=resolved_seed, track=track)
    if output_scope not in ("official", "research"):
        raise ValueError("output_scope must be 'official' or 'research'")
    out_dir = root / cfg["out"] if output_scope == "official" else root / "leaderboard" / "research" / track
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    (out_dir / "LEADERBOARD.md").write_text(to_markdown(board), encoding="utf-8")
    return board


if __name__ == "__main__":
    import sys
    main(track=sys.argv[1] if len(sys.argv) > 1 else "survivor48")
