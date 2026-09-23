"""
Export the board in a form a browser can re-cost for itself.

The engine computes a contestant's net return as

    net = gross - (turnover * cost + short_exposure * borrow)

so a page that holds those three daily series can recompute net returns, equity, Sharpe
and drawdown at *any* cost level, with no server and no Python. That turns the project's
central finding into something you can drag: move the cost dial from zero and watch the
leaderboard reorder and every contestant sink below the bar.

    python scripts/export_web.py --track pit --out docs/web/board.json

The output is fetchable straight from raw.githubusercontent, which sends a permissive CORS
header, so any static page can use it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from beatnothing.engine import BORROW_BPS_ANNUAL, COST_BPS, Backtest
from beatnothing.leaderboard import ROOT, TRACKS, WINDOWS, load_actual, load_submission

# 6 decimals is plenty for daily returns and halves the file
DP = {"gross": 10, "turn": 8, "short": 8}


def series(values, places: int) -> list[float]:
    return [round(float(v), places) for v in np.nan_to_num(np.asarray(values, dtype=float))]


def _pub(published: dict, name: str, window: str, key: str):
    """A field from the published leaderboard, or None when it was not scored there."""
    row = published.get(name, {}).get(window)
    if row is None or key not in row:
        return None
    v = row[key]
    return round(v, 6) if isinstance(v, float) else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="pit", choices=sorted(TRACKS))
    ap.add_argument("--out", default="docs/web/board.json")
    args = ap.parse_args()

    cfg = TRACKS[args.track]
    board_path = ROOT / cfg["out"] / "leaderboard.json"
    published = {e["meta"]["name"]: e["windows"]
                 for e in json.loads(board_path.read_text(encoding="utf-8"))["entries"]}
    actual = load_actual(ROOT / cfg["actual"])
    folders = sorted(p for p in (ROOT / cfg["submissions"]).iterdir() if p.is_dir())

    # same as leaderboard.score_window: slice first, then reindex
    windows = {}
    for wname, (start, end) in WINDOWS.items():
        a = actual.loc[(actual.index >= start) & (actual.index <= end)]
        rows = []
        for folder in folders:
            meta, signal = load_submission(folder)
            w = signal.loc[(signal.index >= start) & (signal.index <= end)]
            if len(a) == 0 or len(w) == 0:
                continue
            w = w.reindex(index=a.index, columns=a.columns)
            rule = meta.get("rule", "long_flat")
            # weight submissions are used as-is (can be short), match score_window
            kind = meta.get("kind", "predictions")
            if kind == "predictions":
                kw = {"predictions": w, "rule": rule, "quantile": float(meta.get("quantile", 0.1))}
            else:
                kw = {"weights": w.fillna(0.0), "allow_short": True}
            bt = Backtest(a, **kw)
            stats = bt.stats()
            dollar_neutral = rule == "long_short" or (
                kind == "weights" and abs(stats["avg_exposure"]) < 0.1
                and stats["avg_short_exposure"] > 0.05)
            gross = bt.daily_returns + bt.daily_costs
            short = bt.weights.clip(upper=0).abs().sum(axis=1)
            rows.append({
                "name": meta["name"],
                "rule": rule,
                "kind": kind,
                "description": meta.get("description", ""),
                "registered": meta.get("registered", ""),
                "is_bar": meta["name"] == "always_long",
                "bar_used": "cash" if dollar_neutral else "universe",
                "published_net_sharpe": round(stats["net_sharpe"], 6),
                # ship the verdict with the series so the demo doesn't just count point estimates
                "net_edge": _pub(published, meta["name"], wname, "net_edge"),
                "ci_low": _pub(published, meta["name"], wname, "ci_low"),
                "ci_high": _pub(published, meta["name"], wname, "ci_high"),
                "clears_bar": _pub(published, meta["name"], wname, "clears_bar"),
                "clears_bar_fwe": _pub(published, meta["name"], wname, "clears_bar_fwe"),
                "p_value_fwe": _pub(published, meta["name"], wname, "p_value_fwe"),
                "gross": series(gross, DP["gross"]),
                "turn": series(bt.turnover, DP["turn"]),
                "short": series(short, DP["short"]),
            })
        windows[wname] = {"dates": [str(d.date()) for d in a.index], "contestants": rows}
        print(f"{wname:20s} {len(a):>5} days, {len(rows):>3} contestants")

    payload = {
        "track": args.track,
        "universe": cfg["universe"],
        "defaults": {"cost_bps": COST_BPS, "borrow_bps_annual": BORROW_BPS_ANNUAL},
        "trading_days": 252,
        "sharpe": "mean(net) / std(net, ddof=0) * sqrt(252)   <- population sd, not sample sd",
        "note": ("net = gross - (turnover * cost_bps/10000 + short * borrow_bps_annual/10000/252). "
                 "Recompute at any cost level; the series themselves never change. A contestant "
                 "with bar_used 'cash' is dollar neutral and is judged against zero, not the bar. Use ddof=0 for the standard deviation or every number will be off by sqrt(n/(n-1))."),
        "windows": windows,
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    kb = out.stat().st_size / 1024
    rows_total = sum(len(w["contestants"]) for w in windows.values())
    print("")
    print(f"wrote {args.out}  {len(windows)} windows, {rows_total} contestant "
          f"windows, {kb:,.0f} KB")


if __name__ == "__main__":
    main()
