"""
Command line entry point.

    beatnothing score my_signal.parquet --actual data/actual_returns.parquet
    beatnothing score my_weights.parquet --actual ... --kind weights --start 2022-01-01
    beatnothing leaderboard [--track survivor48|pit] [--root PATH] [--n-boot 2000]
    beatnothing pbo v1.parquet v2.parquet v3.parquet --actual data/pit/actual_returns_pit.parquet
    beatnothing validate submissions_pit/my_model
    beatnothing members 2008-09-15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _score(args) -> int:
    from .leaderboard import score_signal
    res = score_signal(args.signal, args.actual, kind=args.kind, start=args.start, end=args.end,
                       n_boot=args.n_boot, cost_bps=args.cost_bps, bars_path=args.bars)
    keys = ["window", "days", "net_edge", "ci_low", "ci_high", "clears_bar", "net_sharpe", "bar_sharpe",
            "edge_vs_investable", "ci_low_vs_investable", "ci_high_vs_investable", "clears_investable",
            "investable_sharpe", "gross_sharpe", "net_sharpe_20bps", "sharpe_se", "psr_vs_zero",
            "max_drawdown", "annual_turnover", "avg_exposure", "dollar_pnl"]
    print(json.dumps({k: res[k] for k in keys if k in res}, indent=2, default=str))
    verdict = "clears the bar" if res["clears_bar"] else "does not clear the bar"
    print(f"\nNet Edge {res['net_edge']:+.2f} [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]: {verdict}.", file=sys.stderr)
    return 0


def _leaderboard(args) -> int:
    from .leaderboard import main
    board = main(args.actual, n_boot=args.n_boot, root=args.root, track=args.track,
                 seed=args.seed, output_scope=args.output_scope)
    print(f"scored {len(board['entries'])} submissions on the {args.track} track; leaderboard written")
    return 0


def _pbo(args) -> int:
    import numpy as np
    import pandas as pd
    from .stats import deflated_sharpe, pbo_cscv
    from .engine import Backtest, TRADING_DAYS
    from .leaderboard import load_actual, load_signal

    actual = load_actual(args.actual)
    curves, names = [], []
    for path in args.signals:
        wide = load_signal(path)
        bt = Backtest(actual, predictions=wide, rule=args.rule, quantile=args.quantile)
        curves.append(bt.daily_returns.reindex(actual.index).fillna(0.0).values)
        p = Path(path)
        names.append(p.parent.name if p.stem == "signal" else p.stem)   # every submission file is signal.parquet
    R = np.column_stack(curves)
    res = pbo_cscv(R, n_splits=args.splits)
    srs = R.mean(0) / R.std(0) * np.sqrt(TRADING_DAYS)
    best = int(np.argmax(srs))
    deflated = deflated_sharpe(R[:, best], n_trials=R.shape[1], sr_variance=float(np.var(srs, ddof=1)))
    print(json.dumps({"variants": names, "best": names[best], "best_sharpe": round(float(srs[best]), 3),
                      "probability_of_backtest_overfitting": round(res["pbo"], 3),
                      "median_out_of_sample_rank": res["median_rank"], "combinations": res["n_combinations"],
                      "deflated_sharpe": round(deflated["deflated_sharpe"], 3),
                      "sharpe_the_search_alone_would_give": round(deflated["expected_max_sharpe"], 3)}, indent=2))
    if res["pbo"] > 0.4:
        print("\nAt this overfitting probability the best variant is roughly what searching alone would "
              "produce. Submit the one you chose before looking, or none.", file=sys.stderr)
    return 0


def _validate(args) -> int:
    from .leaderboard import TRACKS, load_actual
    from .validate import validate_submission
    folder = Path(args.folder)
    track = next((t for t, c in TRACKS.items() if c["submissions"] in folder.parts), "survivor48")
    actual_path = Path(args.actual) if args.actual else Path(TRACKS[track]["actual"])
    actual = load_actual(actual_path) if actual_path.exists() else None
    rep = validate_submission(folder, actual)
    print(rep.render())
    return 0 if rep.ok else 1


def _members(args) -> int:
    from .universe import Membership
    members = sorted(Membership().on(args.date))
    print(f"{len(members)} members on {args.date}")
    print(" ".join(members))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="beatnothing",
                                 description="Can your model beat doing nothing, after costs, without knowing the future?")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score one signal file against realised returns")
    s.add_argument("signal", help="parquet with columns Date, Ticker, value")
    s.add_argument("--actual", required=True, help="parquet with columns Date, Ticker, target (next day return)")
    s.add_argument("--kind", choices=["predictions", "weights"], default="predictions")
    s.add_argument("--start", default=None)
    s.add_argument("--end", default=None)
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--cost-bps", type=float, default=10.0)
    s.add_argument("--bars", default=None, help="parquet with Date and RSP columns to also score against the investable bar")
    s.set_defaults(func=_score)

    lb = sub.add_parser("leaderboard", help="score every submission under a benchmark root")
    lb.add_argument("--root", default=None)
    lb.add_argument("--actual", default=None)
    lb.add_argument("--n-boot", type=int, default=2000)
    lb.add_argument("--track", choices=["survivor48", "pit"], default="survivor48")
    lb.add_argument("--seed", type=int, default=None)
    lb.add_argument("--output-scope", choices=["official", "research"], default="official")
    lb.set_defaults(func=_leaderboard)

    p = sub.add_parser("pbo", help="how much of your best variant was the search itself "
                                   "(meaningful from about eight variants upward)")
    p.add_argument("signals", nargs="+", help="one signal parquet per variant you tried")
    p.add_argument("--actual", required=True)
    p.add_argument("--rule", choices=["long_flat", "long_top", "long_short"], default="long_flat")
    p.add_argument("--quantile", type=float, default=0.1)
    p.add_argument("--splits", type=int, default=10)
    p.set_defaults(func=_pbo)

    v = sub.add_parser("validate", help="check a submission folder against the rules")
    v.add_argument("folder")
    v.add_argument("--actual", default=None)
    v.set_defaults(func=_validate)

    m = sub.add_parser("members", help="who was in the S&P 500 on a date")
    m.add_argument("date")
    m.set_defaults(func=_members)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
