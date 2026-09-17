"""
Command line entry point.

    beatnothing score my_signal.parquet --actual data/actual_returns.parquet
    beatnothing score my_weights.parquet --actual ... --kind weights --start 2022-01-01
    beatnothing leaderboard [--root PATH] [--actual PATH] [--n-boot 2000]
    beatnothing members 2008-09-15
"""
from __future__ import annotations

import argparse
import json
import sys


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
    board = main(args.actual, n_boot=args.n_boot, root=args.root)
    print(f"scored {len(board['entries'])} submissions; leaderboard written")
    return 0


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
    lb.set_defaults(func=_leaderboard)

    m = sub.add_parser("members", help="who was in the S&P 500 on a date")
    m.add_argument("date")
    m.set_defaults(func=_members)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
