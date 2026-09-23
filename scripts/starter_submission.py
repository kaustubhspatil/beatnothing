"""
Make a valid contestant in one command, with no API key and no download.

The point of this script is that a stranger should be able to go from `git clone` to a
scored entry on the board in about five minutes, and should never have to guess what the
files are meant to look like. It builds a real, honest, deliberately simple signal out of
the returns panel that is already committed to this repository, writes the two files the
rules ask for, and tells you the two commands that come next.

    python scripts/starter_submission.py --name my_first_try
    beatnothing validate submissions_pit/my_first_try
    beatnothing leaderboard --track pit

Then open `scripts/starter_submission.py`, replace `signal_from_returns` with your own
idea, and run it again. Everything else on the board went through exactly this path.

The one rule the code below is careful about, and the one you must stay careful about: the
value on date t may only use information that existed at the close of t. The returns panel
is indexed by decision date and holds the *next* day's return, so a signal formed on t may
read rows strictly before t and never row t itself. The `.shift(1)` is what enforces that.
Delete it and the board will still score you; the information coefficient check in the
validator is what will notice, and a Sharpe ratio of four is not a discovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = {
    "reversal": "five day reversal: yesterday's losers, on the theory that a short run of "
                "bad days is an overreaction that comes back",
    "momentum": "twelve month momentum skipping the last month, the oldest published "
                "cross sectional factor there is",
    "lowvol": "low volatility: the inverse of the trailing sixty day standard deviation",
}


def signal_from_returns(wide: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Turn a date by ticker frame of realised returns into a prediction for each name.

    `wide` holds, at row t, the return earned between t and t+1. Everything here is
    computed on that frame and then shifted forward one row, so the number attached to
    date t was built only out of rows that had already happened by the close of t.
    """
    if kind == "reversal":
        raw = -wide.rolling(5, min_periods=5).sum()
    elif kind == "momentum":
        raw = wide.rolling(252, min_periods=200).sum() - wide.rolling(21, min_periods=21).sum()
    elif kind == "lowvol":
        raw = -wide.rolling(60, min_periods=40).std()
    else:                                                       # pragma: no cover
        raise ValueError(f"unknown signal {kind}")
    return raw.shift(1)


def cross_sectional_rank(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rank the names against each other within each day, centred on zero.

    This is not decoration. A model trained to predict the level of tomorrow's return is
    largely predicting the market's drift, and the drift is what the bar already earns for
    free. Ranking inside the day removes the common move by construction, so whatever is
    left is the contestant's own opinion about which name beats which.
    """
    return raw.rank(axis=1, pct=True).sub(0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="folder name for your entry, lower case with underscores")
    ap.add_argument("--signal", default="reversal", choices=sorted(SIGNALS))
    ap.add_argument("--rule", default="long_short", choices=["long_flat", "long_top", "long_short"],
                    help="how the engine turns your numbers into positions")
    ap.add_argument("--track", default="pit", choices=["pit", "survivor48"])
    ap.add_argument("--author", default="", help="your name or handle, recorded in meta.json")
    a = ap.parse_args()

    actual_path, out_dir = ((ROOT / "data/pit/actual_returns_pit.parquet", ROOT / "submissions_pit")
                            if a.track == "pit" else
                            (ROOT / "data/actual_returns.parquet", ROOT / "submissions"))
    panel = pd.read_parquet(actual_path)
    wide = panel.pivot(index="Date", columns="Ticker", values="target").sort_index()
    print(f"{len(wide):,} trading days, {wide.shape[1]:,} names, "
          f"{wide.index[0].date()} to {wide.index[-1].date()}")

    values = cross_sectional_rank(signal_from_returns(wide, a.signal))
    long = (values.stack().rename("value").reset_index()
            .rename(columns={"level_0": "Date", "level_1": "Ticker"}))
    long = long[np.isfinite(long["value"])].copy()
    long["value"] = long["value"].astype("float32")
    # drop days you can't rank, don't fill with a constant
    long = long[["Date", "Ticker", "value"]].sort_values(["Date", "Ticker"]).reset_index(drop=True)

    folder = out_dir / a.name
    folder.mkdir(parents=True, exist_ok=True)
    long.to_parquet(folder / "signal.parquet", index=False, compression="zstd")
    meta = {
        "name": a.name,
        "kind": "predictions",
        "rule": a.rule,
        "registered": str(pd.Timestamp.today().date()),
        "author": a.author or "anonymous",
        "universe": "point in time S&P 500 membership" if a.track == "pit" else "48 survivor names",
        "description": f"Starter contestant, {SIGNALS[a.signal]}, ranked within each day and "
                       f"traded under the {a.rule} rule. Built by scripts/starter_submission.py "
                       f"from the committed returns panel, with no data beyond this repository.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "signal_sha256": hashlib.sha256((folder / "signal.parquet").read_bytes()).hexdigest(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    days = long.Date.nunique()
    print(f"\nwrote {folder.relative_to(ROOT)}  ({len(long):,} rows, {days:,} days)")
    print("\nnext:")
    print(f"  beatnothing validate {folder.relative_to(ROOT).as_posix()}")
    print(f"  beatnothing leaderboard --track {a.track}")
    print("\nthen edit signal_from_returns in this script and run it again.")


if __name__ == "__main__":
    main()
