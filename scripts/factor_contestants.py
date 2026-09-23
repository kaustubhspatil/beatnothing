"""
The classic cross sectional factors as reference contestants on the point in time track.

Price only, so that anyone can rebuild them, and monthly rebalanced, as the literature
trades them: the signal is measured on the first trading day of each month and held
until the next. Each contestant is one signal and one rule.

    momentum_12_1     return from t minus 252 to t minus 21 trading days
    reversal_1m       minus the trailing 21 day return
    low_vol_63d       minus the trailing 63 day realised volatility

Writes submissions_pit/<name>/{signal.parquet, meta.json}. The values are the daily
predictions the engine ranks; the rule and quantile live in meta.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submissions_pit"
EVAL_START, REG = "2022-01-01", "2026-09-18"


def monthly_hold(sig: pd.DataFrame) -> pd.DataFrame:
    """Keep the value observed on the first trading day of each month for the whole month."""
    first = sig.groupby([sig.index.year, sig.index.month]).head(1).index
    held = sig.copy()
    held.loc[~held.index.isin(first)] = np.nan
    return held.ffill()


def submit(name: str, wide: pd.DataFrame, members: pd.DataFrame, rule: str, quantile: float, description: str) -> None:
    long = wide.stack().rename("value").reset_index()
    long.columns = ["Date", "Ticker", "value"]
    long = long.merge(members, on=["Date", "Ticker"], how="inner")     # index members only
    long = long[long["Date"] >= EVAL_START]
    # no signal yet = leave the row out (flat when silent)
    long = long[np.isfinite(long["value"])]
    long["value"] = long["value"].astype("float32")
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    long.to_parquet(d / "signal.parquet", index=False, compression="zstd")
    meta = {"name": name, "kind": "predictions", "rule": rule, "quantile": quantile, "training_cutoff": "none (a formula)",
            "registered": REG, "universe": "point in time S&P 500 membership", "rebalance": "monthly, first trading day",
            "description": description,
            # provenance is the script
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_file": Path(__file__).name}
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{name:34s} {rule:10s} q={quantile:.2f} {len(long):>8,} rows {long['Date'].min().date()} -> {long['Date'].max().date()}")


def main() -> None:
    prices = pd.read_parquet(ROOT / "data" / "pit" / "prices_pit.parquet")
    close = prices.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    members = pd.read_parquet(ROOT / "data" / "pit" / "actual_returns_pit.parquet")[["Date", "Ticker"]]
    ret = close.pct_change()

    momentum = close.shift(21) / close.shift(252) - 1
    reversal = -(close / close.shift(21) - 1)
    low_vol = -ret.rolling(63).std() * np.sqrt(252)

    for sig, base, desc in [
        (momentum, "momentum_12_1", "Twelve month momentum skipping the most recent month (Jegadeesh and Titman). Monthly rebalanced."),
        (reversal, "reversal_1m", "One month short term reversal: minus the trailing 21 day return. Monthly rebalanced."),
        (low_vol, "low_vol_63d", "Low volatility: minus the trailing 63 day realised volatility. Monthly rebalanced."),
    ]:
        held = monthly_hold(sig)
        submit(f"{base}_long_short", held, members, "long_short", 0.1,
               desc + " Long the top decile, short the bottom decile, dollar neutral, 50 bps a year borrow.")
        submit(f"{base}_long_top", held, members, "long_top", 0.1,
               desc + " Long only the top decile, equal weight; judged against the universe bar.")


if __name__ == "__main__":
    main()
