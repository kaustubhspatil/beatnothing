"""
Rebuild the price snapshot from Yahoo Finance and refresh data/actual_returns.parquet.

Usage:
    python scripts/download_data.py                # full history 2005 to today
    python scripts/download_data.py --start 2021-06-01

Writes data/raw/sp500_stocks.csv, data/raw/vix.csv, data/raw/treasury_10y.csv, then
recomputes the trailing feature panel with beatnothing.features and stores the
realised next day returns for 2022 onwards in data/actual_returns.parquet together
with a MANIFEST.json of sha256 hashes, so a leaderboard can always be tied to the
exact snapshot that produced it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# The capstone universe: 48 large cap US stocks chosen in August 2026 from the then
# current S&P 500, two per sector bucket, plus SPY and QQQ as benchmarks only.
# This is a survivor universe; the README states what that costs.
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "CRM", "JPM", "BAC", "GS", "MS",
           "WFC", "BLK", "C", "AXP", "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "WMT", "PG", "KO", "PEP",
           "COST", "MCD", "NKE", "XOM", "CVX", "COP", "SLB", "CAT", "BA", "HON", "UPS", "GE", "DIS",
           "NFLX", "CMCSA", "VZ", "LIN", "APD", "NEE", "DUK", "AMT", "PLD", "SPY", "QQQ", "RSP"]
MARKET = {"vix": "^VIX", "treasury_10y": "^TNX"}
# Investable bars: next day returns of ETFs that hold the whole index, dead names included.
# RSP is the equal weight S&P 500 (since 2003), SPY the cap weighted one. Neither is a model input.
BARS = ["RSP", "SPY"]


def _flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download(start: str, end: str) -> None:
    import yfinance as yf
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in TICKERS:
        for attempt in range(3):
            try:
                df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
                if df.empty:
                    raise RuntimeError("empty")
                df = _flat(df)
                df["Ticker"] = t
                df.index.name = "Date"
                frames.append(df.reset_index()[["Date", "Close", "High", "Low", "Open", "Volume", "Ticker"]])
                print(f"ok {t} {len(df)} rows to {df.index[-1].date()}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"retry {t} ({attempt}): {exc}", flush=True)
                time.sleep(3)
    pd.concat(frames, ignore_index=True).to_csv(RAW / "sp500_stocks.csv", index=False)
    for name, tk in MARKET.items():
        df = _flat(yf.download(tk, start=start, end=end, progress=False, auto_adjust=True))
        df.index.name = "Date"
        df.reset_index().to_csv(RAW / f"{name}.csv", index=False)


def rebuild_actual() -> None:
    from beatnothing import features
    prices = pd.read_csv(RAW / "sp500_stocks.csv", parse_dates=["Date"]).sort_values(["Ticker", "Date"])
    vix = pd.read_csv(RAW / "vix.csv", parse_dates=["Date"], index_col="Date")["Close"].rename("vix")
    tsy = pd.read_csv(RAW / "treasury_10y.csv", parse_dates=["Date"], index_col="Date")["Close"].rename("treasury_10y")
    market = pd.concat([vix, tsy], axis=1).sort_index()
    panel = features.build_feature_panel(prices, market)
    panel.to_parquet(ROOT / "data" / "feature_panel.parquet", index=False)
    actual = panel[panel["Date"] >= "2022-01-01"][["Date", "Ticker", "target"]]
    actual.to_parquet(ROOT / "data" / "actual_returns.parquet", index=False)
    # investable bars, aligned like the target: the return earned from the close of t to t+1
    wide = prices.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    bars = pd.DataFrame({b: wide[b].pct_change().shift(-1) for b in BARS if b in wide.columns})
    bars = bars.loc["2005-01-01":].dropna(how="all").reset_index()
    bars.to_parquet(ROOT / "data" / "benchmark_returns.parquet", index=False)
    manifest = {"built": str(date.today()), "rows": int(len(actual)),
                "last_decision_date": str(actual["Date"].max().date()), "bars": list(bars.columns[1:]), "files": {}}
    for p in [RAW / "sp500_stocks.csv", RAW / "vix.csv", RAW / "treasury_10y.csv",
              ROOT / "data" / "feature_panel.parquet", ROOT / "data" / "actual_returns.parquet",
              ROOT / "data" / "benchmark_returns.parquet"]:
        manifest["files"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT / "data" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("actual_returns:", actual.shape, "to", manifest["last_decision_date"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    if not args.skip_download:
        download(args.start, args.end)
    rebuild_actual()
