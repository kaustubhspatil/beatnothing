"""
Shared I/O helpers for benchmark artifacts.

The same parquet formats are used by CLI, scripts and leaderboard builds:
actual returns, submission signals and investable bar series.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_actual(path: Path) -> pd.DataFrame:
    """Realised next-day returns, wide (dates x tickers)."""
    df = pd.read_parquet(path)
    return df.pivot(index="Date", columns="Ticker", values="target").sort_index()


def load_signal(path: Path) -> pd.DataFrame:
    """Signal file in long format (Date, Ticker, value) as wide (dates x tickers)."""
    sig = pd.read_parquet(path)
    return sig.pivot(index="Date", columns="Ticker", values="value").sort_index()


def load_bars(path: Path | None) -> pd.DataFrame | None:
    """Investable bar returns indexed by Date, or None when unavailable."""
    if path is None or not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    return df.set_index("Date").sort_index()


def load_submission(folder: Path) -> tuple[dict, pd.DataFrame]:
    """Submission metadata and wide signal frame."""
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return meta, load_signal(folder / "signal.parquet")
