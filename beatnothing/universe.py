"""
Point in time universe membership and the survivorship coverage report.

The most common way a backtest knows the future is by choosing its universe with
hindsight: "the current S&P 500 constituents" is a list of companies that survived.
This module answers, for any date, who was actually in the index on that date, and
measures how much of that universe a free price source can still supply.

The membership tables ship inside the package (see beatnothing/data/SOURCE.md).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PIT_DIR = Path(__file__).resolve().parent / "data"
MEMBERSHIP_FILE = PIT_DIR / "sp500_membership_by_date.csv.gz"
START_END_FILE = PIT_DIR / "sp500_ticker_start_end.csv.gz"


class Membership:
    """Point in time constituent lookup built from a by date membership table."""

    def __init__(self, path: Path | None = None):
        path = path or MEMBERSHIP_FILE
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        self._dates = df["date"].to_numpy()
        self._sets = [frozenset(t.strip() for t in row.split(",") if t.strip()) for row in df["tickers"]]

    @property
    def first_date(self) -> pd.Timestamp:
        return pd.Timestamp(self._dates[0])

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self._dates[-1])

    def on(self, date) -> frozenset:
        """Constituents in force on `date` (the latest change on or before it)."""
        i = self._dates.searchsorted(pd.Timestamp(date).to_datetime64(), side="right") - 1
        if i < 0:
            raise ValueError(f"no membership data on or before {date}; the table starts {self.first_date.date()}")
        return self._sets[i]

    def mask(self, dates, tickers) -> pd.DataFrame:
        """Boolean frame (dates x tickers): was each ticker a member on each date?"""
        tickers = list(tickers)
        rows = [[t in self.on(d) for t in tickers] for d in pd.DatetimeIndex(dates)]
        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=tickers)

    def continuous(self, start, end) -> frozenset:
        """Tickers that were members on every change date between start and end."""
        lo = max(self._dates.searchsorted(pd.Timestamp(start).to_datetime64(), side="right") - 1, 0)
        hi = self._dates.searchsorted(pd.Timestamp(end).to_datetime64(), side="right")
        out = None
        for s in self._sets[lo:hi]:
            out = s if out is None else out & s
        return out or frozenset()


def start_end_table() -> pd.DataFrame:
    """Each ticker's entry and exit dates (a ticker can appear more than once)."""
    return pd.read_csv(START_END_FILE, parse_dates=["start_date", "end_date"])


def coverage_report(membership: Membership, available_tickers, start, end) -> dict:
    """
    How much of the point in time universe over [start, end] does a price panel cover?

    Reports the members at `start`, how many left during the window, and how many of
    those leavers the price source (`available_tickers`) can no longer supply. That
    missing share is the residual survivorship gap a free data source leaves in the
    benchmark, stated rather than hidden.
    """
    se = start_end_table()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    members_start = membership.on(start)
    left = se[(se["end_date"] >= start) & (se["end_date"] <= end) & se["ticker"].isin(members_start)]
    leavers = sorted(set(left["ticker"]))
    available = set(available_tickers)
    missing = sorted(t for t in leavers if t not in available)
    joined = sorted(membership.on(end) - members_start)
    return {
        "window": [str(start.date()), str(end.date())],
        "members_at_start": len(members_start),
        "left_during_window": len(leavers),
        "left_and_unavailable": len(missing),
        "left_and_unavailable_tickers": missing,
        "joined_during_window": len(joined),
        "share_of_start_universe_unavailable": round(len(missing) / max(len(members_start), 1), 4),
    }
