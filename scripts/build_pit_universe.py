"""
Build the point in time universe: every S&P 500 member on every day, from free data.

Sources, in priority order for each membership ticker:
  1. the Tiingo cache in data/raw/tiingo/<ticker>.csv when it holds a real history
     (at least 250 rows): acquisitions, failures and any name Yahoo deleted
  2. Yahoo under the alias in beatnothing/data/ticker_aliases.json (plain renames)
  3. Yahoo under the ticker itself (dots become dashes)

Then the reference features are computed per name with beatnothing.features, VIX and
the 10 year yield are joined, and the membership mask decides which (date, ticker)
rows exist: a name is in the panel only on days it was in the index. Outputs:

  data/pit/feature_panel_pit.parquet   member days with features and target (gitignored, rebuilt here)
  data/pit/actual_returns_pit.parquet  Date, Ticker, target for member days from 2022 (versioned)
  data/pit/coverage.json               what was covered, from where, and what is still missing
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from beatnothing import Membership, features
from beatnothing.universe import PIT_DIR

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "pit"
START, WARMUP, EVAL_START = "2021-01-01", "2021-06-01", "2022-01-01"


def tiingo_series(ticker: str) -> pd.DataFrame | None:
    p = RAW / "tiingo" / f"{ticker}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if len(d) < 250:
        return None
    d["Date"] = pd.to_datetime(d["date"].str[:10])
    out = pd.DataFrame({"Date": d["Date"], "Close": d["adjClose"], "High": d["adjHigh"], "Low": d["adjLow"],
                        "Open": d["adjOpen"], "Volume": d["adjVolume"]})
    return out.sort_values("Date").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tick = json.loads((ROOT / "data" / "pit_tickers.json").read_text())
    members, yahoo_symbol = tick["members"], tick["yahoo_symbol"]
    aliases = json.loads((PIT_DIR / "ticker_aliases.json").read_text(encoding="utf-8"))
    aliases.pop("_about", None)
    yahoo = pd.read_parquet(RAW / "pit" / "yahoo_prices.parquet")
    yahoo["Date"] = pd.to_datetime(yahoo["Date"])
    by_symbol = {s: g.drop(columns="Ticker").sort_values("Date").reset_index(drop=True) for s, g in yahoo.groupby("Ticker")}
    membership = Membership()
    se = pd.read_csv(PIT_DIR / "sp500_ticker_start_end.csv.gz", parse_dates=["start_date", "end_date"])

    blocks, source, missing = [], {}, []
    for t in members:
        exit_dates = se.loc[se["ticker"] == t, "end_date"]
        last_exit = exit_dates.max() if exit_dates.notna().any() else pd.NaT
        s = tiingo_series(t)
        if s is not None and (pd.isna(last_exit) or s["Date"].max() >= last_exit - pd.Timedelta(days=7)):
            source[t] = "tiingo"
        else:
            sym = yahoo_symbol.get(t, t)
            s = by_symbol.get(sym)
            if s is None or len(s) < 250:
                missing.append(t)
                continue
            source[t] = f"yahoo:{sym}"
        s = s[s["Date"] >= START].copy()
        if len(s) < 70:            # a brand new listing needs about 63 days of history before its first feature row
            missing.append(t)
            source.pop(t, None)
            continue
        s["Ticker"] = t
        blocks.append(s)
    prices = pd.concat(blocks, ignore_index=True)

    raw = RAW / "sp500_stocks.csv"
    vix = pd.read_csv(RAW / "vix.csv", parse_dates=["Date"], index_col="Date")["Close"].rename("vix")
    tsy = pd.read_csv(RAW / "treasury_10y.csv", parse_dates=["Date"], index_col="Date")["Close"].rename("treasury_10y")
    market = pd.concat([vix, tsy], axis=1).sort_index()
    panel = features.build_feature_panel(prices, market, tickers=sorted(prices["Ticker"].unique()))
    panel = panel[panel["Date"] >= WARMUP].reset_index(drop=True)

    # membership mask: keep a row only if the ticker was in the index on that date
    dates = sorted(panel["Date"].unique())
    member_sets = {d: membership.on(d) for d in dates}
    keep = np.fromiter((row.Ticker in member_sets[row.Date] for row in panel[["Date", "Ticker"]].itertuples(index=False)),
                       dtype=bool, count=len(panel))
    panel = panel[keep].reset_index(drop=True)
    # partial day guard: a final session caught mid download leaves a date with a handful of names
    per_day = panel.groupby("Date")["Ticker"].size()
    thin = per_day[per_day < 0.5 * per_day.median()].index
    if len(thin):
        print("dropping thin dates:", [str(d.date()) for d in thin])
        panel = panel[~panel["Date"].isin(thin)].reset_index(drop=True)
    panel.to_parquet(OUT / "feature_panel_pit.parquet", index=False)
    actual = panel[panel["Date"] >= EVAL_START][["Date", "Ticker", "target"]]
    actual.to_parquet(OUT / "actual_returns_pit.parquet", index=False)

    # coverage: member days the panel supplies against member days the index defines
    eval_dates = [d for d in dates if d >= pd.Timestamp(EVAL_START)]
    defined = sum(len(member_sets[d]) for d in eval_dates)
    supplied = int(len(actual))
    per_day = actual.groupby("Date")["Ticker"].nunique()
    report = {"window": [str(eval_dates[0].date()), str(eval_dates[-1].date())],
              "member_days_defined": int(defined), "member_days_supplied": supplied,
              "coverage": round(supplied / defined, 4),
              "names_per_day_mean": round(float(per_day.mean()), 1), "names_per_day_min": int(per_day.min()),
              "tickers_total": len(members), "from_tiingo": sorted(t for t, s in source.items() if s == "tiingo"),
              "from_yahoo_alias": sorted(t for t, s in source.items() if s.startswith("yahoo:") and s[6:] != t.replace(".", "-")),
              "missing": sorted(missing)}
    (OUT / "coverage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("from_tiingo", "from_yahoo_alias")}, indent=1))
    print("from tiingo:", len(report["from_tiingo"]), "| via alias:", len(report["from_yahoo_alias"]), "| missing:", report["missing"])


if __name__ == "__main__":
    main()
