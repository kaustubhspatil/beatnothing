"""Download every point in time member's Yahoo history in batches; record what Yahoo lacks."""
import json, time, warnings; warnings.filterwarnings("ignore")
import pandas as pd, yfinance as yf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
tick = json.load(open(ROOT / "data" / "pit_tickers.json"))
symbols = sorted(set(tick["yahoo_symbol"].values()))
frames, failed = [], []
for i in range(0, len(symbols), 50):
    batch = symbols[i:i + 50]
    for attempt in range(3):
        try:
            data = yf.download(batch, start="2005-01-01", end="2026-09-18", auto_adjust=True, group_by="ticker", threads=True, progress=False)
            break
        except Exception as e:
            print("batch retry", i, e, flush=True); time.sleep(10)
    for s in batch:
        try:
            d = data[s].dropna(how="all")
        except Exception:
            failed.append(s); continue
        if len(d) < 5:
            failed.append(s); continue
        d = d[["Close", "High", "Low", "Open", "Volume"]].copy(); d["Ticker"] = s; d.index.name = "Date"
        frames.append(d.reset_index())
    print(f"batch {i // 50 + 1}/{(len(symbols) + 49) // 50} done; failures so far {len(failed)}", flush=True)
    time.sleep(2)
out = pd.concat(frames, ignore_index=True)
out.to_parquet(ROOT / "data" / "raw" / "pit" / "yahoo_prices.parquet", index=False)
json.dump(failed, open(ROOT / "data" / "raw" / "pit" / "yahoo_failures.json", "w"), indent=1)
print("DONE rows", len(out), "symbols", out.Ticker.nunique(), "failed", len(failed), failed)
