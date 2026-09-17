"""
Kronos small (24.7M params, AAAI 2026, MIT) zero shot next day forecasts for every stock and
every decision date from 2022 01 03. Context: the trailing 400 daily bars; prediction: the next
bar; signal = predicted close over last close minus 1; three samples averaged.

Requires the Kronos repo on the path (git clone https://github.com/shiyu-coder/Kronos kronos)
and the raw prices from scripts/download_data.py.
"""
import sys, time, os, warnings; warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
import pandas as pd, numpy as np, torch
sys.path.insert(0, "kronos")
from model import Kronos, KronosTokenizer, KronosPredictor
CTX, SAMPLES = 400, 3
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
pred = KronosPredictor(mdl, tok, device="cuda", max_context=512)
px = pd.read_csv("data/raw/sp500_stocks.csv", parse_dates=["Date"]).sort_values(["Ticker","Date"])
tickers = sorted(t for t in px.Ticker.unique() if t not in ("SPY","QQQ"))
by = {t: px[px.Ticker==t].reset_index(drop=True) for t in tickers}
dates = sorted(px.Date.unique()); dates = [d for d in dates if pd.Timestamp("2022-01-01") <= d <= pd.Timestamp("2026-09-15")]
out_path = "contestants/kronos_preds.parquet"
done = pd.read_parquet(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=["Date","Ticker","pred_kronos"])
done_dates = set(pd.to_datetime(done.Date.unique())) if len(done) else set()
rows = []; t0 = time.time(); n=0
for d in dates:
    if d in done_dates: continue
    dfs, xts, yts, names, last = [], [], [], [], []
    for t in tickers:
        g = by[t]; i = g.Date.searchsorted(d, side="right")  # rows with Date <= d
        if i < 100: continue
        g_ctx = g.iloc[max(0,i-CTX):i]
        if i >= len(g): continue
        nxt = g.iloc[i]
        dd = g_ctx.rename(columns=str.lower)[["open","high","low","close","volume"]].reset_index(drop=True)
        dd["amount"] = dd["close"]*dd["volume"]
        dfs.append(dd); xts.append(pd.Series(g_ctx.Date.values)); yts.append(pd.Series([nxt.Date]))
        names.append(t); last.append(float(g_ctx.close.iloc[-1]) if "close" in g_ctx else float(g_ctx.Close.iloc[-1]))
    preds = pred.predict_batch(dfs, xts, yts, pred_len=1, T=1.0, top_p=0.9, sample_count=SAMPLES, verbose=False)
    for t, o, l in zip(names, preds, last):
        rows.append({"Date": d, "Ticker": t, "pred_kronos": float(o["close"].iloc[0]/l - 1)})
    n += 1
    if n % 25 == 0:
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(out_path, index=False)
        el = time.time()-t0; print(f"{n} dates done ({d.date()}), {el/60:.1f} min, eta {(len(dates)-len(done_dates)-n)*el/n/60:.1f} min", flush=True)
pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(out_path, index=False)
print("DONE", len(rows), "rows", (time.time()-t0)/60, "min")
