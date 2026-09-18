"""
Kronos small, zero shot, on the point in time universe: every member on every decision
date from 2022, about 590,000 forecasts. Resumable: a parquet checkpoint every ten dates.

Needs the Kronos repository cloned next to this repo as ./kronos (MIT) and the merged
price panel from scripts/build_pit_universe.py.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "kronos")
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

CTX, SAMPLES, BATCH = 400, 3, 64
OUT = "contestants/kronos_pit_preds.parquet"

tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
pred = KronosPredictor(mdl, tok, device="cuda", max_context=512)

px = pd.read_parquet("data/pit/prices_pit.parquet").sort_values(["Ticker", "Date"])
members = pd.read_parquet("data/pit/actual_returns_pit.parquet")[["Date", "Ticker"]]
by = {t: g.reset_index(drop=True) for t, g in px.groupby("Ticker")}
dates = sorted(members["Date"].unique())
mem_by_date = members.groupby("Date")["Ticker"].apply(list).to_dict()

done = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["Date", "Ticker", "value"])
done_dates = set(pd.to_datetime(done["Date"].unique())) if len(done) else set()
rows, n, t0 = [], 0, time.time()
for d in dates:
    if d in done_dates:
        continue
    dfs, xts, yts, names, last = [], [], [], [], []
    for t in mem_by_date[d]:
        g = by.get(t)
        if g is None:
            continue
        i = g["Date"].searchsorted(d, side="right")
        if i < 100 or i >= len(g):
            continue
        ctx = g.iloc[max(0, i - CTX):i]
        dd = ctx.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        dd["amount"] = dd["close"] * dd["volume"]
        dfs.append(dd)
        xts.append(pd.Series(ctx["Date"].values))
        yts.append(pd.Series([g["Date"].iloc[i]]))
        names.append(t)
        last.append(float(ctx["Close"].iloc[-1]))
    # Kronos batches need equal context lengths: group by length (new listings are short)
    groups = {}
    for k in range(len(dfs)):
        groups.setdefault(len(dfs[k]), []).append(k)
    for idx in groups.values():
        for b in range(0, len(idx), BATCH):
            sel = idx[b:b + BATCH]
            out = pred.predict_batch([dfs[k] for k in sel], [xts[k] for k in sel], [yts[k] for k in sel], pred_len=1,
                                     T=1.0, top_p=0.9, sample_count=SAMPLES, verbose=False)
            for k, o in zip(sel, out):
                rows.append({"Date": d, "Ticker": names[k], "value": float(o["close"].iloc[0] / last[k] - 1)})
    n += 1
    if n % 10 == 0:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_parquet(OUT, index=False)
        rows = []
        el = time.time() - t0
        print(f"{n} dates ({d.date()}), {el / 60:.1f} min, eta {(len(dates) - len(done_dates) - n) * el / n / 60:.0f} min", flush=True)
pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(OUT, index=False)
print("DONE", (time.time() - t0) / 60, "min")
