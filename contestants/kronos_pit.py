"""
Kronos small, zero shot, on the point in time universe.

The survivor track already established what happens when this model trades every day: a
gross Sharpe of 0.52 becomes a net minus 0.79 at 241 turns of the book a year. Repeating
that is not informative. The question left is whether the model has any information at all
once it is not traded to death, so here it forecasts every `--stride` trading days and the
signal is held between, which is how the factor contestants are traded too.

Resumable: a checkpoint every ten decision dates, and an out of memory error on one date
skips that date rather than ending the run. Batches are grouped by context length because
Kronos requires equal lengths within a batch, and kept small because a laptop card has
eight gigabytes and five hundred names a day will find the end of them.

Needs the Kronos repository cloned beside this one as ./kronos (MIT) and the merged price
panel from scripts/build_pit_universe.py.

    python contestants/kronos_pit.py --stride 5 --batch 24
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# expandable segments keeps a long run of many different batch shapes from fragmenting
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "kronos")
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

OUT = "contestants/kronos_pit_preds.parquet"


def main(stride: int, batch: int, context: int, samples: int) -> None:
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(mdl, tok, device="cuda" if torch.cuda.is_available() else "cpu", max_context=512)

    px = pd.read_parquet("data/pit/prices_pit.parquet").sort_values(["Ticker", "Date"])
    members = pd.read_parquet("data/pit/actual_returns_pit.parquet")[["Date", "Ticker"]]
    by = {t: g.reset_index(drop=True) for t, g in px.groupby("Ticker")}
    all_dates = sorted(members["Date"].unique())
    decision_dates = all_dates[::stride]
    mem_by_date = members.groupby("Date")["Ticker"].apply(list).to_dict()

    done = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["Date", "Ticker", "value"])
    done_dates = set(pd.to_datetime(done["Date"].unique())) if len(done) else set()
    todo = [d for d in decision_dates if d not in done_dates]
    print(f"{len(all_dates)} trading days, forecasting every {stride} of them: "
          f"{len(decision_dates)} decision dates, {len(todo)} still to do", flush=True)

    rows, n, t0, skipped = [], 0, time.time(), []
    for d in todo:
        dfs, xts, yts, names, last = [], [], [], [], []
        for t in mem_by_date[d]:
            g = by.get(t)
            if g is None:
                continue
            i = g["Date"].searchsorted(d, side="right")
            if i < 100 or i >= len(g):
                continue
            ctx = g.iloc[max(0, i - context):i]
            dd = ctx.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
            dd["amount"] = dd["close"] * dd["volume"]
            dfs.append(dd)
            xts.append(pd.Series(ctx["Date"].values))
            yts.append(pd.Series([g["Date"].iloc[i]]))
            names.append(t)
            last.append(float(ctx["Close"].iloc[-1]))

        groups: dict[int, list[int]] = {}
        for k in range(len(dfs)):
            groups.setdefault(len(dfs[k]), []).append(k)
        try:
            for idx in groups.values():
                for b in range(0, len(idx), batch):
                    sel = idx[b:b + batch]
                    out = pred.predict_batch([dfs[k] for k in sel], [xts[k] for k in sel], [yts[k] for k in sel],
                                             pred_len=1, T=1.0, top_p=0.9, sample_count=samples, verbose=False)
                    for k, o in zip(sel, out):
                        rows.append({"Date": d, "Ticker": names[k], "value": float(o["close"].iloc[0] / last[k] - 1)})
        except torch.cuda.OutOfMemoryError:
            skipped.append(str(pd.Timestamp(d).date()))
            rows = [r for r in rows if r["Date"] != d]          # a half scored date is worse than none
            print(f"  out of memory on {pd.Timestamp(d).date()}, skipping it", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n += 1
        if n % 10 == 0 or n == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(OUT, index=False)
            rows = []
            el = time.time() - t0
            print(f"{n}/{len(todo)} dates ({pd.Timestamp(d).date()}), {el / 60:.1f} min, "
                  f"eta {(len(todo) - n) * el / n / 60:.0f} min, {len(done):,} forecasts", flush=True)
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_parquet(OUT, index=False)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min, {len(done):,} forecasts"
          + (f", skipped {len(skipped)} dates: {skipped}" if skipped else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="forecast every Nth trading day")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--context", type=int, default=400)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    main(a.stride, a.batch, a.context, a.samples)
