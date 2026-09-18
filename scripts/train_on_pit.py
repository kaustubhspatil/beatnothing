"""
Train contestants on the point in time universe, which is the last hindsight left.

Every learned contestant in the first two seasons was fitted on forty eight survivor names
and then asked about five hundred, which is a handicap of unknown size. This script
removes it: the same architectures, trained on the universe that did not know the future,
with the same temporal discipline as the original capstone.

It also adds the contestant the results so far have been asking for. Every long or flat
model here ends up holding almost everything, because a model trained to predict the next
day return under squared error is trained to predict the drift, and the drift is the bar.
The way out is not a better optimiser, it is a different target: rank the names against
each other within each day, which removes the market move by construction, and trade the
ranking long and short. That model cannot inherit the bar's return, so if it shows an edge
the edge is its own.

    python scripts/train_on_pit.py --panel data/pit/feature_panel_pit_2005.parquet
    python scripts/train_on_pit.py --models ranker            # just the cross sectional one

Splits follow the original: fit on everything up to `--train-end`, choose on the years to
`--val-end`, and never look at what follows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submissions_pit"
MODELS = ROOT / "contestants" / "models_pit"
REG = "2026-09-18"
# A ranking signal on daily returns is noisy; the holding period decides whether its
# information survives the trading. Register several and let the board show the trade off.
HOLDS = (1, 5, 21)


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cross_sectional_rank(df: pd.DataFrame, col: str = "target") -> pd.Series:
    """The day's returns turned into a rank in [-0.5, 0.5]. The market move cancels."""
    r = df.groupby("Date")[col].rank(pct=True)
    return (r - 0.5).astype("float32")


def information_coefficient(frame: pd.DataFrame, actual: pd.DataFrame, max_days: int = 400) -> float:
    """Mean daily cross sectional rank correlation with the next day return. The number to
    look at before any Sharpe ratio: a real signal is 0.02 to 0.05, and a Sharpe attached
    to an information coefficient of zero is luck with a holding period."""
    wide = frame.pivot(index="Date", columns="Ticker", values="value")
    s, a = wide.align(actual, join="inner")
    ics = []
    for day in list(s.index)[:max_days]:
        both = s.loc[day].notna() & a.loc[day].notna()
        row = s.loc[day][both]
        if both.sum() >= 20 and row.nunique() > 1:
            ics.append(row.rank().corr(a.loc[day][both].rank()))
    # a signal that is the same for every name on a day has no cross sectional information,
    # which is zero rather than undefined; it is also exactly what a drift predictor produces
    return float(np.nanmean(ics)) if ics else 0.0


def hold_between(frame: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Rebalance every `hold` trading days, carrying the signal in between."""
    if hold <= 1:
        return frame
    wide = frame.pivot(index="Date", columns="Ticker", values="value").sort_index()
    keep = pd.Series(False, index=wide.index)
    keep.iloc[::hold] = True
    held = wide.where(keep, np.nan).ffill(limit=hold - 1)
    out = held.stack().rename("value").reset_index()
    out.columns = ["Date", "Ticker", "value"]
    return out


def register(name: str, frame: pd.DataFrame, rule: str, description: str, extra: dict) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    frame = frame[np.isfinite(frame["value"])].copy()
    frame["value"] = frame["value"].astype("float32")
    frame[["Date", "Ticker", "value"]].to_parquet(d / "signal.parquet", index=False, compression="zstd")
    meta = {"name": name, "kind": "predictions", "rule": rule, "registered": REG,
            "universe": "point in time S&P 500 membership; trained on the same universe",
            "description": description,
            "signal_sha256": hashlib.sha256((d / "signal.parquet").read_bytes()).hexdigest()}
    meta.update(extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    say(f"registered {name:38s} {len(frame):>9,} rows, rule {rule}")


def main(panel_path: str, train_end: str, val_end: str, which: list[str], seeds: tuple) -> None:
    import lightgbm as lgb
    import torch
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    from beatnothing.features import FEATURE_COLUMNS as FC
    from beatnothing.models import CostAwareNet, StockPredictor

    MODELS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path)
    panel["rank_target"] = cross_sectional_rank(panel)
    actual = panel.pivot(index="Date", columns="Ticker", values="target")
    train = panel[panel.Date <= train_end]
    val = panel[(panel.Date > train_end) & (panel.Date <= val_end)]
    test = panel[panel.Date > val_end].copy()
    say(f"{len(panel):,} member days: train {len(train):,} to {train_end}, "
        f"validate {len(val):,} to {val_end}, score {len(test):,} after")
    if len(train) < 50_000:
        say("WARNING: this panel is too short to train on. Build the 2005 universe first.")

    scaler = StandardScaler().fit(train[FC])
    joblib.dump(scaler, MODELS / "scaler_pit.pkl")
    Xtr, Xva, Xte = (scaler.transform(d[FC]).astype(np.float32) for d in (train, val, test))
    base = test[["Date", "Ticker"]].copy()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {"training_cutoff": train_end, "validation": f"{train_end} to {val_end}"}

    if "linear" in which:
        m = LinearRegression().fit(Xtr, train.target)
        joblib.dump(m, MODELS / "linear_pit.pkl")
        register("linear_pit", base.assign(value=m.predict(Xte)), "long_flat",
                 "Ordinary least squares on the 17 trailing features, trained on the point in time universe.",
                 {**common, "model_sha256": hashlib.sha256((MODELS / "linear_pit.pkl").read_bytes()).hexdigest()})

    if "lightgbm" in which:
        for target, rule, tag in [("target", "long_flat", "lightgbm_pit"),
                                  ("rank_target", "long_short", "lightgbm_rank_pit")]:
            t0 = time.time()
            g = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=500,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
                                  random_state=42, verbose=-1)
            g.fit(train[FC], train[target], eval_set=[(val[FC], val[target])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            joblib.dump(g, MODELS / f"{tag}.pkl")
            signal = base.assign(value=g.predict(test[FC]))
            ic = information_coefficient(signal, actual)
            say(f"{tag}: {g.best_iteration_} trees in {time.time() - t0:.0f}s, information coefficient {ic:+.4f}")
            desc = ("Gradient boosted trees on the next day return, squared error." if target == "target"
                    else "Gradient boosted trees on the within day cross sectional rank of the next day "
                         "return, so the market move is removed from the target by construction.")
            holds = [1] if target == "target" else HOLDS
            for hold in holds:
                suffix = "" if hold == 1 else f"_hold{hold}"
                register(tag + suffix, hold_between(signal, hold), rule,
                         desc + ("" if hold == 1 else f" Rebalanced every {hold} trading days, held between."),
                         {**common, "objective": target, "trees": int(g.best_iteration_),
                          "information_coefficient": round(ic, 5), "rebalance_days": hold,
                          "model_sha256": hashlib.sha256((MODELS / f"{tag}.pkl").read_bytes()).hexdigest()})

    if "ffnn" in which:
        from torch.utils.data import DataLoader, TensorDataset
        for target, rule, tag in [("target", "long_flat", "ffnn_pit"),
                                  ("rank_target", "long_short", "ffnn_rank_pit")]:
            preds = []
            for seed in seeds:
                torch.manual_seed(seed)
                net = StockPredictor(len(FC), (128, 64, 32), 0.3).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
                ytr = torch.tensor(train[target].values, dtype=torch.float32).reshape(-1, 1)
                yva = torch.tensor(val[target].values, dtype=torch.float32).reshape(-1, 1).to(dev)
                dl = DataLoader(TensorDataset(torch.tensor(Xtr), ytr), batch_size=4096, shuffle=True)
                Xva_t = torch.tensor(Xva).to(dev)
                best, best_state, bad = np.inf, None, 0
                for epoch in range(60):
                    net.train()
                    for xb, yb in dl:
                        loss = torch.nn.functional.mse_loss(net(xb.to(dev)), yb.to(dev))
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                    net.eval()
                    with torch.no_grad():
                        v = float(torch.nn.functional.mse_loss(net(Xva_t), yva))
                    if v < best - 1e-9:
                        best, best_state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
                    else:
                        bad += 1
                        if bad >= 8:
                            break
                net.load_state_dict(best_state)
                torch.save(best_state, MODELS / f"{tag}_seed{seed}.pt")
                net.eval()
                with torch.no_grad():
                    preds.append(np.concatenate([net(torch.tensor(Xte[i:i + 65536]).to(dev)).cpu().numpy().ravel()
                                                 for i in range(0, len(Xte), 65536)]))
                say(f"{tag} seed {seed}: stopped at epoch {epoch}, validation {best:.3e}")
            signal = base.assign(value=np.mean(preds, axis=0))
            ic = information_coefficient(signal, actual)
            say(f"{tag}: information coefficient {ic:+.4f}")
            desc = ("Feedforward network 128 64 32 on the next day return, squared error, mean of "
                    f"{len(seeds)} seeds." if target == "target" else
                    "Feedforward network 128 64 32 on the within day cross sectional rank of the next day "
                    f"return, mean of {len(seeds)} seeds.")
            hashes = {f"seed{s}": hashlib.sha256((MODELS / f"{tag}_seed{s}.pt").read_bytes()).hexdigest() for s in seeds}
            for hold in ([1] if target == "target" else HOLDS):
                suffix = "" if hold == 1 else f"_hold{hold}"
                register(tag + suffix, hold_between(signal, hold), rule,
                         desc + ("" if hold == 1 else f" Rebalanced every {hold} trading days, held between."),
                         {**common, "objective": target, "seeds": list(seeds), "information_coefficient": round(ic, 5),
                          "rebalance_days": hold, "model_sha256": hashes})

    say("done. Rebuild the board with: beatnothing leaderboard --track pit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="data/pit/feature_panel_pit_2005.parquet")
    ap.add_argument("--train-end", default="2018-12-31")
    ap.add_argument("--val-end", default="2021-12-31")
    ap.add_argument("--models", nargs="+", default=["linear", "lightgbm", "ffnn"],
                    choices=["linear", "lightgbm", "ffnn", "ranker"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    models = ["lightgbm", "ffnn"] if a.models == ["ranker"] else a.models
    main(a.panel, a.train_end, a.val_end, models, tuple(a.seeds))
