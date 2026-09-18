"""
Score the frozen reference contestants on the point in time universe.

Nothing is retrained. Every model was fitted on the 48 survivor names (data to 2018)
and is now asked about every S&P 500 member on every day since 2022, including the
names that later vanished. Cross sectional models (linear, feedforward, LightGBM, the
cost aware nets) score any name from its own trailing features; the sequence models
score any name from its own 60 day window. Kronos is left out here: 600 names times
1,200 days of autoregressive generation is a day of GPU time, and its verdict on the
survivor track already stands.

Writes submissions_pit/<name>/{signal.parquet, meta.json}.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from beatnothing.features import FEATURE_COLUMNS as FC
from beatnothing.models import CostAwareNet, FinancialCNN1D, FinancialLSTM, StockPredictor

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "contestants" / "models"
OUT = ROOT / "submissions_pit"
SEQ_LEN, EVAL_START, REG = 60, "2022-01-01", "2026-09-18"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def submit(name: str, frame: pd.DataFrame, kind: str, description: str, extra: dict | None = None) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    frame = frame[frame["Date"] >= EVAL_START].copy()
    frame["value"] = frame["value"].astype("float32")      # 575k rows per contestant: keep the repository lean
    frame[["Date", "Ticker", "value"]].to_parquet(d / "signal.parquet", index=False, compression="zstd")
    meta = {"name": name, "kind": kind, "training_cutoff": "2018-12-31", "validation": "2019-01-01 to 2021-12-31",
            "registered": REG, "universe": "point in time S&P 500 membership; model trained on 48 survivor names",
            "description": description}
    meta.update(extra or {})
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{name:40s} {len(frame):>8,} rows  {frame['Date'].min().date()} -> {frame['Date'].max().date()}")


def predict_sequences(model, panel: pd.DataFrame, X_scaled: np.ndarray, dev) -> tuple[np.ndarray, np.ndarray]:
    """Score 60 day windows one ticker at a time (600 names x 1,200 days of windows would not fit in memory).
    Windows never cross a ticker boundary; the first 59 member days of each name are skipped."""
    order = panel.sort_values(["Ticker", "Date"]).index.to_numpy()
    tick = panel.loc[order, "Ticker"].to_numpy()
    preds, ends = [], []
    start = 0
    with torch.no_grad():
        for i in range(1, len(order) + 1):
            if i == len(order) or tick[i] != tick[start]:
                block = X_scaled[order[start:i]]
                if len(block) >= SEQ_LEN:
                    win = np.lib.stride_tricks.sliding_window_view(block, (SEQ_LEN, block.shape[1]))[:, 0]
                    win = np.ascontiguousarray(win, dtype=np.float32)
                    preds.append(model(torch.tensor(win).to(dev)).cpu().numpy().ravel())
                    ends.append(order[start + SEQ_LEN - 1:i])
                start = i
    return np.concatenate(preds), np.concatenate(ends)


def main() -> None:
    panel = pd.read_parquet(ROOT / "data" / "pit" / "feature_panel_pit.parquet")
    scaler = joblib.load(MODELS / "feature_scaler.pkl")
    X = scaler.transform(panel[FC]).astype(np.float32)
    base = panel[["Date", "Ticker"]].copy()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # linear and feedforward
    lin = joblib.load(MODELS / "linear_baseline.pkl")
    submit("linear_incumbent", base.assign(value=lin.predict(X)), "predictions",
           "OLS on the 17 trailing features, frozen 2026 08 20, scored on every member.", {"model_sha256": sha(MODELS / "linear_baseline.pkl")})
    ffnn = StockPredictor(len(FC))
    ffnn.load_state_dict(torch.load(MODELS / "ffnn_best.pt", map_location="cpu"))
    ffnn.eval()
    with torch.no_grad():
        p = np.concatenate([ffnn(torch.tensor(X[i:i + 65536])).numpy().ravel() for i in range(0, len(X), 65536)])
    submit("ffnn_128_64_32", base.assign(value=p), "predictions",
           "Feedforward 128 64 32, MSE, frozen 2026 08 20, scored on every member.", {"model_sha256": sha(MODELS / "ffnn_best.pt")})
    gbm = joblib.load(MODELS / "lgbm_baseline.pkl")
    submit("lightgbm_mse", base.assign(value=gbm.predict(panel[FC])), "predictions",
           "LightGBM, MSE objective, early stopped after three trees, scored on every member.", {"model_sha256": sha(MODELS / "lgbm_baseline.pkl")})
    submit("always_long", base.assign(value=1.0), "predictions", "The universe bar: long every member every day.")

    # sequence models, one ticker at a time
    for name, ctor, file, desc in [("lstm_60d", lambda: FinancialLSTM(len(FC)), "lstm_60d.pt", "Two layer LSTM over 60 day windows"),
                                   ("cnn1d_60d", lambda: FinancialCNN1D(len(FC)), "cnn1d_60d.pt", "Three block 1D CNN over 60 day windows")]:
        m = ctor()
        m.load_state_dict(torch.load(MODELS / file, map_location="cpu"))
        m.to(dev).eval()
        p, ends = predict_sequences(m, panel, X, dev)
        seq_base = panel.loc[ends, ["Date", "Ticker"]].reset_index(drop=True)
        submit(name, seq_base.assign(value=p), "predictions", f"{desc}, frozen 2026 08 20, scored on every member.",
               {"model_sha256": sha(MODELS / file)})

    # cost aware nets: weights per (date, name) = sigmoid / N over the names present that day, mean of three seeds
    dates = np.sort(panel["Date"].unique())
    for lam in (0, 10):
        acc = None
        for seed in (42, 43, 44):
            net = CostAwareNet(len(FC))
            net.load_state_dict(torch.load(MODELS / f"costaware_lam{lam}_seed{seed}.pt", map_location="cpu"))
            net.eval()
            with torch.no_grad():
                s = torch.sigmoid(net.net(torch.tensor(X)).squeeze(-1)).numpy()
            acc = s if acc is None else acc + s
        s = acc / 3
        n_per_day = panel.groupby("Date")["Ticker"].transform("size").to_numpy()
        submit(f"costaware_net_lambda{lam}bps_3seed_mean", base.assign(value=s / n_per_day), "weights",
               f"Cost aware network, {lam} bps turnover term, mean of seeds 42 43 44, weights = sigmoid / N over the members present that day.",
               {"model_sha256": {f"seed{sd}": sha(MODELS / f"costaware_lam{lam}_seed{sd}.pt") for sd in (42, 43, 44)}, "lambda_bps": lam})


if __name__ == "__main__":
    main()
