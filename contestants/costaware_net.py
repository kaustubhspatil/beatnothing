"""
Cost-aware end-to-end network: outputs portfolio weights directly and is trained to
maximise the NET Sharpe ratio (returns minus 10 bps x turnover) over windows of
consecutive trading days. Long/flat only: w_it = sigmoid(f(x_it)) / N, so each name
holds at most 1/N of capital and the rest sits in cash.

Ablation: lambda_cost in {0, 10 bps} to show what the cost term does to turnover.
Three seeds each. Early stopping on validation net Sharpe (2019-2021).
"""
import sys, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, joblib, warnings
warnings.filterwarnings("ignore")
TRAIN_END, VAL_END = "2018-12-31", "2021-12-31"
SC = "contestants"
panel = pd.read_parquet("data/feature_panel.parquet")
from beatnothing.features import FEATURE_COLUMNS as fc
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(panel.loc[panel.Date <= TRAIN_END, fc])   # fit on training years only
panel[fc] = scaler.transform(panel[fc])
tickers = sorted(panel.Ticker.unique()); N = len(tickers); tix = {t:i for i,t in enumerate(tickers)}
dates = np.array(sorted(panel.Date.unique())); dix = {d:i for i,d in enumerate(dates)}
T = len(dates); F = len(fc)
X = np.zeros((T, N, F), np.float32); Y = np.zeros((T, N), np.float32); M = np.zeros((T, N), np.float32)
ti = panel.Date.map(dix).values; ni = panel.Ticker.map(tix).values
X[ti, ni] = panel[fc].values.astype(np.float32); Y[ti, ni] = panel.target.values.astype(np.float32); M[ti, ni] = 1.0
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xg, Yg, Mg = (torch.tensor(a).to(dev) for a in (X, Y, M))
i_tr_end = int(np.searchsorted(dates, np.datetime64(TRAIN_END), side="right"))
i_va_end = int(np.searchsorted(dates, np.datetime64(VAL_END), side="right"))
i_te_end = int(np.searchsorted(dates, np.datetime64("2025-12-31"), side="right"))
print(f"tensor {X.shape}; train {i_tr_end} days, val {i_va_end-i_tr_end}, test {i_te_end-i_va_end}, forward {T-i_te_end}")

class CostAwareNet(nn.Module):
    def __init__(self, f, hidden=(128,64,32), dropout=0.3):
        super().__init__(); layers=[]; p=f
        for h in hidden: layers += [nn.Linear(p,h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]; p=h
        layers.append(nn.Linear(p,1)); self.net = nn.Sequential(*layers)
    def forward(self, x, mask):                       # x: (L, N, F) -> weights (L, N), each <= 1/N
        s = torch.sigmoid(self.net(x).squeeze(-1)) * mask
        return s / mask.sum(-1, keepdim=True).clamp(min=1)

def net_returns(w, y, mask, cost, w_prev=None):
    gross = (w * y * mask).sum(-1)
    prev = torch.cat([w_prev.unsqueeze(0) if w_prev is not None else torch.zeros_like(w[:1]), w[:-1]], 0)
    turnover = (w - prev).abs().sum(-1)
    return gross - cost * turnover, turnover

def sharpe_loss(r):
    return -(r.mean() / (r.std() + 1e-8)) * np.sqrt(252)

@torch.no_grad()
def evaluate(model, a, b, cost):
    model.eval(); w = model(Xg[a:b], Mg[a:b]); r, to = net_returns(w, Yg[a:b], Mg[a:b], cost)
    return float(r.mean()/(r.std()+1e-8)*np.sqrt(252)), float(to.sum()/((b-a)/252)), w.cpu().numpy(), r.cpu().numpy()

def train(seed, lam_cost, L=126, epochs=60, lr=1e-3, patience=8, eval_cost=10/10_000):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CostAwareNet(F).to(dev); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = -1e9, None, 0
    starts = np.arange(0, i_tr_end - L)
    for ep in range(epochs):
        model.train(); np.random.shuffle(starts)
        for s in starts[:64]:                          # 64 random half-year windows per epoch
            w = model(Xg[s:s+L], Mg[s:s+L]); r, _ = net_returns(w, Yg[s:s+L], Mg[s:s+L], lam_cost)
            loss = sharpe_loss(r); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val_sh, val_to, _, _ = evaluate(model, i_tr_end, i_va_end, eval_cost)
        if val_sh > best: best, best_state, bad = val_sh, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= patience: break
    model.load_state_dict(best_state)
    return model, best, ep

results, weights_out = {}, {}
for lam_bps in (0, 10):
    for seed in (42, 43, 44):
        t0 = time.time(); model, val_sh, ep = train(seed, lam_bps/10_000)
        te_sh, te_to, w_te, r_te = evaluate(model, i_va_end, i_te_end, 10/10_000)
        fw_sh, fw_to, w_fw, r_fw = evaluate(model, i_te_end, T, 10/10_000)
        key = f"costaware_lam{lam_bps}_seed{seed}"
        results[key] = dict(lambda_bps=lam_bps, seed=seed, val_net_sharpe=round(val_sh,3), epochs=ep,
                            test_net_sharpe=round(te_sh,3), test_turnover=round(te_to,1), test_avg_exposure=round(float(w_te.sum(-1).mean()),3),
                            fwd_net_sharpe=round(fw_sh,3), fwd_turnover=round(fw_to,1), fwd_avg_exposure=round(float(w_fw.sum(-1).mean()),3),
                            train_seconds=round(time.time()-t0,1))
        print(key, results[key], flush=True)
        W = np.concatenate([w_te, w_fw]); D = dates[i_va_end:T]
        weights_out[key] = pd.DataFrame(W, index=pd.DatetimeIndex(D), columns=tickers)
        torch.save(model.state_dict(), f"{SC}/{key}.pt")
json.dump(results, open(f"{SC}/costaware_results.json","w"), indent=2)
pd.concat(weights_out, names=["model","Date"]).to_parquet(f"{SC}/costaware_weights.parquet")
print("DONE")
