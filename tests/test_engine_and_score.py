import numpy as np
import pandas as pd
import pytest

from beatnothing import Backtest, net_edge, sharpe, cost_grid
from beatnothing.canary import peek, hindsight_universe, off_by_one, truncation_test
from beatnothing.engine import weights_from_predictions


def _panel(n_days=600, n_stocks=8, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i}" for i in range(n_stocks)]
    r = pd.DataFrame(rng.normal(0.0003, 0.015, (n_days, n_stocks)), index=idx, columns=cols)
    return r


def test_always_long_matches_equal_weight_mean_before_costs():
    r = _panel()
    pred = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    bt = Backtest(r, predictions=pred, cost_bps=0.0)
    assert np.allclose(bt.daily_returns.values, r.mean(axis=1).values)
    assert bt.turnover.iloc[0] == pytest.approx(1.0)
    assert bt.turnover.iloc[1:].abs().max() < 1e-12


def test_costs_are_charged_on_every_flip():
    r = _panel()
    pattern = np.where(np.arange(len(r)) % 2 == 0, 1.0, -1.0)
    flip = pd.DataFrame(np.tile(pattern[:, None], (1, r.shape[1])), index=r.index, columns=r.columns)
    bt0 = Backtest(r, predictions=flip, cost_bps=0.0)
    bt10 = Backtest(r, predictions=flip, cost_bps=10.0)
    assert bt10.daily_returns.mean() < bt0.daily_returns.mean()
    # a full flip every day costs 1 unit of turnover per day (2 when re entering)
    assert bt10.turnover.iloc[2:].mean() == pytest.approx(1.0)


def test_weights_contract_is_enforced():
    r = _panel()
    w = pd.DataFrame(0.5, index=r.index, columns=r.columns)   # sums to 4 > 1
    with pytest.raises(ValueError):
        Backtest(r, weights=w)
    w = pd.DataFrame(-0.1, index=r.index, columns=r.columns)
    with pytest.raises(ValueError):
        Backtest(r, weights=w)
    with pytest.raises(ValueError):
        Backtest(r)


def test_weights_and_predictions_paths_agree():
    r = _panel()
    pred = pd.DataFrame(np.random.default_rng(3).normal(size=r.shape), index=r.index, columns=r.columns)
    w = weights_from_predictions(pred, r)
    a = Backtest(r, predictions=pred).daily_returns
    b = Backtest(r, weights=w).daily_returns
    assert np.allclose(a.values, b.values)


def test_net_edge_of_the_bar_against_itself_is_zero_and_never_clears():
    r = _panel()
    bar = r.mean(axis=1).values
    out = net_edge(bar, bar, n_boot=200)
    assert out["net_edge"] == pytest.approx(0.0)
    assert not out["clears_bar"]
    assert out["ci_low"] <= 0 <= out["ci_high"]


def test_peek_canary_is_loud():
    r = _panel()
    bt = Backtest(r, predictions=peek(r))
    assert bt.stats()["net_sharpe"] > 5.0


def test_hindsight_canary_beats_the_bar():
    r = _panel(n_stocks=30)
    bt = Backtest(r, predictions=hindsight_universe(r, top_n=5), cost_bps=0.0)
    bar = Backtest(r, predictions=pd.DataFrame(1.0, index=r.index, columns=r.columns), cost_bps=0.0)
    assert bt.stats()["total_return"] > bar.stats()["total_return"]


def test_off_by_one_shift_is_the_wrong_direction():
    r = _panel()
    f = r.rolling(5).mean()
    shifted = off_by_one(f)
    assert np.allclose(shifted.iloc[10].values, f.iloc[11].values)


def test_truncation_test_passes_for_trailing_features_and_fails_for_centered():
    r = _panel(n_stocks=3)
    prices = (1 + r).cumprod().stack().rename("Close").reset_index()
    prices.columns = ["Date", "Ticker", "Close"]

    def trailing(px):
        out = px.sort_values(["Ticker", "Date"]).copy()
        out["sma5"] = out.groupby("Ticker")["Close"].transform(lambda s: s.rolling(5).mean())
        out["target"] = out.groupby("Ticker")["Close"].transform(lambda s: s.pct_change().shift(-1))
        return out[["Date", "Ticker", "sma5", "target"]]

    def centered(px):
        out = px.sort_values(["Ticker", "Date"]).copy()
        out["z"] = out.groupby("Ticker")["Close"].transform(lambda s: (s - s.mean()) / s.std())
        return out[["Date", "Ticker", "z"]]

    assert truncation_test(trailing, prices, "2021-06-30")["passed"]
    assert not truncation_test(centered, prices, "2021-06-30")["passed"]


def test_cost_grid_is_monotone_for_a_trading_signal():
    r = _panel()
    pred = pd.DataFrame(np.random.default_rng(5).normal(size=r.shape), index=r.index, columns=r.columns)
    g = cost_grid(r, predictions=pred)
    vals = [g[c] for c in sorted(g)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_sharpe_basics():
    assert sharpe(np.zeros(10)) == 0.0
    assert sharpe(np.full(252, 0.001) + np.random.default_rng(0).normal(0, 1e-4, 252)) > 10
