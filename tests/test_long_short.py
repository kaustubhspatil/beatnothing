import numpy as np
import pandas as pd
import pytest

from beatnothing import Backtest
from beatnothing.engine import weights_from_predictions


def _actual(n=300, k=20, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(rng.normal(0.0003, 0.015, (n, k)), index=idx, columns=[f"S{i}" for i in range(k)])


def test_long_short_is_dollar_neutral_with_gross_one():
    a = _actual()
    pred = pd.DataFrame(np.random.default_rng(1).normal(size=a.shape), index=a.index, columns=a.columns)
    w = weights_from_predictions(pred, a, rule="long_short", quantile=0.1)
    assert np.allclose(w.sum(axis=1), 0.0)
    assert np.allclose(w.abs().sum(axis=1), 1.0)
    assert (w > 0).sum(axis=1).eq(2).all() and (w < 0).sum(axis=1).eq(2).all()   # top and bottom 2 of 20


def test_long_top_holds_the_top_decile_only():
    a = _actual()
    pred = pd.DataFrame(np.random.default_rng(2).normal(size=a.shape), index=a.index, columns=a.columns)
    w = weights_from_predictions(pred, a, rule="long_top", quantile=0.1)
    assert (w > 0).sum(axis=1).eq(2).all()
    assert np.allclose(w.sum(axis=1), 1.0)


def test_borrow_is_charged_on_the_short_book_only():
    a = _actual()
    a[:] = 0.0                                                   # no price moves, only costs
    pred = pd.DataFrame(np.random.default_rng(3).normal(size=a.shape), index=a.index, columns=a.columns)
    pred[:] = pred.iloc[0].values                                # constant ranking, no turnover after day one
    bt = Backtest(a, predictions=pred, rule="long_short", cost_bps=0.0, borrow_bps_annual=252 * 100)  # 1% a day on shorts
    assert bt.stats()["avg_short_exposure"] == pytest.approx(0.5)
    assert bt.daily_returns.iloc[5] == pytest.approx(-0.005)     # half the book short at 1%/day
    long_only = Backtest(a, predictions=pred, rule="long_top", cost_bps=0.0, borrow_bps_annual=252 * 100)
    assert long_only.daily_returns.iloc[5] == pytest.approx(0.0)


def test_short_weights_need_permission_and_gross_cap_holds():
    a = _actual()
    w = pd.DataFrame(-0.02, index=a.index, columns=a.columns)
    with pytest.raises(ValueError):
        Backtest(a, weights=w)
    bt = Backtest(a, weights=w, allow_short=True)
    assert bt.stats()["avg_short_exposure"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        Backtest(a, weights=w * 3, allow_short=True)             # gross 1.2 > 1
