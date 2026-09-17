import numpy as np
import pandas as pd

from beatnothing.leaderboard import bar_returns, score_window


def _actual(n=500, k=6, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(rng.normal(0.0004, 0.012, (n, k)), index=idx, columns=[f"S{i}" for i in range(k)])


def test_investable_bar_adds_a_second_edge():
    a = _actual()
    always = pd.DataFrame(1.0, index=a.index, columns=a.columns)
    inv = a.mean(axis=1) - 0.0003                       # an index that lags the universe every day
    res = score_window(a, always, "predictions", a.index[0], a.index[-1], bar_returns(a),
                       n_boot=200, investable=inv)
    assert res["net_edge"] == 0.0                       # the bar against itself
    assert res["edge_vs_investable"] > 0                # the bar beats a lagging index
    assert res["investable_days"] == len(a)
    assert res["ci_low_vs_investable"] <= res["edge_vs_investable"] <= res["ci_high_vs_investable"]


def test_investable_bar_is_optional_and_needs_overlap():
    a = _actual()
    always = pd.DataFrame(1.0, index=a.index, columns=a.columns)
    res = score_window(a, always, "predictions", a.index[0], a.index[-1], bar_returns(a), n_boot=50)
    assert "edge_vs_investable" not in res
    disjoint = pd.Series(0.0, index=pd.bdate_range("2010-01-01", periods=100))
    res = score_window(a, always, "predictions", a.index[0], a.index[-1], bar_returns(a), n_boot=50,
                       investable=disjoint)
    assert "edge_vs_investable" not in res             # fewer than 20 overlapping days: no claim made
