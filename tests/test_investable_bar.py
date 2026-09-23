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
    inv = a.mean(axis=1) - 0.0003                       # index lags the universe every day
    stats, r, b = score_window(a, always, "predictions", a.index[0], a.index[-1], bar_returns(a),
                               n_boot=200, investable=inv)
    assert np.allclose(r.values, b.values)              # bar contestant vs itself
    assert stats["edge_vs_investable"] > 0              # beats a lagging index
    assert stats["investable_days"] == len(a)
    assert stats["ci_low_vs_investable"] <= stats["edge_vs_investable"] <= stats["ci_high_vs_investable"]


def test_investable_bar_is_optional_and_needs_overlap():
    a = _actual()
    always = pd.DataFrame(1.0, index=a.index, columns=a.columns)
    stats, _, _ = score_window(a, always, "predictions", a.index[0], a.index[-1], bar_returns(a), n_boot=50)
    assert "edge_vs_investable" not in stats
    disjoint = pd.Series(0.0, index=pd.bdate_range("2010-01-01", periods=100))
    stats, _, _ = score_window(a, always, "predictions", a.index[0], a.index[-1], bar_returns(a), n_boot=50,
                               investable=disjoint)
    assert "edge_vs_investable" not in stats            # < 20 overlapping days, no claim


def test_score_window_pads_a_silent_contestant_to_cash():
    a = _actual(n=200)
    talkative = pd.DataFrame(1.0, index=a.index, columns=a.columns)
    silent = talkative.iloc[:100]                       # stops submitting halfway through
    stats, r, _ = score_window(a, silent, "predictions", a.index[0], a.index[-1], bar_returns(a), n_boot=20)
    assert stats["days"] == len(a)                      # still covers every day
    assert np.allclose(r.values[120:], 0.0)             # silent half is flat, not missing
