import numpy as np
import pandas as pd

from beatnothing import Backtest
from beatnothing.leaderboard import TRACKS


def test_weights_are_zeroed_on_days_a_name_is_not_in_the_universe():
    idx = pd.bdate_range("2022-01-03", periods=10)
    actual = pd.DataFrame(0.01, index=idx, columns=["A", "B"])
    actual.loc[idx[5:], "B"] = np.nan                      # B leaves the index after five days
    w = pd.DataFrame(0.5, index=idx, columns=["A", "B"])   # a contestant that keeps holding B
    bt = Backtest(actual, weights=w, cost_bps=0.0)
    assert bt.weights.loc[idx[6], "B"] == 0.0
    assert bt.weights.loc[idx[6], "A"] == 0.5
    # after the exit only A earns: half the book in A at 1% a day
    assert abs(bt.daily_returns.loc[idx[7]] - 0.005) < 1e-12
    assert abs(bt.daily_returns.loc[idx[2]] - 0.010) < 1e-12


def test_prediction_path_ignores_non_members_too():
    idx = pd.bdate_range("2022-01-03", periods=6)
    actual = pd.DataFrame(0.01, index=idx, columns=["A", "B"])
    actual.loc[idx[3:], "B"] = np.nan
    pred = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    bt = Backtest(actual, predictions=pred, cost_bps=0.0)
    assert bt.stats()["avg_positions"] == 1.5              # two names for three days, one for three


def test_tracks_are_declared():
    assert set(TRACKS) == {"survivor48", "pit"}
    for cfg in TRACKS.values():
        assert {"actual", "submissions", "out", "universe"} <= set(cfg)
