import numpy as np
import pytest

from beatnothing.stats import (TRADING_DAYS, deflated_sharpe, expected_max_sharpe, hac_covariance,
                               joint_sharpe_tests, ledoit_wolf_test, pbo_cscv, politis_white_block_size,
                               sharpe_diff_and_se)


def _ar1(n, rho, sigma, mu, rng):
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x + mu


def test_single_series_se_matches_the_iid_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 4000)
    sr, se = sharpe_diff_and_se(x)
    sr_daily = sr / np.sqrt(TRADING_DAYS)
    lo_se = np.sqrt((1 + 0.5 * sr_daily ** 2) / len(x)) * np.sqrt(TRADING_DAYS)
    assert abs(se - lo_se) / lo_se < 0.25          # HAC on iid data lands near the analytical value


def test_difference_se_is_near_the_independent_sum():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0006, 0.010, 5000)
    y = rng.normal(0.0002, 0.012, 5000)
    _, se_x = sharpe_diff_and_se(x)
    _, se_y = sharpe_diff_and_se(y)
    _, se_d = sharpe_diff_and_se(x, y)
    assert abs(se_d - np.hypot(se_x, se_y)) / se_d < 0.25


def test_autocorrelation_widens_the_standard_error():
    rng = np.random.default_rng(2)
    iid = rng.normal(0.0004, 0.01, 3000)
    dependent = _ar1(3000, 0.4, 0.01, 0.0004, rng)
    _, se_iid = sharpe_diff_and_se(iid)
    _, se_dep = sharpe_diff_and_se(dependent)
    assert se_dep > se_iid * 1.2                   # ignoring dependence would understate the error


def test_hac_covariance_is_symmetric_and_positive_on_the_diagonal():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(500, 4))
    S = hac_covariance(Z)
    assert np.allclose(S, S.T, atol=1e-12)
    assert np.all(np.diag(S) > 0)


def test_a_real_gap_is_detected_and_pure_noise_is_not():
    rng = np.random.default_rng(4)
    n = 1500
    common = rng.normal(0, 0.01, n)
    good = common + rng.normal(0.0012, 0.004, n)
    bar = common + rng.normal(0.0000, 0.004, n)
    res = ledoit_wolf_test(good, bar, n_boot=300, seed=5)
    assert res["p_value"] < 0.05 and res["ci_low"] > 0 and res["clears_bar"]

    same = common + rng.normal(0.0000, 0.004, n)
    null = ledoit_wolf_test(same, bar, n_boot=300, seed=6)
    assert null["p_value"] > 0.05 and not null["clears_bar"]


def test_a_bar_against_itself_is_degenerate_but_safe():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0004, 0.01, 800)
    res = ledoit_wolf_test(x, x, n_boot=100, seed=8)
    assert res["net_edge"] == 0.0 and res["se"] == 0.0
    assert res["p_value"] == 1.0 and not res["clears_bar"] and not res["clears_bar_fwe"]
    # the same must hold when the two series differ only by floating point dust
    dusty = x + np.full_like(x, 1e-18)
    res = ledoit_wolf_test(dusty, x, n_boot=100, seed=8)
    assert res["se"] == 0.0 and res["p_value"] == 1.0 and not res["clears_bar_fwe"]


def test_cash_bar_reduces_to_the_sharpe_ratio_itself():
    rng = np.random.default_rng(9)
    x = rng.normal(0.0005, 0.01, 1200)
    sr, _ = sharpe_diff_and_se(x)
    res = ledoit_wolf_test(x, np.zeros_like(x), n_boot=100, seed=10)
    assert abs(res["net_edge"] - sr) < 1e-12


def test_stepdown_is_never_more_generous_than_the_individual_test():
    rng = np.random.default_rng(11)
    n = 1200
    bar = rng.normal(0.0003, 0.01, n)
    strategies = {f"c{i}": bar + rng.normal(0.0, 0.004, n) for i in range(8)}
    strategies["winner"] = bar + rng.normal(0.0015, 0.004, n)
    bars = {k: bar for k in strategies}
    res = joint_sharpe_tests(strategies, bars, n_boot=300, seed=12)["contestants"]
    for name, r in res.items():
        assert r["p_value_fwe"] >= r["p_value"] - 1e-12
    assert res["winner"]["clears_bar_fwe"]         # a real edge still survives the correction
    assert sum(r["clears_bar_fwe"] for r in res.values()) == 1


def test_stepdown_refuses_a_board_of_pure_noise():
    rng = np.random.default_rng(13)
    n = 1000
    bar = rng.normal(0.0003, 0.01, n)
    strategies = {f"c{i}": bar + rng.normal(0.0, 0.004, n) for i in range(15)}
    res = joint_sharpe_tests(strategies, {k: bar for k in strategies}, n_boot=300, seed=14)["contestants"]
    assert not any(r["clears_bar_fwe"] for r in res.values())


def test_joint_tests_demand_one_calendar():
    a = np.zeros(100)
    with pytest.raises(ValueError):
        joint_sharpe_tests({"x": a}, {"x": np.zeros(90)}, n_boot=10)


def test_influence_series_reproduces_the_standard_error():
    from beatnothing.stats import influence_series
    rng = np.random.default_rng(19)
    x = rng.normal(0.0005, 0.01, 2000)
    y = rng.normal(0.0002, 0.01, 2000)
    inf = influence_series(x, y)
    assert abs(inf.mean()) < 1e-12                 # an influence function is centered by construction
    _, se = sharpe_diff_and_se(x, y, prewhite=False)
    naive = inf.std() / np.sqrt(len(inf)) * np.sqrt(TRADING_DAYS)
    assert abs(se - naive) / se < 0.3              # its spread is the standard error, up to the HAC correction


def test_block_size_grows_with_dependence():
    rng = np.random.default_rng(15)
    iid = rng.normal(0, 1, 2000)
    dependent = _ar1(2000, 0.6, 1.0, 0.0, rng)
    assert politis_white_block_size(dependent) > politis_white_block_size(iid)


def test_pbo_is_one_half_for_pure_noise_and_low_for_a_real_edge():
    rng = np.random.default_rng(16)
    noise = rng.normal(0.0002, 0.01, (2000, 12))
    assert 0.3 < pbo_cscv(noise, n_splits=8)["pbo"] < 0.7

    with_edge = rng.normal(0.0002, 0.01, (2000, 12))
    with_edge[:, 3] += 0.0015                      # one variant is genuinely better
    assert pbo_cscv(with_edge, n_splits=8)["pbo"] < 0.2


def test_pbo_rejects_impossible_geometry():
    rng = np.random.default_rng(17)
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(size=(100, 3)), n_splits=7)
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(size=(20, 3)), n_splits=8)


def test_deflation_gets_harsher_as_the_search_widens():
    rng = np.random.default_rng(18)
    x = rng.normal(0.0006, 0.01, 1500)
    few = deflated_sharpe(x, n_trials=2, sr_variance=0.25)
    many = deflated_sharpe(x, n_trials=500, sr_variance=0.25)
    assert many["expected_max_sharpe"] > few["expected_max_sharpe"]
    assert many["deflated_sharpe"] < few["deflated_sharpe"]
    assert expected_max_sharpe(1000, 0.0) == 0.0
