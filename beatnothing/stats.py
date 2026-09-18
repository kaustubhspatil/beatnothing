"""
Statistics: is the edge real, or is it fifteen contestants and a coin?

Three problems stand between a number on a leaderboard and a claim anyone should act on.

**The Sharpe difference is not normal and not independent.** Daily returns are fat
tailed and their volatility clusters, so the textbook standard error understates the
uncertainty. `sharpe_diff_and_se` uses the delta method over the four moments that
define two Sharpe ratios, with a prewhitened kernel covariance (Andrews and Monahan)
that is robust to both heteroskedasticity and autocorrelation. `joint_sharpe_tests`
then studentizes a circular block bootstrap, which is the Ledoit and Wolf (2008)
procedure: resample blocks, recompute both the difference and its standard error inside
every resample, and compare the centered studentized statistic with the one observed.

**A leaderboard is a multiple test.** Fifteen contestants each tested at five percent
produce a false winner better than half the time. The same joint bootstrap feeds a
Romano and Wolf (2005) stepdown, which controls the probability of even one false claim
across the entire board while keeping far more power than Bonferroni, because it
resamples all contestants on the same dates and so inherits their dependence.

**A submitter with many variants finds one that worked.** `pbo_cscv` implements the
combinatorially symmetric cross validation of Bailey, Borwein, Lopez de Prado and Zhu
(2014): split the record into subsets, choose the best variant on half of them, and see
where it ranks on the other half. Pure noise lands at one half. `deflated_sharpe` is the
same idea in closed form, discounting a Sharpe ratio by how many trials it took to find.

Everything here is computed in daily units and reported annualized.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew

TRADING_DAYS = 252
EULER = 0.5772156649015329
_TINY = 1e-300


# ── The delta method over Sharpe moments ─────────────────────────────────

def _sr_and_grad(v: np.ndarray) -> tuple[float, np.ndarray]:
    """Sharpe of one series from its moments (mean, mean of squares) and the gradient."""
    m, g = v
    d = max(g - m * m, _TINY)
    sr = m / np.sqrt(d)
    grad = np.array([g / d ** 1.5, -m / (2 * d ** 1.5)])
    return float(sr), grad


def _sr_diff_and_grad(v: np.ndarray) -> tuple[float, np.ndarray]:
    """SR(x) minus SR(y) from the four moments, and the gradient with respect to them."""
    m1, m2, g1, g2 = v
    d1 = max(g1 - m1 * m1, _TINY)
    d2 = max(g2 - m2 * m2, _TINY)
    diff = m1 / np.sqrt(d1) - m2 / np.sqrt(d2)
    grad = np.array([g1 / d1 ** 1.5, -g2 / d2 ** 1.5,
                     -m1 / (2 * d1 ** 1.5), m2 / (2 * d2 ** 1.5)])
    return float(diff), grad


def andrews_bandwidth(U: np.ndarray) -> float:
    """Andrews (1991) automatic bandwidth for the Bartlett kernel, AR(1) plug in."""
    n = len(U)
    num = den = 0.0
    for a in range(U.shape[1]):
        x = U[:, a]
        denom = float((x[:-1] ** 2).sum())
        rho = float((x[:-1] * x[1:]).sum() / denom) if denom > 0 else 0.0
        rho = float(np.clip(rho, -0.97, 0.97))
        s2 = float(((x[1:] - rho * x[:-1]) ** 2).mean())
        num += 4 * rho ** 2 * s2 ** 2 / ((1 - rho) ** 6 * (1 + rho) ** 2)
        den += s2 ** 2 / (1 - rho) ** 4
    alpha1 = num / den if den > 0 else 0.0
    return float(min(max(1.1447 * (alpha1 * n) ** (1 / 3), 1.0), max(n - 2, 1)))


def hac_covariance(Z: np.ndarray, prewhite: bool = True, bandwidth: float | None = None) -> np.ndarray:
    """
    Long run covariance of the columns of Z, robust to heteroskedasticity and
    autocorrelation. Bartlett kernel with an automatic bandwidth, optionally after
    prewhitening each column with an AR(1) and recoloring afterwards.
    """
    Z = np.asarray(Z, float)
    Zc = Z - Z.mean(0)
    k = Zc.shape[1]
    A = np.zeros(k)
    if prewhite and len(Zc) > 3:
        for a in range(k):
            den = float((Zc[:-1, a] ** 2).sum())
            rho = float((Zc[:-1, a] * Zc[1:, a]).sum() / den) if den > 0 else 0.0
            A[a] = float(np.clip(rho, -0.97, 0.97))
        U = Zc[1:] - Zc[:-1] * A
    else:
        A = np.zeros(k)
        U = Zc
    n = len(U)
    if n < 3:
        return U.T @ U / max(n, 1)
    L = andrews_bandwidth(U) if bandwidth is None else float(bandwidth)
    S = U.T @ U / n
    for j in range(1, int(L) + 1):
        w = 1.0 - j / (L + 1.0)
        G = U[j:].T @ U[:-j] / n
        S = S + w * (G + G.T)
    if prewhite:
        D = np.diag(1.0 / (1.0 - A))
        S = D @ S @ D.T
    return S


def influence_series(x, y=None) -> np.ndarray:
    """
    The per observation influence of the Sharpe difference: how much each day moves the
    statistic. Its long run variance is the variance of the statistic, so this is the
    series whose dependence decides the bootstrap block length. Choosing the block from
    the raw difference instead is a trap: the difference of two return series has almost
    no autocorrelation in its level while its squares are strongly persistent, which
    yields blocks of a day or two, a bootstrap that throws away volatility clustering,
    and a test that rejects a true null about twice as often as it promises.
    """
    x = np.asarray(x, float)
    if y is None or np.asarray(y, float).std() == 0:
        Z = np.column_stack([x, x * x])
        _, grad = _sr_and_grad(Z.mean(0))
    else:
        y = np.asarray(y, float)
        Z = np.column_stack([x, y, x * x, y * y])
        _, grad = _sr_diff_and_grad(Z.mean(0))
    return (Z - Z.mean(0)) @ grad


def sharpe_diff_and_se(x, y=None, prewhite: bool = True, annualize: bool = True) -> tuple[float, float]:
    """
    Sharpe(x) minus Sharpe(y) and the standard error of that difference.

    `y` may be None or a constant series, which means the comparison is against cash and
    the statistic is simply the Sharpe ratio of `x`. Returned annualized by default.
    """
    x = np.asarray(x, float)
    f = np.sqrt(TRADING_DAYS) if annualize else 1.0
    T = len(x)
    if T < 3:
        return 0.0, 0.0
    if y is None or np.asarray(y, float).std() == 0:
        Z = np.column_stack([x, x * x])
        v = Z.mean(0)
        stat, grad = _sr_and_grad(v)
    else:
        y = np.asarray(y, float)
        Z = np.column_stack([x, y, x * x, y * y])
        v = Z.mean(0)
        stat, grad = _sr_diff_and_grad(v)
    if Z[:, 0].std() == 0:
        return 0.0, 0.0
    psi = hac_covariance(Z, prewhite=prewhite)
    var = float(grad @ psi @ grad) / T
    return stat * f, float(np.sqrt(max(var, 0.0))) * f


# ── Block bootstrap machinery ────────────────────────────────────────────

def politis_white_block_size(x, default: int = 5) -> int:
    """
    Politis and White (2004) automatic block length for the circular block bootstrap,
    from the flat top autocorrelation rule. Falls back to `default` when the series is
    too short or the rule degenerates.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    T = len(x)
    if T < 40 or x.std() == 0:
        return default
    K = max(5, int(np.sqrt(np.log10(T))))
    m_max = min(int(np.ceil(np.sqrt(T))) + K, T - 3)
    var = float((x * x).mean())
    if var <= 0:
        return default
    acf = np.array([1.0] + [float((x[:-k] * x[k:]).mean() / var) for k in range(1, m_max + 1)])
    c = 2.0 * np.sqrt(np.log10(T) / T)
    m = 0
    for k in range(1, len(acf) - K):
        if np.all(np.abs(acf[k:k + K]) < c):
            m = k
            break
    M = min(max(2 * m, 1), len(acf) - 1)

    def lam(t: float) -> float:
        t = abs(t)
        return 1.0 if t <= 0.5 else (2.0 * (1.0 - t) if t <= 1.0 else 0.0)

    js = np.arange(-M, M + 1)
    weights = np.array([lam(j / M) for j in js])
    rho = acf[np.abs(js)]
    G = float(np.sum(weights * np.abs(js) * rho))
    g0 = float(np.sum(weights * rho))
    D = (4.0 / 3.0) * g0 ** 2
    if D <= 0 or G == 0:
        return default
    b = (2.0 * G ** 2 / D) ** (1 / 3) * T ** (1 / 3)
    return int(np.clip(round(b), 1, max(T // 3, 1)))


def circular_block_indices(T: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of one circular block bootstrap resample of length T."""
    block = max(int(block), 1)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
    return idx[:T]


# ── The joint test: studentized bootstrap plus the stepdown ──────────────

def joint_sharpe_tests(strategies: dict, bars: dict, n_boot: int = 1000, block_size: int | None = None,
                       alpha: float = 0.05, seed: int = 0, prewhite: bool = True) -> dict:
    """
    Test every contestant against its own bar, on one shared bootstrap.

    `strategies` and `bars` map a name to a daily return array of the same length; a bar
    of zeros (or any constant) means the contestant is judged against cash. Every
    resample draws one set of dates and applies it to all contestants, so the stepdown
    inherits the dependence between them.

    For each name the result carries the annualized difference, its HAC standard error,
    the studentized interval, the one sided p value for the claim "this beats its bar",
    and the Romano and Wolf adjusted p value that controls the chance of even one false
    claim across the whole board. `clears_bar` means the interval is above zero;
    `clears_bar_fwe` means the claim survives the correction, which is the verdict.
    """
    names = list(strategies)
    if not names:
        return {"contestants": {}, "n_boot": n_boot, "block_size": block_size, "alpha": alpha}
    T = len(next(iter(strategies.values())))
    for n in names:
        if len(strategies[n]) != T or len(bars[n]) != T:
            raise ValueError(f"{n}: strategy and bar series must all share one calendar")
    if block_size is None:
        sizes = [politis_white_block_size(influence_series(strategies[n], bars[n])) for n in names]
        block_size = int(max(np.median(sizes), 1))

    diff = np.zeros(len(names))
    se = np.zeros(len(names))
    live = np.zeros(len(names), dtype=bool)
    for j, n in enumerate(names):
        # A contestant that is its own bar has nothing to test. Decide that from the series
        # rather than from the standard error: the quadratic form behind the error can come
        # back as 1e-17 instead of zero on a different linear algebra library, and dividing
        # a near zero difference by a near zero error would feed noise into the stepdown.
        if np.allclose(strategies[n], bars[n], rtol=0, atol=1e-15):
            continue
        diff[j], se[j] = sharpe_diff_and_se(strategies[n], bars[n], prewhite=prewhite)
        live[j] = se[j] > 0
    se = np.where(live, se, 0.0)
    t = np.where(live, diff / np.where(live, se, 1.0), 0.0)

    rng = np.random.default_rng(seed)
    X = {n: np.asarray(strategies[n], float) for n in names}
    B = {n: np.asarray(bars[n], float) for n in names}
    boot = np.zeros((n_boot, len(names)))
    for b in range(n_boot):
        idx = circular_block_indices(T, block_size, rng)
        for j, n in enumerate(names):
            if not live[j]:
                continue
            d_b, s_b = sharpe_diff_and_se(X[n][idx], B[n][idx], prewhite=prewhite)
            boot[b, j] = (d_b - diff[j]) / s_b if s_b > 0 else 0.0

    out = {}
    for j, n in enumerate(names):
        col = boot[:, j]
        if live[j]:
            lo = diff[j] - float(np.quantile(col, 1 - alpha / 2)) * se[j]
            hi = diff[j] - float(np.quantile(col, alpha / 2)) * se[j]
            p_one = float((np.sum(col >= t[j]) + 1) / (n_boot + 1))
            p_two = float((np.sum(np.abs(col) >= abs(t[j])) + 1) / (n_boot + 1))
        else:
            lo = hi = diff[j]
            p_one = p_two = 1.0
        out[n] = {"net_edge": float(diff[j]), "se": float(se[j]), "t_stat": float(t[j]),
                  "ci_low": float(lo), "ci_high": float(hi), "p_value": p_one, "p_value_two_sided": p_two,
                  "clears_bar": bool(lo > 0)}

    # Romano and Wolf stepdown: adjusted p values, in descending order of the statistic
    order = list(np.argsort(-t))
    remaining = list(order)
    running = 0.0
    for j in order:
        col_max = boot[:, remaining].max(axis=1) if remaining else np.zeros(n_boot)
        p_adj = float((np.sum(col_max >= t[j]) + 1) / (n_boot + 1))
        running = max(running, p_adj)          # adjusted p values must not decrease down the order
        out[names[j]]["p_value_fwe"] = running
        out[names[j]]["clears_bar_fwe"] = bool(running <= alpha and live[j] and t[j] > 0)
        remaining.remove(j)
    return {"contestants": out, "n_boot": n_boot, "block_size": int(block_size), "alpha": alpha,
            "n_contestants": len(names)}


def ledoit_wolf_test(x, y=None, n_boot: int = 1000, block_size: int | None = None,
                     alpha: float = 0.05, seed: int = 0) -> dict:
    """The single pair case of `joint_sharpe_tests`: one strategy against one bar."""
    y = np.zeros(len(np.asarray(x))) if y is None else y
    res = joint_sharpe_tests({"x": x}, {"x": y}, n_boot=n_boot, block_size=block_size,
                             alpha=alpha, seed=seed)
    out = res["contestants"]["x"]
    out["block_size"] = res["block_size"]
    out["n_boot"] = n_boot
    return out


# ── Overfitting: how much of this was the search? ────────────────────────

def pbo_cscv(returns: np.ndarray, n_splits: int = 16) -> dict:
    """
    Probability of backtest overfitting by combinatorially symmetric cross validation.

    `returns` is a matrix of daily returns, one column per variant the submitter tried.
    The record is cut into `n_splits` equal blocks; for every way of splitting those
    blocks into two halves, the variant with the best Sharpe on one half is looked up on
    the other, and its relative rank is turned into a logit. The probability that the
    logit is not positive is the answer: pure noise gives one half, a genuine edge gives
    something near zero.
    """
    R = np.asarray(returns, float)
    if R.ndim != 2 or R.shape[1] < 2:
        raise ValueError("pass a matrix with at least two variants in columns")
    S = int(n_splits)
    if S % 2 or S < 4:
        raise ValueError("n_splits must be even and at least 4")
    T, N = R.shape
    size = T // S
    if size < 5:
        raise ValueError(f"{T} observations cannot be cut into {S} usable blocks")
    blocks = R[:size * S].reshape(S, size, N)
    s1 = blocks.sum(axis=1)                    # S x N
    s2 = (blocks ** 2).sum(axis=1)             # S x N

    def sharpe(idx: tuple) -> np.ndarray:
        cnt = size * len(idx)
        m = s1[list(idx)].sum(0) / cnt
        v = s2[list(idx)].sum(0) / cnt - m * m
        return np.divide(m, np.sqrt(np.maximum(v, _TINY)))

    logits, ranks = [], []
    all_blocks = set(range(S))
    for train in itertools.combinations(range(S), S // 2):
        test = tuple(sorted(all_blocks - set(train)))
        best = int(np.argmax(sharpe(train)))
        r = float(rankdata(sharpe(test))[best])
        w = r / (N + 1.0)
        ranks.append(r)
        logits.append(float(np.log(w / (1.0 - w))))
    logits = np.asarray(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_splits": S, "n_variants": int(N),
            "n_combinations": int(len(logits)), "median_logit": float(np.median(logits)),
            "median_rank": float(np.median(ranks)), "logits": logits}


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """
    The Sharpe ratio the best of `n_trials` independent worthless strategies is expected
    to show, given the variance of Sharpe ratios across those trials. Bailey and Lopez de
    Prado (2014). Annualized in, annualized out.
    """
    n = max(int(n_trials), 2)
    z1 = norm.ppf(1 - 1.0 / n)
    z2 = norm.ppf(1 - 1.0 / (n * np.e))
    return float(np.sqrt(max(sr_variance, 0.0)) * ((1 - EULER) * z1 + EULER * z2))


def probabilistic_sharpe(daily_returns, benchmark_sharpe_annual: float = 0.0) -> float:
    """
    Probability that the true Sharpe exceeds a benchmark, accounting for sample length,
    skew and kurtosis. Bailey and Lopez de Prado (2012).
    """
    r = np.asarray(daily_returns, float)
    n = len(r)
    if n < 3 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    sr_b = benchmark_sharpe_annual / np.sqrt(TRADING_DAYS)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def deflated_sharpe(daily_returns, n_trials: int, sr_variance: float) -> dict:
    """
    The probabilistic Sharpe ratio measured against what the search itself would have
    produced. `sr_variance` is the variance of the annualized Sharpe ratios across the
    trials that were run.
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return {"deflated_sharpe": probabilistic_sharpe(daily_returns, sr0),
            "expected_max_sharpe": sr0, "n_trials": int(n_trials)}
