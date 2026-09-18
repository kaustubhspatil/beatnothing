"""
One engine for every contestant.

A contestant hands in either predictions (dates x tickers, any real number) or
weights (dates x tickers). Predictions become positions by one of three fixed rules:

    long_flat   equal weight long every name with a positive prediction, cash otherwise
                (the default, and the rule of season one)
    long_top    equal weight long the top `quantile` of names by prediction each day
    long_short  long the top `quantile` and short the bottom `quantile`, equal weight
                within each side, half the capital on each side, so the book is dollar
                neutral and its gross exposure is one

Weights are used as given. Long only unless `allow_short`, gross exposure never above
one, no leverage. Every unit of turnover pays `cost_bps`, day one included, and every
dollar held short pays `borrow_bps_annual` a year. Net numbers only.

Deliberately simple: trades at the close, no market impact beyond the cost parameter,
no short rebate. Simple enough that nobody can hide anything in it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS = 10.0
BORROW_BPS_ANNUAL = 50.0
TRADING_DAYS = 252
INITIAL_CAPITAL = 1_000_000.0
RULES = ("long_flat", "long_top", "long_short")


def _equal_weight(mask: pd.DataFrame) -> pd.DataFrame:
    n = mask.sum(axis=1)
    return mask.div(n.replace(0, np.nan), axis=0).fillna(0.0)


def weights_from_predictions(predictions: pd.DataFrame, actual: pd.DataFrame, rule: str = "long_flat",
                             quantile: float = 0.1) -> pd.DataFrame:
    """Turn predictions into weights under one of the fixed rules, on names with a realised return."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    valid = actual.notna()
    if rule == "long_flat":
        return _equal_weight((predictions > 0) & valid)
    ranks = predictions.where(valid).rank(axis=1, pct=True)
    longs = (ranks > 1 - quantile) & valid
    if rule == "long_top":
        return _equal_weight(longs)
    shorts = (ranks <= quantile) & valid
    return 0.5 * _equal_weight(longs) - 0.5 * _equal_weight(shorts)


class Backtest:
    """
    Args:
        actual:       realised next day simple returns (dates x tickers), aligned so
                      that row t is the return earned from the close of t to t+1.
        predictions:  contestant predictions (dates x tickers), or None if `weights` given.
        weights:      contestant weights (dates x tickers); gross exposure at most 1.
        rule:         how predictions become weights (see module docstring).
        quantile:     fraction of names on each side for long_top and long_short.
        cost_bps:     cost per unit of turnover, in basis points.
        borrow_bps_annual: annual cost of every dollar held short.
        allow_short:  accept negative weights (set automatically for long_short).
    """

    def __init__(self, actual: pd.DataFrame, predictions: pd.DataFrame | None = None,
                 weights: pd.DataFrame | None = None, rule: str = "long_flat", quantile: float = 0.1,
                 cost_bps: float = COST_BPS, borrow_bps_annual: float = BORROW_BPS_ANNUAL,
                 allow_short: bool = False, capital: float = INITIAL_CAPITAL):
        if (predictions is None) == (weights is None):
            raise ValueError("pass exactly one of predictions or weights")
        allow_short = allow_short or rule == "long_short"
        if weights is None:
            predictions, actual = predictions.align(actual, join="inner")
            weights = weights_from_predictions(predictions, actual, rule, quantile)
        else:
            weights, actual = weights.align(actual, join="inner")
            # no position in a name on a day it has no realised return (not a member, not listed)
            weights = weights.fillna(0.0).where(actual.notna(), 0.0)
            if not allow_short and (weights < -1e-12).any().any():
                raise ValueError("weights must be non negative unless allow_short=True")
            if (weights.abs().sum(axis=1) > 1.0 + 1e-9).any():
                raise ValueError("gross exposure must be at most 1 on every day (no leverage)")
        self.actual, self.weights, self.rule = actual, weights, rule
        self.cost = cost_bps / 10_000
        self.borrow = borrow_bps_annual / 10_000 / TRADING_DAYS
        self.capital = capital
        gross = (weights * actual.fillna(0.0)).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1)
        turnover.iloc[0] = weights.iloc[0].abs().sum()
        short_exposure = weights.clip(upper=0).abs().sum(axis=1)
        self.turnover = turnover
        self.daily_costs = turnover * self.cost + short_exposure * self.borrow
        self.daily_borrow = short_exposure * self.borrow
        self.daily_returns = gross - self.daily_costs
        self.equity = capital * (1 + self.daily_returns).cumprod()
        self.drawdown = self.equity / self.equity.cummax() - 1.0

    def stats(self) -> dict:
        from .score import sharpe, max_drawdown
        r = self.daily_returns
        years = len(r) / TRADING_DAYS
        total = self.equity.iloc[-1] / self.capital - 1
        return {
            "days": int(len(r)),
            "rule": self.rule,
            "total_return": float(total),
            "annualized_return": float((1 + total) ** (1 / years) - 1) if years > 0 else float("nan"),
            "annualized_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
            "net_sharpe": sharpe(r),
            "max_drawdown": max_drawdown(self.equity),
            "avg_exposure": float(self.weights.sum(axis=1).mean()),
            "avg_gross_exposure": float(self.weights.abs().sum(axis=1).mean()),
            "avg_short_exposure": float(self.weights.clip(upper=0).abs().sum(axis=1).mean()),
            "avg_positions": float((self.weights != 0).sum(axis=1).mean()),
            "annual_turnover": float(self.turnover.sum() / years) if years > 0 else float("nan"),
            "total_costs": float((self.daily_costs * self.equity.shift(1).fillna(self.capital)).sum()),
            "total_borrow": float((self.daily_borrow * self.equity.shift(1).fillna(self.capital)).sum()),
            "dollar_pnl": float(self.equity.iloc[-1] - self.capital),
        }

    def monthly_returns(self) -> pd.Series:
        return (1 + self.daily_returns).resample("ME").prod() - 1
