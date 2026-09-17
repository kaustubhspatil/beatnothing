"""
One engine for every contestant.

A contestant hands in either predictions (dates x tickers, any real number) or
weights (dates x tickers, each in [0, 1], rows summing to at most 1). Predictions are
turned into the fixed rule "equal weight long every name with a positive prediction,
cash otherwise". Weights are used as given. Either way the engine charges a cost on
every unit of turnover, including the first day, and reports net numbers only.

Deliberately simple: trades at the close, long or flat, no leverage, no shorting,
no market impact beyond the cost parameter. Simple enough that nobody can hide
anything in it, which is the point of a benchmark engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS = 10.0
TRADING_DAYS = 252
INITIAL_CAPITAL = 1_000_000.0


def weights_from_predictions(predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Equal weight across positive predictions where a realised return exists."""
    signal = (predictions > 0) & actual.notna()
    n_long = signal.sum(axis=1)
    return signal.div(n_long.replace(0, np.nan), axis=0).fillna(0.0)


class Backtest:
    """
    Args:
        actual:       realised next day simple returns (dates x tickers), aligned so
                      that row t is the return earned from the close of t to t+1.
        predictions:  contestant predictions (dates x tickers), or None if `weights` given.
        weights:      contestant weights (dates x tickers) in [0, 1], row sums <= 1.
        cost_bps:     cost per unit of turnover, in basis points.
    """

    def __init__(self, actual: pd.DataFrame, predictions: pd.DataFrame | None = None,
                 weights: pd.DataFrame | None = None, cost_bps: float = COST_BPS,
                 capital: float = INITIAL_CAPITAL):
        if (predictions is None) == (weights is None):
            raise ValueError("pass exactly one of predictions or weights")
        if weights is None:
            predictions, actual = predictions.align(actual, join="inner")
            weights = weights_from_predictions(predictions, actual)
        else:
            weights, actual = weights.align(actual, join="inner")
            weights = weights.fillna(0.0)
            if (weights < -1e-12).any().any():
                raise ValueError("weights must be non negative (no shorting)")
            if (weights.sum(axis=1) > 1.0 + 1e-9).any():
                raise ValueError("weights must sum to at most 1 on every day (no leverage)")
        self.actual, self.weights = actual, weights
        self.cost = cost_bps / 10_000
        self.capital = capital
        gross = (weights * actual.fillna(0.0)).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1)
        turnover.iloc[0] = weights.iloc[0].abs().sum()
        self.turnover = turnover
        self.daily_costs = turnover * self.cost
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
            "total_return": float(total),
            "annualized_return": float((1 + total) ** (1 / years) - 1) if years > 0 else float("nan"),
            "annualized_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
            "net_sharpe": sharpe(r),
            "max_drawdown": max_drawdown(self.equity),
            "avg_exposure": float(self.weights.sum(axis=1).mean()),
            "avg_positions": float((self.weights > 0).sum(axis=1).mean()),
            "annual_turnover": float(self.turnover.sum() / years) if years > 0 else float("nan"),
            "total_costs": float((self.daily_costs * self.equity.shift(1).fillna(self.capital)).sum()),
            "dollar_pnl": float(self.equity.iloc[-1] - self.capital),
        }

    def monthly_returns(self) -> pd.Series:
        return (1 + self.daily_returns).resample("ME").prod() - 1
