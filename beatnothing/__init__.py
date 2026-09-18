"""
beatnothing: can your model beat doing nothing, after costs, without knowing the future?

A small, reproducible benchmark harness for daily equity signals. Every contestant is
scored through one identical engine against one bar: an equal weight position in the
same point in time universe, held with no skill at all. The score is the Net Edge,
the net Sharpe of the strategy minus the net Sharpe of doing nothing, with a paired
bootstrap confidence interval so that noise cannot masquerade as alpha.
"""
from .canary import hindsight_universe, off_by_one, peek, truncation_test
from .engine import Backtest, COST_BPS, TRADING_DAYS
from .score import bootstrap_sharpe_diff, cost_grid, net_edge, probabilistic_sharpe, sharpe
from .universe import Membership, coverage_report, start_end_table

__version__ = "0.2.2"
__all__ = ["Backtest", "COST_BPS", "TRADING_DAYS", "net_edge", "sharpe", "bootstrap_sharpe_diff",
           "probabilistic_sharpe", "cost_grid", "Membership", "coverage_report", "start_end_table",
           "peek", "off_by_one", "hindsight_universe", "truncation_test", "__version__"]
