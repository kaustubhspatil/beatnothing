# beatnothing

**Can your model beat doing nothing, after costs, without knowing the future?**

A small, reproducible benchmark for daily equity signals. One engine, one bar, one
score. The bar is the dumbest possible strategy: hold every stock in the universe at
equal weight and never think again. The score is the **Net Edge**, your net Sharpe
minus that bar's net Sharpe on the same days, with a paired bootstrap interval so
that a lucky year cannot pose as skill.

<p align="center">
  <img src="leaderboard/figures/net_edge.png" width="880" alt="Net Edge with 95% intervals for every contestant on the sealed window and on 2026 to date">
</p>

Nine contestants so far, from an ordinary linear model to a 2026 financial foundation
model, and **not one of them clears the bar**, on the sealed 2022 to 2025 window or on
the 176 trading days of 2026 that none of the frozen models had ever seen. The full
table with intervals, turnover, drawdowns and dollars is in
[`leaderboard/LEADERBOARD.md`](leaderboard/LEADERBOARD.md).

## Why this exists

Three things go wrong in almost every published machine learning trading result, and
the 2026 literature is now saying so out loud: leakage introduced by the evaluation
protocol rather than the model, costs that are reported gross or not at all, and a
universe chosen with hindsight. Recent benchmarks address one of these each, and most
of them run on CRSP data that nobody outside a university can rerun. This one runs on
a laptop with free data and addresses all three, then adds the thing that actually
settles arguments: a forward track where frozen signals meet days that did not exist
when they were registered.

<table>
<tr><th>Failure</th><th>What the harness does about it</th></tr>
<tr><td>Costs</td><td>Every contestant pays 10 bps on every unit of turnover, day one included. Gross and 20 bps numbers sit beside the net number so you can see who only wins for free.</td></tr>
<tr><td>Leakage</td><td>Shipped leak canaries (a model that peeks at t+1, an off by one feature shift, a hindsight universe) must score absurdly well in your pipeline, or your pipeline is broken. A truncation test proves features are trailing only by deleting the future and recomputing.</td></tr>
<tr><td>Hindsight universe</td><td>Point in time S&P 500 membership from 1996 to 2026 and a coverage report that states, in numbers, what a free price source can no longer supply. For the sealed window: 94 of the 505 members on 2021 12 31 left the index, and 47 of those, including both 2023 bank failures, have no prices on Yahoo Finance any more.</td></tr>
<tr><td>Noise posing as alpha</td><td>Net Edge carries a 95% stationary block bootstrap interval on the paired daily difference. A contestant clears the bar only when the whole interval is above zero. On 176 days the interval is wide, and the leaderboard says so instead of ranking noise.</td></tr>
<tr><td>Frozen means frozen</td><td>Every submission records the sha256 of its model files and a registration date. A monthly GitHub Action pulls new prices and rescores everything on the days that arrived after registration.</td></tr>
</table>

## What the first leaderboard says

Sealed window 2022 to 2025, 48 large cap US stocks, net of 10 bps:

* **Costs reorder the field.** A linear model that trades 159 times its capital a year has a gross Sharpe of 0.51 and a net Sharpe of minus 0.30. A 1D CNN goes from 1.08 gross to minus 0.05 net. The order of contestants at 0 bps is not the order at 10 bps.
* **The MSE optimum is nearly "always long".** LightGBM, early stopped honestly on validation loss, stops after three trees and holds 47 of 48 names. Squared error on daily returns rewards predicting the drift, and the drift is the bar.
* **A cost aware objective fixes turnover, not alpha.** A network trained end to end on net Sharpe with a 10 bps turnover term trades 3 times a year instead of 12 and lands within 0.04 of the bar across three seeds. Remove the cost term and the same architecture trades four times as much and loses 0.7 of Sharpe.
* **The foundation model inverts too.** Kronos small, used zero shot, has a gross Sharpe of 0.52 and a net Sharpe of minus 0.79, with a 53% drawdown, because it flips positions 241 times a year. A hundred times the parameters of the feedforward network, pretrained on 12 billion bars, and the same cost arithmetic as a linear regression.
* **Nothing clears the bar.** The best learned contestant, the feedforward network, sits at a Net Edge of minus 0.03 with an interval of [minus 0.06, minus 0.01]: measurably, slightly worse than doing nothing. The cost aware network matches the bar's Sharpe with half the exposure and half the drawdown, which makes it the bar delevered, not alpha.

2026 to date (176 days, models frozen in August 2026 on data ending December 2025):
the ranking reproduces. The incumbent linear signal is flat after costs, the
feedforward network trails the bar by 0.08, the cost aware network matches it, and the
high turnover models fall behind. All intervals include zero: eight months cannot
separate anything from anything, which is itself the finding.

<p align="center">
  <img src="leaderboard/figures/cost_inversion.png" width="880" alt="Gross versus net Sharpe for every contestant">
</p>

## Contestants

<table>
<tr><th>Contestant</th><th>What it is</th><th>Trained</th></tr>
<tr><td>always long</td><td>the bar</td><td>never</td></tr>
<tr><td>linear incumbent</td><td>OLS on 17 trailing technical features</td><td>2005 to 2018</td></tr>
<tr><td>feedforward NN</td><td>128 64 32, MSE, early stopped</td><td>2005 to 2018</td></tr>
<tr><td>LSTM 60d</td><td>two layers over 60 day windows</td><td>2005 to 2018</td></tr>
<tr><td>1D CNN 60d</td><td>three conv blocks over 60 day windows</td><td>2005 to 2018</td></tr>
<tr><td>LightGBM (MSE)</td><td>gradient boosted trees, early stopped on validation</td><td>2005 to 2018</td></tr>
<tr><td>Kronos small, zero shot</td><td>24.7M parameter financial foundation model (AAAI 2026), 400 bar context, no training on this data at all</td><td>pretrained by its authors</td></tr>
<tr><td>cost aware net, 10 bps term</td><td>weights out, trained on net Sharpe over 126 day windows, mean of 3 seeds</td><td>2005 to 2018</td></tr>
<tr><td>cost aware net, no cost term</td><td>the same with the turnover penalty removed (ablation)</td><td>2005 to 2018</td></tr>
</table>

The first five come frozen from the
[deep learning equity signal capstone](https://github.com/kaustubhspatil/sp500-quantitative-deep-learning),
where their training is documented notebook by notebook. Their sha256 hashes are in
each `meta.json`.

## Use it

```bash
pip install -e ".[data,figures,dev]"
pytest -q                                   # engine invariants, canaries fire, scoring math
python scripts/download_data.py             # fresh prices, features, realised returns, manifest
python -m beatnothing.leaderboard           # score every submission on every window
python scripts/make_figures.py
```

Score your own signal in four lines:

```python
from beatnothing import Backtest, net_edge
from beatnothing.leaderboard import load_actual
actual = load_actual("data/actual_returns.parquet")          # dates x tickers, next day returns
mine = Backtest(actual, predictions=my_predictions)           # or weights=my_weights
bar = Backtest(actual, predictions=actual * 0 + 1)            # always long, same days, same costs
print(net_edge(mine.daily_returns, bar.daily_returns))       # net edge, interval, clears_bar
```

To enter the leaderboard, read [`contestants/README.md`](contestants/README.md) and open
a pull request with a `submissions/<name>/` folder.

## Honest limits

* **The universe is a survivor universe.** The 48 names were chosen in August 2026 from
  the then current index. Ten of them joined the index after 2005. This flatters every
  contestant and the bar equally, so Net Edge survives it, but absolute numbers do not.
  The point in time module and the coverage report exist so the next version can run on
  the true membership; that needs a price source that still carries delisted names.
* **Costs are a flat 10 bps.** No market impact, no borrow, no slippage that grows with
  size. It is the friction a small book pays, which is the honest scale of a laptop
  benchmark.
* **Long or flat only.** No shorting, no leverage. A long short signal can be evaluated
  by submitting weights for the long book only, or by extending the engine, which is a
  welcome pull request.
* **Eight months is not evidence.** The forward track exists to accumulate it. Check
  back in a year.

## Credits and sources

Point in time membership: [fja05680/sp500](https://github.com/fja05680/sp500) (MIT).
Kronos: Shi et al., *Kronos: A Foundation Model for the Language of Financial Markets*,
AAAI 2026, [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) (MIT). The
stationary bootstrap is Politis and Romano (1994); the probabilistic Sharpe ratio is
Bailey and Lopez de Prado (2012); the Sharpe standard error is Lo (2002). Prices from
Yahoo Finance through yfinance; the snapshot hash is in `data/MANIFEST.json`.

MIT licensed. Built by Kaustubh Patil.
