# beatnothing

[![tests](https://github.com/kaustubhspatil/beatnothing/actions/workflows/tests.yml/badge.svg)](https://github.com/kaustubhspatil/beatnothing/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/beatnothing.svg?color=2a78d6)](https://pypi.org/project/beatnothing/)
[![python](https://img.shields.io/pypi/pyversions/beatnothing.svg)](https://pypi.org/project/beatnothing/)
[![license](https://img.shields.io/github/license/kaustubhspatil/beatnothing)](https://github.com/kaustubhspatil/beatnothing/blob/master/LICENSE)

A benchmark for daily equity signals: **can your model beat doing nothing, after costs,
without seeing the future?**

"Doing nothing" means holding every stock at equal weight. Every contestant gets the same
prices, pays the same fee on every trade, and is scored on the gap to that bar, with a
confidence interval. That gap is the **Net Edge**.

25 contestants so far: neural nets (FFNN, LSTM, 1D CNN, a ranking network), gradient
boosted trees, classic factors, and the Kronos financial foundation model used zero-shot.
**None of them beats the bar.**

<p align="center">
  <img src="https://raw.githubusercontent.com/kaustubhspatil/beatnothing/master/leaderboard/pit/figures/net_edge.png" width="900" alt="Net Edge with 95% intervals for all 25 contestants on the point in time universe, 2022 to 2025 and 2026 to date">
</p>
<p align="center"><em>All contestants, both windows. Green would mean the whole interval is above zero.</em></p>

```bash
pip install beatnothing
```

## Rules

| Rule | Details |
|---|---|
| One engine | Predictions become positions by one of three fixed rules: long every positive name (default), long the top decile, or long top / short bottom decile (dollar neutral, 50 bps/yr borrow). Gross exposure ≤ 1. |
| Two bars | **Universe bar**: equal weight, same names, same days, same costs. **Investable bar**: RSP (equal-weight S&P 500 ETF), which holds every member including the ones that later died. |
| Costs | 10 bps per unit of turnover, from day one. Gross and 20 bps numbers are shown too. |
| Score | Net Edge = net Sharpe minus the bar's net Sharpe, with a 95% paired block bootstrap interval. You only beat the bar if the whole interval is above zero. |
| Frozen entries | Each submission stores a sha256 of its signal and a registration date. A monthly job rescores everything on new days. |

## Season one: 48 survivor stocks

Sealed window 2022–2025, net of 10 bps. Full table in
[`leaderboard/LEADERBOARD.md`](https://github.com/kaustubhspatil/beatnothing/blob/master/leaderboard/LEADERBOARD.md).

| Contestant | Net Edge | 95% interval | Edge vs RSP | Net Sharpe | Gross Sharpe | Turnover/yr |
|---|---|---|---|---|---|---|
| Always long (bar) | 0.00 | | +0.44 [+0.19, +0.76] | +0.88 | +0.88 | 0.3x |
| Feedforward network | −0.03 | [−0.06, −0.01] | +0.41 [+0.17, +0.72] | +0.85 | +0.88 | 4.3x |
| Cost-aware network, 10 bps term | −0.04 | [−0.10, +0.01] | +0.40 [+0.15, +0.72] | +0.83 | +0.87 | 2.9x |
| LightGBM, MSE | −0.14 | [−0.45, +0.09] | +0.30 [−0.13, +0.72] | +0.74 | +0.79 | 7.2x |
| LSTM, 60 day | −0.53 | [−1.00, −0.12] | −0.08 [−0.61, +0.44] | +0.35 | +0.62 | 43x |
| Cost-aware network, no cost term | −0.70 | [−1.27, −0.09] | −0.26 [−0.89, +0.35] | +0.18 | +0.53 | 11.5x |
| 1D CNN, 60 day | −0.93 | [−1.55, −0.29] | −0.49 [−1.17, +0.16] | −0.05 | +1.08 | 265x |
| Linear regression (incumbent) | −1.17 | [−1.82, −0.62] | −0.73 [−1.42, −0.10] | −0.30 | +0.51 | 159x |
| Kronos small, zero-shot | −1.67 | [−2.09, −1.26] | −1.23 [−1.71, −0.79] | −0.79 | +0.52 | 241x |

A few models "beat" RSP here, but so does the bar itself by the same amount. That's the
stock list being picked with hindsight, not skill.

<p align="center">
  <img src="https://raw.githubusercontent.com/kaustubhspatil/beatnothing/master/leaderboard/figures/cost_inversion.png" width="900" alt="Gross versus net Sharpe for each season one contestant">
</p>

What season one showed:

- **Costs change the ranking.** The 1D CNN has the best gross Sharpe (1.08) and a negative
  net Sharpe, because it turns over 265x a year.
- **MSE models end up long everything.** LightGBM early-stops after three trees and holds
  47 of 48 names. Predicting the average daily return is just the bar again.
- **Kronos has the same problem as linear regression.** Gross +0.52, net −0.79, 53%
  drawdown, 241x turnover.
- **A cost-aware objective fixes turnover, not alpha.** Trained on net Sharpe with a
  10 bps turnover term it trades 3x a year and lands on the bar. Without the term it loses
  0.7 Sharpe to fees.

## Season two: point-in-time universe

<p align="center">
  <img src="https://raw.githubusercontent.com/kaustubhspatil/beatnothing/master/leaderboard/pit/figures/two_bars.png" width="900" alt="Point in time universe bar against RSP and SPY, 2022 to 2026">
</p>

Every S&P 500 member on every day since 2022, 615 names including the ones that failed or
got acquired, built from free data: Yahoo for 564 names, Yahoo under a new ticker for 16
renames, and Tiingo's free tier for 39 delisted names. Coverage is 99.5% of member days.

With this universe the bar matches RSP (Sharpe 0.47 vs 0.44, edge +0.03 [−0.01, +0.06]).
On the survivor universe that gap was +0.44, so the hindsight premium is gone.

| Contestant (PIT track) | Net Edge | 95% interval | Net Sharpe | Gross Sharpe | Max DD | Turnover/yr |
|---|---|---|---|---|---|---|
| Momentum 12-1, long/short vs cash | +0.21 | [−0.79, +1.20] | +0.21 | +0.27 | −17% | 7.0x |
| Momentum 12-1, top decile | +0.16 | [−0.48, +0.80] | +0.63 | +0.66 | −24% | 7.2x |
| Low volatility, top decile | +0.02 | [−0.74, +0.76] | +0.49 | +0.55 | −14% | 7.7x |
| Cost-aware network, 10 bps term | +0.00 | [−0.06, +0.07] | +0.47 | +0.50 | −12% | 3.2x |
| Always long (bar) | 0.00 | | +0.47 | +0.47 | −21% | 0.4x |
| Feedforward network | −0.02 | [−0.03, −0.01] | +0.45 | +0.47 | −21% | 4.3x |
| LightGBM, MSE | −0.11 | [−0.45, +0.12] | +0.36 | +0.40 | −22% | 7.4x |
| LSTM, 60 day | −0.25 | [−0.53, +0.00] | +0.21 | +0.52 | −22% | 51x |
| Low volatility, long/short vs cash | −0.27 | [−1.25, +0.69] | −0.27 | −0.22 | −28% | 6.7x |
| Cost-aware network, no cost term | −0.33 | [−0.89, +0.24] | +0.14 | +0.49 | −6% | 12x |
| Kronos small, zero-shot, weekly | −0.35 | [−0.55, −0.16] | +0.12 | +0.38 | −25% | 50x |
| Reversal 1m, top decile | −0.38 | [−0.81, +0.03] | +0.09 | +0.17 | −27% | 21x |
| Reversal 1m, long/short vs cash | −0.47 | [−1.38, +0.49] | −0.47 | −0.25 | −21% | 21x |
| Linear regression (incumbent) | −0.80 | [−1.09, −0.52] | −0.34 | +0.47 | −38% | 151x |
| 1D CNN, 60 day | −1.03 | [−1.38, −0.70] | −0.56 | +0.53 | −52% | 235x |
| Kronos small, zero-shot, daily | −1.31 | [−1.56, −1.06] | −0.84 | +0.35 | −55% | 225x |

Five contestants have a positive point estimate, and every one of their intervals includes
zero. After the multiple-testing correction the lowest adjusted p value is 0.996.

**Kronos daily vs weekly.** Gross Sharpe is about the same (0.35 vs 0.38), so the model
knows no more at a daily horizon. Trading daily costs 1.19 Sharpe, weekly costs 0.26.
Weekly is still well below the bar.

**Classic factors.** Momentum is the only factor with a positive edge, and it depends on
cheap borrow. Raising the borrow cost removes it:

| Dollar-neutral net Sharpe | 50 bps | 200 bps | 500 bps | 1000 bps |
|---|---|---|---|---|
| Momentum 12-1 | +0.21 | +0.14 | +0.01 | −0.21 |
| Low volatility 63d | −0.27 | −0.33 | −0.44 | −0.62 |
| Reversal 1m | −0.47 | −0.55 | −0.70 | −0.96 |

The Tiingo data is personal-use only, so the repo ships realised returns for the scored
window instead of raw prices. `scripts/build_pit_universe.py` rebuilds the panel with your
own free token.

## Season three: trained on the point-in-time universe

Season two contestants were trained on 48 survivors. Season three retrains on the full
universe back to 2013 (797 names, 97.4% coverage): fit to 2018, tune on 2019–2021, score
on 2022–2025. It stops at 2013 because the free data doesn't include the 2008 failures
(Lehman, Bear Stearns, WaMu and others).

The new contestant is a ranking network that ranks stocks against each other each day and
trades long/short, so it can't just ride the market.

| Ranking network | IC | Gross Sharpe | Net Sharpe | Turnover | Cost |
|---|---|---|---|---|---|
| daily | +0.0154 | +0.72 | −1.50 | 222x | 2.22 Sharpe |
| weekly | +0.0007 | +0.32 | −0.37 | 67x | 0.69 |
| monthly | −0.0039 | −0.10 | −0.30 | 20x | 0.20 |

It has the most real information of any contestant at a one-day horizon, but that
information is gone within a week, and trading it daily costs three times what it earns.

## How results are tested

- **Intervals.** Studentized circular block bootstrap (Ledoit & Wolf 2008) with a HAC
  standard error via the delta method. Block length from Politis & White, applied to the
  statistic's influence function.
- **Multiple testing.** A Romano-Wolf stepdown across the whole board. The verdict uses
  **adjusted p**, not raw p.
- **Overfitting.** `beatnothing.stats` includes the probability of backtest overfitting
  (CSCV) and the deflated Sharpe ratio for contestants with many variants.

<p align="center">
  <img src="https://raw.githubusercontent.com/kaustubhspatil/beatnothing/master/leaderboard/figures/stats_validation.png" width="900" alt="Measured size, power, familywise error and overfitting probability">
</p>

`scripts/validate_stats.py` checks all of this on 250 simulated markets with fat tails and
volatility clustering:

| Test | Result |
|---|---|
| No real edge. How often is one claimed? | studentized **4.8%** (target 5%), percentile 2.4% |
| Real edge of 1/3 Sharpe. How often is it found? | studentized **42.4%**, percentile 38.8% |
| 15 useless contestants. How often does one "win"? | separately **61%**, after stepdown **1%** |
| 12 noise variants. Overfitting probability of the best? | **0.66** (0.5 = pure noise); with one real variant **0.00** |

Numbers in [`leaderboard/stats_validation.json`](https://github.com/kaustubhspatil/beatnothing/blob/master/leaderboard/stats_validation.json).

## Leakage checks

The package includes contestants that cheat on purpose. Run them through your own
pipeline first. If they don't score absurdly well, something is wrong.

```python
from beatnothing import Backtest, peek, off_by_one, hindsight_universe
Backtest(actual, predictions=peek(actual)).stats()["net_sharpe"]      # uses tomorrow's return
Backtest(actual, predictions=hindsight_universe(actual, 10)).stats()  # picks the 10 best names in advance
```

`truncation_test` deletes future data, recomputes features, and checks rows are identical.
On the point-in-time panel it compared 46,605 rows across 60 names and found zero
difference on all 17 features
([`data/pit/truncation_test.json`](https://github.com/kaustubhspatil/beatnothing/blob/master/data/pit/truncation_test.json)).

Delisted stocks stay in until their removal date. The bar holds Silicon Valley Bank through
its −60.4% day and First Republic through its −90.5% day. Membership uses effective dates,
not announcement dates.

## Survivorship

<p align="center">
  <img src="https://raw.githubusercontent.com/kaustubhspatil/beatnothing/master/leaderboard/figures/survivorship_gap.png" width="900" alt="Members on 31 December 2021, how many left the index, and how many no longer have prices">
</p>

The package ships point-in-time S&P 500 membership from 1996 to 2026
(`beatnothing members 2008-09-15` shows who was in the index that day). Of the 505 members
at the end of 2021, 94 left the index by 2025 and 47 of those no longer have prices on
Yahoo.

## Submit a contestant

No API key or download needed, the returns panel is in the repo:

```bash
git clone https://github.com/kaustubhspatil/beatnothing && cd beatnothing
pip install -e ".[dev]"
python scripts/starter_submission.py --name my_first_try --author "your handle"
beatnothing validate submissions_pit/my_first_try
```

The starter builds a 5-day reversal signal (IC +0.0142, net Sharpe −1.14, 201x turnover).
Swap `signal_from_returns` for your own idea and open a PR with the folder. See
[`contestants/README.md`](https://github.com/kaustubhspatil/beatnothing/blob/master/contestants/README.md).

Every PR is checked automatically: structure, exposure rules, metadata, and an IC-based
leakage check (real daily signals have an IC of about 0.02–0.05). You don't have to share
your model, just the signal file and metadata.

## Library and CLI

```python
from beatnothing import Backtest, net_edge
from beatnothing.leaderboard import load_actual, bar_returns
actual = load_actual("data/actual_returns.parquet")
mine = Backtest(actual, predictions=my_predictions)      # or weights=my_weights
print(net_edge(mine.daily_returns, bar_returns(actual)))
```

```bash
beatnothing score my_signal.parquet --actual data/actual_returns.parquet
beatnothing members 2020-03-16
beatnothing leaderboard --root .
```

Reproduce the board from a clean clone (about two minutes):

```bash
pytest -q                                # 50 tests
beatnothing leaderboard --track pit
```

From raw data:

```bash
pip install -e ".[data,figures]"
python scripts/download_data.py
python scripts/build_pit_universe.py     # needs a free Tiingo token
python scripts/validate_stats.py
python scripts/make_figures.py pit
```

## Roadmap

1. An outside contestant (every entry is currently mine)
2. Configurable cost model with spread and square-root impact
3. A second market and a weekly horizon
4. DOI and preprint
5. Extend the universe back to 2008
6. A second maintainer
7. An LLM agent contestant

## Limits

- Season one uses 48 survivor stocks picked in 2026. That flatters everyone equally, so
  Net Edge holds up but absolute numbers don't.
- Costs are a flat 10 bps with no market impact, and borrow is a flat rate. The
  sensitivity columns are there for that reason.
- Large caps only. Most published edge is in smaller stocks.
- Net Edge compares Sharpe ratios, so it doesn't reward lower drawdown or diversification.
- The leakage check catches accidents, not someone deliberately hiding a leak.
- Eight months of forward data isn't enough to judge anything yet.

Found a mistake? Open a
["challenge a result"](https://github.com/kaustubhspatil/beatnothing/issues/new?template=challenge.yml)
issue with the claim, what you ran, and what you think is wrong.

## Cite

```
Patil, K. (2026). beatnothing: a net of cost benchmark for daily equity signals. https://github.com/kaustubhspatil/beatnothing
```

A `CITATION.cff` is included. The first five contestants come from the
[deep learning equity signal project](https://github.com/kaustubhspatil/sp500-quantitative-deep-learning).

## Credits

Point-in-time membership: [fja05680/sp500](https://github.com/fja05680/sp500) (MIT).
Kronos: Shi et al., AAAI 2026, [repo](https://github.com/shiyu-coder/Kronos) (MIT).
Stationary bootstrap: Politis & Romano (1994). Probabilistic Sharpe ratio: Bailey & Lopez
de Prado (2012). Sharpe standard error: Lo (2002). Prices from Yahoo Finance via yfinance;
snapshot hash in `data/MANIFEST.json`.

MIT licensed. Built by Kaustubh Patil.
