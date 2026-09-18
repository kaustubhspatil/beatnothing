# beatnothing: a net of cost benchmark for daily equity signals

**Kaustubh Patil**
Version 0.1, September 2026. Source, data and every number: https://github.com/kaustubhspatil/beatnothing

## Abstract

Published machine learning results on equity return prediction are hard to compare and
easy to overstate, for three reasons that are properties of the evaluation rather than of
the models: transaction costs are often reported gross or not at all, the universe is
usually chosen with hindsight, and a leaderboard of many candidates is a multiple test
that is rarely treated as one. We present a benchmark that fixes the evaluation and holds
it still. Every contestant is scored through one engine, on one point in time universe of
S&P 500 members that includes the companies that later failed or were acquired, net of a
flat transaction cost and a borrow charge, against one bar: an equal weight position in
the same universe, held with no skill at all. The score is the Net Edge, the difference in
net Sharpe ratio between a contestant and its bar on the same days, with a studentized
block bootstrap interval and a stepdown correction across the whole board. We report
fifteen contestants, including four architectures of neural network, gradient boosted
trees, three classic cross sectional factors, and a pretrained financial foundation model
used zero shot. None of them clears the bar on a sealed four year window, and none clears
it on the eight months that arrived after the models were frozen. We quantify two effects
that the evaluation makes visible and that a conventional backtest hides: choosing the
universe in hindsight was worth 0.44 of annualized Sharpe ratio, more than any contestant
produced, and charging realistic costs reverses the ranking of the field, taking the
contestant with the highest gross Sharpe ratio to a negative net one. The harness, the
universe and the forward track are open, built entirely from free data sources, and
reproducible on a laptop.

## 1. The problem

A backtest is a measurement instrument, and like any instrument it can be wrong in ways
that flatter its operator. Three failures recur in the literature on machine learning for
equity returns, and the 2026 crop of benchmarks addresses them one at a time.

**Costs are charged late or not at all.** A signal that turns its book over two hundred
times a year and one that turns it over four are not comparable on gross returns, and the
ranking between them is not stable under a change of cost assumption. A benchmark of deep
time series models on a CRSP panel reports that the net Sharpe ratio at twenty basis
points is negative for every model promoted by its own selection procedure, which is the
same finding arrived at from the other direction.

**The evaluation protocol leaks.** Centered features, global normalization, and execution
conventions that use information from the bar being traded all inflate results without any
mistake in the model. A paired study of these conventions finds the inflation is large for
some and negligible for others, which is an argument for fixing the protocol once and
sharing it rather than reimplementing it per paper.

**The universe is chosen with hindsight.** This is the least discussed and, we find, the
largest. "The current constituents of the S&P 500" is a list of companies that survived.
Using it to evaluate a strategy over a past window gives every contestant, and the bar,
a set of names that did not fail. Section 5 measures what that is worth.

To these we add a fourth, which is a property of benchmarks specifically rather than of
individual papers. **A leaderboard is a multiple test.** Fifteen contestants each tested
at the five percent level produce a false winner more than half the time. A benchmark that
does not correct for its own size is a machine for generating spurious winners, and the
more successful it becomes the more it generates.

## 2. Design

### 2.1 One engine

A contestant submits either predictions or weights, per date and ticker, and declares one
of three rules by which predictions become positions: long every name with a positive
value, long the top decile, or long the top decile and short the bottom decile, dollar
neutral. Weights are used as given. Gross exposure never exceeds one; there is no leverage.
Every unit of turnover pays ten basis points, charged on the first day as well, and every
dollar held short pays fifty basis points a year to borrow. The engine trades at the close
and models no market impact beyond the cost parameter. It is deliberately simple, because
its purpose is to be a fixed point that no contestant can adjust.

Two conventions do real work. The first is that a contestant holds nothing in a name on a
day that name is not in the universe, which is what makes point in time membership
enforceable. The second we call **flat when silent**: a day on which a contestant submits
no signal is a day in cash, not a day removed from its record. Without it, a contestant
could submit only on the days it liked and have its Sharpe ratio computed on that subset,
which is a selection effect no statistical test would detect.

### 2.2 Two bars

The primary bar is the **universe bar**: long every member of the contestant's universe,
equal weight, on the same days, paying the same costs. It is what a participant would earn
with no skill whatsoever, and it is the null against which Net Edge is measured.

Beside it we report the **investable bar**, the equal weight S&P 500 exchange traded fund,
which holds every index member by construction and cannot have selected its universe with
hindsight. It pays no turnover cost because it is one position held throughout. The gap
between the two bars is not a nuisance parameter; it is the measurement of Section 5.

### 2.3 The score

For a contestant with daily net returns $r_t$ and its bar $b_t$ over the same $T$ days,

$$\text{Net Edge} = \widehat{SR}(r) - \widehat{SR}(b)$$

both annualized. Because the two series share a calendar, the paired difference is far
better determined than either Sharpe ratio alone, which matters: contestants here are
highly correlated with their bar, and it is the difference we are asking about.

## 3. Statistics

### 3.1 The interval

Sharpe ratio differences are not normally distributed, the underlying returns are fat
tailed, and their volatility clusters. We follow Ledoit and Wolf (2008). Write the four
moments $v = (\mu_1, \mu_2, \gamma_1, \gamma_2)$ with $\gamma_i = E[r_i^2]$, so that the
difference is a smooth function $\Delta(v)$. Its standard error follows from the delta
method,

$$s^2(\hat\Delta) = \frac{1}{T} \nabla\Delta(\hat v)' \; \Psi \; \nabla\Delta(\hat v)$$

with $\Psi$ a heteroskedasticity and autocorrelation consistent estimate of the long run
covariance of the moment vector, computed with a Bartlett kernel after prewhitening each
component with an autoregression and recoloring afterwards, and a bandwidth chosen by the
Andrews plug in rule. The interval and the p value come from a circular block bootstrap in
which **both** the difference and its standard error are recomputed inside every resample,
so that the quantity being resampled is the studentized statistic.

The block length is chosen by the Politis and White rule applied to the statistic's
**influence function** rather than to the return series. This is not a detail. The
difference of two daily return series has almost no autocorrelation in its level, so the
rule applied naively returns blocks of one or two days, and a bootstrap with such blocks
discards the volatility clustering that actually drives the estimator's variance.

### 3.2 The correction

All contestants are resampled on the same dates, so their statistics inherit the
dependence between them, which is strong. On that joint distribution we run the stepdown
of Romano and Wolf (2005): order the studentized statistics, and at each step compare the
largest remaining one with the bootstrap distribution of the maximum over the hypotheses
not yet rejected. The output is an adjusted p value per contestant that controls the
probability of even one false claim across the board, at far higher power than a
Bonferroni correction because it uses the correlation between contestants rather than
assuming the worst. The adjusted p value is what determines a verdict on the leaderboard.

### 3.3 The search

A contestant who tried forty variants and submits the best has told us little. We
implement the combinatorially symmetric cross validation of Bailey et al. (2014), which
splits the record into blocks, selects the best variant on every half and evaluates its
rank on the complementary half, and reports the probability that the relative rank is no
better than the median. Pure noise gives one half. We also report the deflated Sharpe
ratio, which discounts an observed Sharpe by what the search alone would be expected to
produce.

### 3.4 The statistics are measured, not asserted

The machinery above is itself validated by simulation, on return series with fat tails,
clustered volatility, and a contestant correlated with its bar at 0.9. On 250 simulations
of four years of daily data, the studentized test claims an edge when there is none 4.8
percent of the time against a nominal five, and finds a real edge of a third of a Sharpe
ratio 42.4 percent of the time. A percentile bootstrap of the same paired difference is
conservative, firing at 2.4 percent, and pays for it with lower power at 38.8 percent.
Size remains near the nominal level as the record lengthens from two years to sixteen,
so the residual distortion is a finite sample effect rather than a defect of the method.

Constructing that null correctly is harder than it looks, and the first attempt was wrong
in a way that would have been invisible in the output: adding zero mean noise to the bar
leaves the mean unchanged but raises the variance, so every simulated contestant was
genuinely worse than its bar, and the familywise experiment could not have produced a
false winner at all. The reported experiment gives every contestant the bar's mean and
variance exactly.

## 4. The universe

Point in time membership of the S&P 500 comes from a public community maintained record
of every index change since 1996. For each member on each day we need a price, including
for companies that no longer exist. We assemble this from free sources in three layers.
Names that still trade come from one public quote provider. Sixteen names changed ticker
without ceasing to trade and are mapped through a hand verified alias table that ships
with the package. The remainder, thirty nine companies that were acquired or failed, come
from the free tier of a second provider that retains delisted histories, each fetched to
its final trading day.

Coverage of the sealed and forward windows is 99.5 percent of the member days the index
defines, about 501 names a day. The residual is one company with no free price history
after a 2026 merger and four 2026 spin offs too young to have features. Because the second
provider's free tier is licensed for personal use, the repository versions the realised
returns derived from those histories rather than the raw prices, and ships the script that
rebuilds the full panel from a reader's own free credentials.

## 5. Results

Full tables, intervals, adjusted p values and per contestant diagnostics are in the
repository; we state here only what the benchmark was built to measure.

**Hindsight in the universe is worth more than any contestant produced.** The forty eight
name survivor universe, chosen in 2026 from names that were then in the index, gives its
own do nothing bar an annualized Sharpe ratio of 0.88 over 2022 to 2025 against 0.44 for
the investable equal weight index over the same days, a difference of 0.44 with a
confidence interval of 0.19 to 0.76. On the point in time universe the same bar scores
0.47 against the index's 0.44, a difference of 0.03 with an interval spanning zero. The
premium is not a subtlety; it is larger than the Net Edge of every contestant in this
report, and it would pass any test of statistical robustness applied to the strategy
alone, because it is in the data rather than in the statistics.

**Costs reorder the field.** On the survivor universe the contestant with the highest
gross Sharpe ratio, a one dimensional convolutional network at 1.08, has a negative net
Sharpe ratio, because it turns its book over 265 times a year. A pretrained financial
foundation model used zero shot goes from a gross 0.52 to a net minus 0.79 at 241 turns a
year. The rank order of the field at zero cost is not its rank order at ten basis points.

**Nothing clears the bar.** On both universes, on the sealed window and on the 176 days
that arrived after every model was frozen, no contestant's adjusted p value clears five
percent. The best learned contestant is measurably, slightly worse than doing nothing. A
network trained end to end on net Sharpe with a turnover penalty lands on the bar with
half the exposure. Gradient boosted trees under a squared error objective early stop after
three trees and become the bar in disguise, which is what minimizing squared error on
daily returns does: it predicts the drift, and the drift is the bar.

**The classic factors calibrate as the literature predicts**, which is the check that the
instrument is measuring the right thing. Twelve month momentum skipping the last month is
the only factor with a positive Net Edge on both a long only and a dollar neutral
construction, and even so its interval spans zero over four years. Short term reversal is
destroyed by its own turnover. Low volatility is flat. A harness that had shown momentum
at a Sharpe ratio of two would have been reporting a bug.

## 6. Limitations

The contestants in this report were trained on the survivor universe and evaluated on the
point in time one, which removes the hindsight from the evaluation but not from the
training. Extending the universe to 2005 and retraining is the next version. Costs are a
flat ten basis points with no market impact and no size dependence, which is the friction
of a small book. The borrow charge is a constant and does not model availability. Prices
are adjusted closes from a free provider and the membership record is community
maintained, both serviceable and neither authoritative. The forward track is eight months
old; it becomes evidence at one or two years. Finally, the benchmark evaluates daily long
or flat and dollar neutral books on United States large capitalization equities, and
nothing here should be read as applying beyond that.

## 7. Related work

Benchmarks of deep time series models on equity panels establish the net of cost result on
proprietary data. Work on decision time leakage isolates which evaluation conventions
inflate results. The probability of backtest overfitting and the deflated Sharpe ratio
provide the multiple testing correction for a single researcher's search, and composite
robustness grades assemble them into a score for a submitted strategy. Commercial services
exist that certify when a signal existed, using cryptographic timestamps, and that grade
the statistical fragility of a submitted backtest. What distinguishes the present work is
that it supplies the market rather than grading the participant's own: the universe, the
costs, the bar and the engine are fixed and shared, so that the comparison between two
contestants is a comparison and not a coincidence of two different experimental setups.

## 8. Availability

Everything is MIT licensed and reproducible from free sources. The package installs from
the Python package index, the leaderboard rebuilds with one command, and a scheduled job
rescores every registered contestant each month on the days that have arrived since it was
frozen. Contributors whose contestant is merged and survives a year of forward track are
named authors on the next version of this report.

## References

Andrews, D. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix
estimation. *Econometrica* 59(3).

Andrews, D. and Monahan, J. (1992). An improved heteroskedasticity and autocorrelation
consistent covariance matrix estimator. *Econometrica* 60(4).

Bailey, D. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier.
*Journal of Risk* 15(2).

Bailey, D. and Lopez de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio
Management* 40(5).

Bailey, D., Borwein, J., Lopez de Prado, M. and Zhu, Q. (2014). The probability of
backtest overfitting. *Journal of Computational Finance*.

Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers.
*Journal of Finance* 48(1).

Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe
ratio. *Journal of Empirical Finance* 15(5).

Lo, A. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4).

Politis, D. and Romano, J. (1994). The stationary bootstrap. *Journal of the American
Statistical Association* 89(428).

Politis, D. and White, H. (2004). Automatic block length selection for the dependent
bootstrap. *Econometric Reviews* 23(1).

Romano, J. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping.
*Econometrica* 73(4).

Shi, Y. et al. (2026). Kronos: a foundation model for the language of financial markets.
*AAAI*.
