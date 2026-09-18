# Study guide: build this from nothing

Everything this project knows, in the order you would need to learn it, with the commands
to rebuild every piece. It assumes you can write Python and have never touched finance.

There is a plain language version in Hinglish at the end. If the jargon starts winning,
skip to [section 12](#12-hinglish-samajh-lo-simple-mein) and come back.

**Contents**

1. [The question, and why it is the whole project](#1-the-question)
2. [Eight ideas you must hold in your head](#2-eight-ideas)
3. [The data, and the four ways it lies to you](#3-the-data)
4. [Part one: the models](#4-part-one-the-models)
5. [Part two: the engine](#5-part-two-the-engine)
6. [Part three: the statistics](#6-part-three-the-statistics)
7. [Part four: the universe that did not know the future](#7-part-four-the-honest-universe)
8. [Rebuild it, step by step](#8-rebuild-it-step-by-step)
9. [Checkpoints: how to know you got it right](#9-checkpoints)
10. [Every mistake made here, and what it cost](#10-every-mistake)
11. [Glossary](#11-glossary)
12. [Hinglish, samajh lo simple mein](#12-hinglish-samajh-lo-simple-mein)

---

## 1. The question

Almost every published trading backtest compares its strategy against zero, or against an
index measured without trading costs. Both comparisons are too easy, and they are too easy
in a way that is invisible unless you go looking.

This project asks one question instead:

> **Does your model beat holding everything at equal weight and never trading again, after
> both of you pay the same costs, on a window neither of you could have seen?**

That is the entire project. Everything else is machinery for answering it honestly.

The answer so far, across twenty five contestants including four neural architectures,
gradient boosted trees, classic factors and a 2026 financial foundation model, is **no**.
Not one clears the bar. That result is not interesting on its own, because plenty of people
suspect it. It is interesting because it is measured, with error bars, on sealed data, with
a multiple testing correction, and anybody can try to overturn it.

**Why this matters more than a better model.** A model that beats zero tells you nothing.
A model that beats an index gross of costs tells you slightly less than nothing, because it
hides the fee. The only comparison that carries information is against the thing you would
have done with no skill whatsoever, paying what you would actually pay.

---

## 2. Eight ideas

Learn these eight and the rest of the project is implementation detail.

### 2.1 Doing nothing is a strategy, and it is a strong one

Hold every stock in your universe at equal weight. Rebalance almost never. Over 2022 to
2025 on this universe that earns a Sharpe ratio of about +0.88 net of costs. Any model you
build starts the race behind this, because this costs nothing to run and cannot be wrong.

### 2.2 Costs are not a detail, they are the whole game

Every time you change a position you pay. Here it is 10 basis points of the value traded,
which is 0.1%. That sounds tiny. Multiply by turnover.

A model that rebalances its whole book daily turns over roughly 250 times a year. At 10 bps
that is 250 × 0.001 = **25% of capital a year, paid in fees.** No daily equity signal on
large cap US stocks produces 25% a year gross. The arithmetic is unforgiving and it kills
almost everything.

This produces the single most reliable finding in the project: **gross rank is not net
rank.** The 1D CNN has the best gross Sharpe of any contestant, +1.08, and a negative net
Sharpe, because it flips its book 265 times a year.

### 2.3 Predicting tomorrow's return means predicting the drift, and the drift is the bar

Train a model with squared error on next day returns. The minimiser of squared error is the
conditional mean. The conditional mean of a stock's daily return is dominated by the market
going up. So your model learns to say "up" for almost everything, you go long almost
everything, and you have rebuilt the bar with extra steps and extra fees.

This is why LightGBM, stopped honestly on validation loss, stops after **three trees** and
holds 47 of 48 names. It is not broken. It found the answer.

The way out is to change the target, not the optimiser: rank the names against each other
**within each day**, so the common market move cancels by construction.

### 2.4 Knowing something and being able to trade it are different questions

The cross sectional ranking network has the highest information coefficient on the board,
+0.0154, which is a genuine, real signal. It still loses:

| rebalance | information coefficient | gross Sharpe | net Sharpe | turnover | cost in Sharpe |
|---|---|---|---|---|---|
| daily | +0.0154 | +0.72 | −1.50 | 222× | 2.22 |
| weekly | +0.0007 | +0.32 | −0.37 | 67× | 0.69 |
| monthly | −0.0039 | −0.10 | −0.30 | 20× | 0.20 |

Read the first column down: the information is real at one day and gone within five. Read
the last column up: harvesting it daily costs three times what it produces gross. Trade fast
enough to catch the signal and costs eat it; slow down to save the costs and the signal has
already decayed. **This is the most important result in the project.**

### 2.5 Survivorship bias is not a caveat, it is a number

Pick your stock universe today and backtest it over the last ten years, and you have
quietly filtered for companies that survived. That is hindsight, and here is what it is
worth, measured:

| universe | premium over the honest one |
|---|---|
| 48 names chosen in 2026 | **+0.44 Sharpe**, interval [+0.19, +0.76] |
| point in time membership | **+0.03 Sharpe**, interval [−0.01, +0.06] |

+0.44 of Sharpe is larger than most edges anybody publishes. Most backtests contain this
and do not measure it.

The fix is a **point in time universe**: on every date, the names that were actually in the
index on that date, with dead companies left in at the price they actually died at. Silicon
Valley Bank is in this universe, losing 60.4% on its last day and 84.6% cumulatively. First
Republic is in it, at −90.5% and −99.8%.

### 2.6 A leaderboard manufactures winners if you let it

Put fifteen contestants with no edge at all on a board and test each at the 5% level. The
chance that at least one looks significant is not 5%. Simulated here, it is **61%**.

So the leaderboard does not report a naive p value. It runs a **Romano and Wolf stepdown**,
which controls the familywise error across the whole board. Under the same simulation that
drops the false winner rate from 61% to **1%**.

If you take one methodological idea from this project into your own work, take this one. It
applies to any leaderboard, any hyperparameter sweep, any A/B dashboard.

### 2.7 A Sharpe ratio without an error bar is a rumour

Sharpe ratios are noisy. On 176 trading days the standard error is about ±1.2. That means a
strategy reporting 1.0 and one reporting 0.0 are not distinguishable on eight months of
data.

So every number here carries a 95% interval from a **studentized circular block bootstrap**,
which respects the fact that returns are serially correlated. You clear the bar only when
the entire interval is above zero.

### 2.8 Frozen means frozen, and it must be verifiable

Anybody can claim they predicted something. The benchmark makes it checkable: every
submission records the sha256 of its signal file and a registration date, entries arrive by
pull request so the public git history dates them, and a monthly job rescores everything on
days that arrived **after** registration. The signal cannot change. Only the calendar moves.

This is the property no private backtest can imitate, and it is the strongest thing the
project has.

---

## 3. The data

### 3.1 What you need

| thing | what it is | source used here |
|---|---|---|
| daily prices | open, high, low, close, volume, adjusted for splits and dividends | Yahoo Finance via `yfinance`, free |
| index membership over time | who was in the S&P 500 on each date, by effective date | `fja05680/sp500`, MIT licensed |
| prices for dead companies | the ones that were delisted, acquired or went bankrupt | Yahoo where it still has them, Tiingo free tier for the rest |
| market context | VIX, 10 year treasury yield | Yahoo (`^VIX`, `^TNX`) |

Everything here was built on free data. That is a deliberate constraint and it has one real
cost, explained in 3.4.

### 3.2 The four ways price data lies

**Survivorship.** Covered in 2.5. The universe must be point in time.

**Look ahead in features.** Any feature computed on date t that uses information from after
t makes the model look brilliant and it is worthless. The test is mechanical: delete all
data after date t, recompute the features, and check they are bit for bit identical. If
they change, you have a leak.

**Adjustment restatement.** Adjusted close prices change retroactively when a dividend or
split happens. A backtest run today sees prices that did not exist then. This is small for
daily large cap work but it is real, and it is why the data snapshot is hashed in
`data/MANIFEST.json`.

**Announcement versus effective dates.** Index changes are announced days before they take
effect. A membership table built from announcement dates quietly drops a failing company
early and avoids part of its loss. This project uses **effective dates**: SVB leaves on 15
March 2023, First Republic on 4 May 2023. It also forgoes the pop a stock gets when its
inclusion is announced, which is the same rule applied fairly in both directions.

### 3.3 The seventeen features

All trailing, all computable at the close of day t:

```
ret_1d  ret_5d  ret_21d  ret_63d          returns over 1, 5, 21, 63 days
vol_21d  atr_14                            realised volatility, average true range
rsi_14  macd_hist                          momentum oscillators
boll_pct_b  boll_bw                        position in and width of Bollinger bands
px_vs_sma10  px_vs_sma50                   price against its moving averages
volume_z                                   volume, standardised
vix  vix_chg_5d                            market fear, level and change
treasury_10y  treasury_chg_21d             rates, level and change
```

Seventeen is not a magic number. The point is that they are all backward looking and all
cheap. If a model cannot beat doing nothing with these, adding an eighteenth is unlikely to
be the problem.

### 3.4 Where free data runs out, and why the universe stops in 2013

Of 362 companies that left the index before 2021, **291 were recoverable** from free
sources. That gives 98% coverage from 2013 but only 85% in 2005, and the missing names in
2008 are exactly the ones that matter: Lehman Brothers, Bear Stearns, Washington Mutual,
Countrywide, Fannie Mae, Freddie Mac.

So the universe **deliberately stops at 2013**. Extending it to 2008 on free data would put
survivorship bias into the one window where it would most distort the answer. Stopping is
the honest choice, and it means **nothing in this project has been tested through a
crisis.** That is the single largest remaining gap, and the only one money would fix.

### 3.5 Ticker reuse, the trap that looks like diligence

Dead companies are sometimes recoverable under a successor ticker. Twenty two renames were
verified and are in `beatnothing/data/ticker_aliases.json`. Four candidates were **rejected**
and are recorded with reasons, because they pass a naive continuity check and are still
wrong:

| looks like | actually |
|---|---|
| NYX → ICE | NYSE Euronext was acquired by a different company |
| CCE → CCEP | a different corporate entity |
| ESV → VAL | Valaris is new equity issued after the old was wiped out |
| CBS → PARA | same |

In the last two the old shareholders lost everything. Splicing the new price series onto the
old one would erase a total loss and replace it with a continuing company. Recording the
rejections matters as much as recording the accepted ones, so nobody repeats the work or
the mistake.

---

## 4. Part one: the models

Five architectures, all trained with the same discipline: fit on everything up to
2018‑12‑31, choose on 2019 to 2021, never look at what follows.

| model | what it is | what happened |
|---|---|---|
| linear regression | ordinary least squares on the 17 features | the incumbent; 159× turnover, net Sharpe −0.30 |
| feedforward network | 128, 64, 32 with dropout 0.3, three seeds averaged | lands almost exactly on the bar |
| LSTM | 60 day windows | 43× turnover, net Sharpe +0.35 |
| 1D CNN | 60 day windows | best gross Sharpe of all, +1.08; negative net |
| ResNet‑18 on chart images | transfer learning, charts rendered as pictures | AUC 0.52 frozen, 0.49 fine tuned; never beat the majority class |
| LightGBM | gradient boosted trees | stops after 3 trees, holds 47 of 48 names |
| cost aware network | trained end to end on net Sharpe with a 10 bps turnover term | turnover falls from 12× to 2.9×; lands on the bar |
| Kronos small, zero shot | 24.7M parameter financial foundation model, AAAI 2026, no training | gross +0.52, net −0.79, 241× turnover, −53% drawdown |
| cross sectional ranking network | predicts within day rank, traded long short | highest information coefficient on the board; still loses |

**The pattern.** Every model trained on the level of returns converges on the bar, because
of 2.3. The cost aware objective fixes turnover, not alpha: it learns to stop trading, which
is correct, and then it is the bar. The foundation model with a hundred thousand times the
parameters of the linear regression obeys exactly the same arithmetic as the linear
regression. Scale does not repeal costs.

---

## 5. Part two: the engine

One engine scores everybody. Nobody brings their own backtester, because a custom
backtester is where results are manufactured.

### 5.1 From predictions to positions

A contestant submits numbers and declares one of three rules:

| rule | what it does |
|---|---|
| `long_flat` | long every name with a positive value, equal weight |
| `long_top` | long the top decile, equal weight |
| `long_short` | long the top decile, short the bottom, dollar neutral, 50 bps a year borrow charge |

Gross exposure is capped at one. No leverage. Weights are used as given.

**Flat when silent.** A day on which a contestant produces no signal is a day it holds cash.
It is not skipped and it does not silently inherit a previous position. If a model can only
rank on 5% of days, its score reflects the 95% it sat out.

**Tie breaking matters more than it sounds.** Ranking with averaged ties means a signal with
few distinct values gives most names the same rank, nothing clears a decile threshold, and
the book sits empty while reporting a valid score. The engine uses first occurrence ordering
so a decile rule always holds a decile, and the validator warns when a signal is too coarse
for the rule it declared.

### 5.2 The two bars

**The universe bar.** Always long, equal weight, same universe, same days, same costs. This
is what you would earn with zero skill on the names you chose.

**The investable bar, RSP.** The equal weight S&P 500 ETF. It holds every index member by
construction, dead ones included, and it could not have chosen its universe with hindsight.

Read them together. A contestant that beats RSP while failing the universe bar has
demonstrated exactly one thing: that its universe was chosen with hindsight. That is what
the +0.44 in 2.5 is.

A dollar neutral book is judged against cash, not against the index, because it is not
taking market exposure.

### 5.3 The score

```
Net Edge = your net Sharpe − the bar's net Sharpe, on the same days
```

with a 95% paired stationary block bootstrap interval. You clear the bar only when the whole
interval is above zero, and then only after the stepdown correction of 2.6.

---

## 6. Part three: the statistics

This is the part that separates a benchmark from a backtest. The machinery:

| technique | why | reference |
|---|---|---|
| studentized circular block bootstrap | Sharpe differences are serially correlated; a naive bootstrap gets the size wrong | Ledoit and Wolf (2008) |
| HAC covariance, AR(1) prewhitening | the four moments behind two Sharpe ratios are autocorrelated | Andrews and Monahan (1992) |
| automatic Bartlett bandwidth | choosing it by hand is a researcher degree of freedom | Andrews (1991) |
| delta method over four moments | turns moment uncertainty into Sharpe difference uncertainty | standard |
| block length from the influence function | the block should match the dependence of the *statistic*, not the raw series | Politis and White (2004) |
| Romano and Wolf stepdown | controls familywise error across the board | Romano and Wolf (2005) |
| probability of backtest overfitting | how much of your best variant was the search itself | Bailey et al. (2014) |
| deflated and probabilistic Sharpe | adjusts for the number of trials and non normality | Bailey and Lopez de Prado |

### 6.1 The statistics were tested against themselves

Do not trust a test because it has a famous name. Simulate data where you know the truth
and check that the test says so:

| question | result |
|---|---|
| size, nominal 5% | **4.8%** false positive rate on data with no edge |
| power | 42% at a small real edge, where the older percentile interval gets 39% |
| false winner on a 15 contestant board with no edge | **61%** uncorrected, **1%** with stepdown |
| probability of backtest overfitting, 12 pure noise variants | **66%**; given one variant with a real edge, **0%** |

The third row is why the verdict column reports an adjusted p value and not an interval.

On the real board, after the stepdown across 24 contestants, the lowest adjusted p value is
**0.997**. Nothing comes close, and now we know the test would have said so if it did.

---

## 7. Part four: the honest universe

The first two seasons trained on 48 survivor names and evaluated on 500, which removes
hindsight from the evaluation but not from the training. Season three removes it from both:
same architectures, refitted on the point in time universe back to 2013.

Result: the ordering barely changes, the absolute numbers roughly halve, and **the
conclusion does not move.** A strategy that hugs the bar hugs whatever bar it is given.

Season three is also where the ranking network of 2.4 arrives, and where its failure becomes
the most interesting result on the board.

---

## 8. Rebuild it, step by step

### Step 0. Environment

```bash
git clone https://github.com/kaustubhspatil/beatnothing && cd beatnothing
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev,data,figures]"
pytest -q                                            # 50 tests, all should pass
```

### Step 1. The five minute path, to see the shape of everything

```bash
python scripts/starter_submission.py --name my_first_try --author "your handle"
beatnothing validate submissions_pit/my_first_try
beatnothing leaderboard --track pit
```

The starter builds a five day reversal signal from the returns panel committed to the repo.
No API key, no download. It scores an information coefficient of +0.0142 and a net Sharpe
of −1.14 at 201× turnover, which teaches 2.2 and 2.4 in one command.

### Step 2. Prices

```bash
python scripts/download_data.py            # survivor universe prices, VIX, treasury
python scripts/download_pit_prices.py      # every point in time member, including dead ones
```

The second needs a free Tiingo token for the names Yahoo no longer serves. About twenty
requests of a 500 per hour free quota. Set `TIINGO_TOKEN` in your environment.

### Step 3. The point in time universe

```bash
python scripts/build_pit_universe.py --start 2013-01-01 --eval-start 2022-01-01 --suffix _2013
```

This reads index membership by effective date, resolves verified ticker renames, refuses the
rejected ones, and writes the returns panel plus a coverage report. Check
`data/pit/coverage_2013.json` before continuing: coverage below about 95% means you are
rebuilding survivorship bias.

### Step 4. Features

Computed from prices only, all trailing. The leakage test is not optional:

```python
from beatnothing.features import build_feature_panel

full = build_feature_panel(prices, market)
cut  = build_feature_panel(prices[prices.Date <= "2020-06-30"],
                           market[market.index <= "2020-06-30"])
both = full[full.Date <= "2020-06-30"].reset_index(drop=True)
assert both.equals(cut.reset_index(drop=True))   # deleting the future must change nothing
```

### Step 5. Train the contestants

```bash
python scripts/train_on_pit.py --panel data/pit/feature_panel_pit_2013.parquet
python scripts/train_on_pit.py --models ranker        # just the cross sectional ones
python scripts/factor_contestants.py                  # momentum, reversal, low volatility
python contestants/kronos_pit.py --stride 1           # the foundation model, GPU, resumable
```

Splits: fit to 2018‑12‑31, choose on 2019 to 2021, never look after. If you change that, you
are no longer running this benchmark.

For Kronos on a consumer GPU: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, batch 24,
`empty_cache()` per date, group inputs by context length before batching, and let an out of
memory error skip a date rather than kill the run. That took peak memory from 7.8 GB to 1.5
GB and the estimate from 11 hours to 40 minutes.

### Step 6. Validate every submission

```bash
python scripts/validate_submissions.py --write-hash
```

Checks structure, the engine's rules, missing and infinite values, a minimum of 120 days,
and a leakage smell test. An information coefficient above 0.15 or a net Sharpe above 5 is
treated as a bug report against your pipeline, not a discovery.

### Step 7. Statistics, board and figures

```bash
python scripts/validate_stats.py                 # size, power, familywise, overfitting
beatnothing leaderboard --track survivor48
beatnothing leaderboard --track pit
python scripts/make_figures.py survivor48
python scripts/make_figures.py pit
```

### Step 8. Publish

```bash
python -m build && python -m twine check dist/*
python -m twine upload dist/*
```

Note the two traps that cost time here. GitHub resolves relative image paths in a README;
**PyPI does not**, so figures must use absolute `raw.githubusercontent.com` URLs. And GitHub
proxies badge images through a cache that will keep serving "package or version not found"
long after the package exists; a purge may not dislodge it, and the fix that works is a
slightly different badge URL.

---

## 9. Checkpoints

If you rebuilt this correctly, these should all be true. If one is not, stop and find out
why before continuing.

| check | expected |
|---|---|
| leakage test in step 4 | features identical when the future is deleted |
| point in time coverage from 2013 | about 98% |
| SVB last day return | about −60.4% |
| First Republic cumulative | about −99.8% |
| LightGBM on next day returns, early stopped | stops after about 3 trees, holds nearly everything |
| bar net Sharpe, sealed window, survivor universe | about +0.88 |
| bar edge over RSP, survivor universe | about +0.44 |
| bar edge over RSP, point in time universe | about +0.03 |
| measured size of the test | about 4.8% at nominal 5% |
| familywise, 15 null contestants | about 61% uncorrected, about 1% corrected |
| contestants clearing the bar | zero |

---

## 10. Every mistake

The mistakes are the most useful part of this document, because most of them are available
to anybody doing this work and several nearly published a false claim.

**An information coefficient of +0.028 that was not real.** The metric averaged the daily
rank correlation over days where the signal varied. An early stopped tree model predicts one
constant and varies on 5% of days, so it was credited with a respectable coefficient
measured on an unrepresentative sliver. It nearly went into the README as a finding. The fix
is to report the **share of days the signal can rank** alongside the number, and return zero
below half. *Lesson: a metric that silently drops observations will eventually flatter the
worst model you have.*

**A familywise simulation that reported 0% both ways.** The null was not a null: adding zero
mean noise to the bar leaves the mean but raises the variance, so every simulated contestant
was genuinely worse and the board could not have crowned anybody. Matching both mean and
variance turned 0% into the real 61% versus 1%. *Lesson: when a simulation reports that
nothing happens, suspect the simulation first.*

**Decile rules that held no names.** Averaged tie ranking plus a coarse signal meant nothing
crossed the threshold and the book sat empty while reporting a valid score. Fixed with first
occurrence ordering and a validator warning. *Lesson: an empty book must be loud.*

**A variance floor that underflowed.** `1e-300` raised to the power 1.5 underflows to zero
and turns a gradient into a NaN, which killed the whole leaderboard build. Raised to
`1e-30`. *Lesson: floors need to survive the operations applied to them, not just be small.*

**A degenerate comparison that divided near zero by near zero.** A contestant identical to
its own bar produced a standard error of 2.6e−17 on one BLAS and 0 on another, so CI passed
on one Python version and failed on another. Fixed by detecting the degeneracy from the
series with `np.allclose`, not from the standard error. *Lesson: never branch on a floating
point value being exactly zero.*

**A test that destroyed the thing it was testing.** A helper used `.stack()`, which drops
NaN on older pandas, so the missing values test had no missing values on Python 3.10.
Switched to `.melt()`. *Lesson: when CI fails on one version only, suspect the test.*

**Four of my own submissions failed my own validator**, on 1,382 NaNs for names too young to
have a momentum score. That is the validator working. *Lesson: if your gate never rejects
you, it is not a gate.*

**A 191 MB file rejected by GitHub**, requiring a gitignore fix, `git rm --cached`, an
amended commit and a `git gc` that took the repo from 235 MB to 58 MB.

**An overnight chain that failed silently** because a subprocess ran with `check=False`.
*Lesson: a pipeline that cannot fail loudly will fail quietly.*

**Three stale numbers in the README**, left from an earlier fourteen contestant board:
the stepdown count, the number of positive point estimates, and the lowest adjusted p value.
The verdict was unchanged but the numbers did not match the data. *Lesson: numbers in prose
go stale silently while numbers in code do not. Generate them if you can.*

---

## 11. Glossary

**Sharpe ratio.** Return divided by volatility, annualised. Reward per unit of risk. Above
1 is good, and almost nothing sustains it.

**Basis point (bps).** One hundredth of a percent. 10 bps = 0.1%.

**Turnover.** How much of your book you replace, annualised. 200× means you replace the
whole book two hundred times a year.

**Information coefficient.** The daily correlation between your ranking of stocks and what
actually happened. A real one is 0.02 to 0.05. This is much smaller than people expect.

**Drawdown.** The worst fall from a previous peak.

**Point in time.** Data as it was known on the date in question, not as it is known now.

**Survivorship bias.** The distortion from studying only the things that survived.

**Look ahead bias.** Using information that did not exist yet.

**Dollar neutral.** Equal money long and short, so the market's direction roughly cancels.

**Bootstrap.** Estimating uncertainty by resampling your data many times. A **block**
bootstrap resamples chunks, to preserve the fact that days near each other are related.

**Familywise error.** The chance of at least one false positive across a whole set of tests.

**Stepdown.** A procedure that tests the strongest candidate first and works down, keeping
the familywise error controlled without being as blunt as Bonferroni.

**Zero shot.** Using a pretrained model with no further training on your data.

---

## 12. Hinglish, samajh lo simple mein

Ab bina jargon ke. Agar aap finance se nahi hain, yeh section akela bhi kaafi hai.

### Sawaal kya hai

Maan lo aapke paas paise hain aur aap share market mein lagana chahte hain. Do raaste hain.

**Raasta ek, sabse aasaan.** Market ki 500 badi companies mein barabar barabar paisa laga
do. Phir kuch mat karo. Bas baithe raho. Isme na dimaag lagta hai, na mehnat, na fees.

**Raasta do.** Ek smart computer model banao, jo roz batata hai ki kaunsa share upar jayega,
aur uske hisaab se roz khareedo aur becho.

Poora project sirf ek sawaal poochta hai: **kya raasta do, raasta ek se behtar hai, jab
dono par same kharcha lagta ho?**

Humne 25 alag alag model try kiye. Neural networks, decision trees, aur ek naya "foundation
model" jo 12 arab market bars par train hua hai. **Ek bhi nahi jeeta.** Kuch bhi nahi.

### Har baar kyon haarte hain

Teen wajah hain, aur teeno seekhne layak hain.

**Ek. Broker har trade par paisa kaatta hai.** Ek trade par sirf 0.1%. Lagta hai kuch bhi
nahi. Lekin agar model roz pura portfolio badalta hai, toh saal mein 250 baar trade hua,
matlab **saal ka 25% sirf fees mein chala gaya.** Koi bhi model itna kamata hi nahi. Ganit
seedha hai aur bilkul bebaak: fees pehle hi sab kha jaati hai.

Isliye project ka sabse pakka result yeh hai: **fees se pehle jo model number one tha, fees
ke baad woh sabse neeche chala jaata hai.** Ek model ka gross sabse accha tha, 1.08. Net
mein woh negative hai, kyunki woh saal mein 265 baar pura portfolio palat deta hai.

**Do. Model asal mein wahi seekh leta hai jo hum free mein kar rahe the.** Jab aap model ko
kehte ho "kal ka return batao", toh sabse safe jawab hai "thoda upar", kyunki market aam
taur par upar hi jaata hai. Toh model sab shares ke liye "upar" bol deta hai, aap sab kuch
khareed lete ho, aur aapne wahi raasta ek bana diya, bas upar se fees bhi de di.

Ek model to sirf **teen faisle** lekar ruk gaya aur 48 mein se 47 shares pakad liye. Woh
kharab nahi tha. Usne jawab dhoondh liya tha.

**Teen. Jo model sach mein kuch jaanta tha, woh bhi haar gaya, aur yeh sabse dilchasp baat
hai.** Humne ek model banaya jo market ki direction nahi, balki yeh batata hai ki *kaunsa
share kaunse se behtar karega.* Uske paas sach much information thi, sabse zyada. Phir bhi:

- Roz trade karo: information kaam karti hai, par fees teen guna zyada le jaati hai
- Hafte mein ek baar: fees bach gayi, par tab tak information khatam ho chuki hoti hai
- Mahine mein ek baar: information hai hi nahi

Matlab: **kuch jaanna aur us jaankari se paisa kama lena, do alag alag cheezein hain.** Yeh
line poore project ka nichod hai.

### Do aur baatein jo sab galat karte hain

**Pehli: aaj ki companies chunkar purana test mat karo.** Agar aap aaj ki 50 acchi companies
lekar unka 10 saal purana record dekhoge, toh aapne chupke se sirf woh chuni jo bach gayi.
Jo companies doob gayi, unko to aapne list mein liya hi nahi.

Humne naapa ki is chori ka kitna faayda milta hai: **0.44 Sharpe.** Yeh itna bada hai ki jo
"edge" log research papers mein dhoondhte hain, woh usse chhota hota hai. Matlab aapka poora
"profit" sirf is galti se aa sakta hai.

Sahi tareeka: har din wahi companies lo jo us din sach mein index mein thi, **aur doobi hui
companies ko bhi usi nuksan ke saath rakho.** Silicon Valley Bank is list mein hai, apne
aakhri din 60% girta hua. First Republic hai, 99.8% doobta hua.

**Doosri: 25 model ek saath test karoge toh koi na koi galti se jeet hi jayega.** Socho 15
bekaar models ko test karo. Kya chance hai ki koi ek galti se accha dikh jaye? 5% nahi.
Humne simulate karke dekha: **61%.** Har doosra leaderboard ek jhootha winner banata hai.

Isliye humne ek statistical correction lagayi. Usse yeh 61% se **1%** ho gaya. Yeh cheez
sirf trading ke liye nahi hai. Jahan bhi aap bahut saare options mein se "best" chunte ho,
yeh dikkat aati hai.

### Toh natija kya hai

Humne 25 model banaye, poori imaandaari se test kiye, aur likh diya ki **koi nahi jeeta.**

Yeh haar nahi hai. Yeh natija hai, aur imaandaar natija hai.

Zyada log jeet dikhane ke liye kuch na kuch chhupa lete hain. Fees kam dikha dete hain, ya
purani companies chun lete hain, ya 50 model try karke sirf jo jeeta wahi dikhate hain. Yeh
project unn teeno raaston ko band karta hai, aur phir bhi imaandaari se jawab deta hai.

Aur sabse badi baat: **poora code aur data public hai.** Agar aapko lagta hai ki humse
galti hui, aap khud check kar sakte ho, aur galat sabit kar sakte ho. Ek command mein aap
apna model iss leaderboard par daal sakte ho.

Asli science yahi hoti hai. Jo cheez galat sabit ho sakti ho, wahi bharose ke layak hai.
