# Submitting a contestant

There are two tracks, one engine. `submissions/` scores on the 48 name survivor universe
of season one; `submissions_pit/` scores on the point in time universe, every S&P 500
member on each day, dead names included. Submit to the point in time track unless you
have a reason not to: it is the one where the bar could not choose its names.

A contestant is a folder under the track's submissions directory with two files.

## `signal.parquet`

Long format, three columns, nothing else.

<table>
<tr><th>column</th><th>type</th><th>meaning</th></tr>
<tr><td><code>Date</code></td><td>datetime</td><td>the decision date; the signal is formed from information available at that close</td></tr>
<tr><td><code>Ticker</code></td><td>string</td><td>the stock, as the membership table spells it</td></tr>
<tr><td><code>value</code></td><td>float</td><td>a prediction (any real number) or a weight, depending on <code>kind</code></td></tr>
</table>

One row per date and ticker, no missing or infinite values, at least 120 trading days.

## `meta.json`

<table>
<tr><th>field</th><th>required</th><th>meaning</th></tr>
<tr><td><code>name</code></td><td>yes</td><td>must equal the folder name</td></tr>
<tr><td><code>kind</code></td><td>yes</td><td><code>predictions</code> or <code>weights</code></td></tr>
<tr><td><code>description</code></td><td>yes</td><td>one paragraph a stranger can understand</td></tr>
<tr><td><code>registered</code></td><td>yes</td><td>the date you froze it; the forward track counts only days after this</td></tr>
<tr><td><code>rule</code></td><td>predictions</td><td><code>long_flat</code> (default), <code>long_top</code>, or <code>long_short</code></td></tr>
<tr><td><code>quantile</code></td><td>no</td><td>the fraction on each side for the ranking rules, a tenth by default</td></tr>
<tr><td><code>training_cutoff</code></td><td>no</td><td>the last date the model was allowed to learn from</td></tr>
<tr><td><code>model_sha256</code> or <code>source_sha256</code></td><td>no, but expected</td><td>the hash of the frozen weights, or of the script for a formula</td></tr>
</table>

## The rules that make the score mean something

* **The signal on date t may use nothing observed after the close of t.** The truncation test in `beatnothing.canary` is how to prove that for your own feature code.
* **Flat when silent.** A day with no signal is a day in cash, not a day removed from your record. You cannot submit only on the days you liked the look of and have your Sharpe computed on that subset.
* **Gross exposure never exceeds one.** Long only unless you declare `long_short`, in which case the top decile is held long, the bottom decile short, dollar neutral, and the short book pays 50 basis points a year to borrow. A dollar neutral book is judged against cash, because that is what it competes with.
* **Universe selection is part of the signal.** State how the universe was chosen and on which date. `beatnothing.universe` tells you who was actually in the index on any day.
* **Frozen means frozen.** A resubmission is a new contestant with a new registration date.

## How cheating is caught

Every entry is checked by a machine before a human looks at it, on every pull request:

```bash
python scripts/validate_submissions.py --folder submissions_pit/my_model
```

Structure first: columns, types, duplicates, missing values, the engine's exposure
rules, and metadata complete enough to audit later. Then plausibility, which is what
catches a leak. A cross sectional signal on daily equity returns has an information
coefficient of roughly 0.02 to 0.05; that is what the published record says and what
every honest contestant here produces. A submission whose rank correlation with the next
day return is an order of magnitude larger has not found something the field missed. The
same applies to the net Sharpe ratio: above about three, on a long only book of large cap
equities, the explanation is almost never skill.

The thresholds are deliberately loose, so that nothing real is refused. A flagged entry
is not banned; it is held for a human, and the reason is printed. The suite in
`tests/test_validate.py` proves the check has teeth by feeding it the leakage canaries:
the contestant that peeks at tomorrow's return cannot reach the board.

Its first run refused four of the maintainer's own factor submissions, for names too
young to have a momentum score being written as blanks instead of left out. That is the
point of having it.

## Reference contestants

The two scripts here show the full path from raw prices to a submission:
`kronos_zero_shot.py` runs a pretrained financial foundation model with no training at
all, and `costaware_net.py` trains a network end to end on net Sharpe with a turnover
penalty across three seeds. `../scripts/factor_contestants.py` builds the classic factors
from prices alone, and `../scripts/score_frozen_on_pit.py` rescores frozen models on the
point in time universe.
