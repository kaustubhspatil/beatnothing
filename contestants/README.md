# Submitting a contestant

A contestant is a folder under `submissions/<name>/` with two files.

**`signal.parquet`**, long format, three columns:

<table>
<tr><th>column</th><th>meaning</th></tr>
<tr><td><code>Date</code></td><td>the decision date; the signal is formed from information available at that close</td></tr>
<tr><td><code>Ticker</code></td><td>the stock</td></tr>
<tr><td><code>value</code></td><td>a prediction (any real number; positive means long) or a weight in [0, 1]</td></tr>
</table>

**`meta.json`** with `name`, `kind` (`predictions` or `weights`), `training_cutoff`,
`registered` (the date you froze the model; forward scoring counts only days after it),
a one paragraph `description`, and the sha256 of every frozen model file.

Rules that make the score mean something:

* The signal on date t may use nothing observed after the close of t. The truncation
  test in `beatnothing.canary` is how to prove that for your own feature code.
* Weights are long only and sum to at most one on every day. The engine refuses anything else.
* Predictions are turned into positions by one fixed rule: equal weight across every name
  with a positive value, cash otherwise. If you want position sizing, submit weights.
* Universe selection is part of the signal. State how the universe was chosen, and on which
  date. `beatnothing.universe` tells you who was actually in the index on any day.
* Frozen means frozen. A resubmission is a new contestant with a new registration date.

The two reference scripts in this folder show the full path from raw prices to a
submission: `kronos_zero_shot.py` (a pretrained financial foundation model used
without any training) and `costaware_net.py` (a network trained end to end on net
Sharpe with a turnover penalty, three seeds).
