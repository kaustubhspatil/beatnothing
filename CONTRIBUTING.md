# Contributing

There are three useful things you can do here, and they are listed in order of how much
the benchmark needs them.

## 1. Submit a contestant

The fastest path, no API key and no download, because the returns panel is in the repo:

```bash
git clone https://github.com/kaustubhspatil/beatnothing
cd beatnothing
pip install -e ".[dev]"
python scripts/starter_submission.py --name my_first_try --author "your handle"
beatnothing validate submissions_pit/my_first_try
```

That writes a real entry built from a five day reversal signal, and it loses, which is the
point: it shows you the shape of the files and the shape of the problem at the same time.
Now open `scripts/starter_submission.py`, replace `signal_from_returns` with your own idea,
run it again, and open a pull request with the folder.

You do not have to use that script. Any tool in any language may produce the two files;
`contestants/README.md` is the full specification. You do not have to share your model
either. The signal file, its sha256 and the date of the pull request are enough to freeze
an entry, and the forward track will score it on days that did not exist when you sent it.

What the machine checks before a human looks: the file structure, the engine's rules, no
missing or infinite values, at least 120 days, and a leakage smell test. An entry that
fails cannot merge. An information coefficient above 0.15 or a net Sharpe above 5 is
treated as a bug report against your pipeline, not as a discovery, because on daily US
large caps it has always been one.

## 2. Break a result

More valuable than a new contestant. Every number on the board is produced by code in this
repository from data in this repository, so every number is attackable. If the statistics
are wrong, if the cost model flatters the bar, if a universe still contains hindsight, if
a test has the wrong size, open an issue with the "challenge a result" template. A
reproduction that overturns a published claim gets the claim changed and gets you credited
in the technical report. The benchmark's whole value is that this is possible.

The places I would attack first, in the author's own order:

- the cost model is a single flat 10 bps with no spread, no impact and no borrow scarcity
- the universe starts in 2013, so nothing here has been tested through a crisis
- the point in time membership is reconstructed from free sources, at 98% coverage
- one market, one horizon, one currency

## 3. Improve the harness

Tests, documentation, a second cost model, another market. Run `pytest` before you open
the pull request; CI runs the same suite on Python 3.10 through 3.13.

## House rules

Commit messages say what changed and why in plain sentences. No generated boilerplate, no
co-author trailers. Be accurate rather than promotional, including about your own entry.
