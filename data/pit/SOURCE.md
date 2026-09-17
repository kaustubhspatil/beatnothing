# Point in time S&P 500 membership

Both files come from the MIT licensed repository https://github.com/fja05680/sp500
(snapshot taken 17 September 2026).

* `sp500_membership_by_date.csv` lists the full constituent set on every date the index
  changed, from 1996 to 18 August 2026.
* `sp500_ticker_start_end.csv` lists each ticker's entry and exit dates.
* `still_listed_leavers.json` records which of the members on 31 December 2021 that later
  left the index still returned prices from Yahoo Finance on 17 September 2026. The
  complement (47 names) is the residual survivorship gap reported on the leaderboard page.

The maintainer cross checks Wikipedia's change log every couple of months; early years
are less reliable than recent ones, and the index rarely holds exactly 500 names.
