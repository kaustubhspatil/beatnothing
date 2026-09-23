import json

import numpy as np
import pandas as pd

from beatnothing import Membership, coverage_report, start_end_table
from beatnothing.cli import main


def test_membership_ships_with_the_package():
    m = Membership()
    on_lehman_weekend = m.on("2008-09-15")
    assert 480 <= len(on_lehman_weekend) <= 520
    assert "AAPL" in on_lehman_weekend
    assert m.first_date.year == 1996
    assert m.last_date.year >= 2026


def test_coverage_report_counts_leavers():
    m = Membership()
    rep = coverage_report(m, available_tickers=set(), start="2021-12-31", end="2026-09-17")
    assert rep["members_at_start"] == 505
    assert rep["left_during_window"] == 94
    assert rep["left_and_unavailable"] == 94          # nothing available, all leavers missing
    assert "SIVB" in rep["left_and_unavailable_tickers"]
    assert len(start_end_table()) > 1000


def test_cli_score_reports_net_edge(tmp_path, capsys):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=300)
    cols = [f"S{i}" for i in range(6)]
    actual = pd.DataFrame(rng.normal(0.0004, 0.012, (300, 6)), index=idx, columns=cols)
    long = actual.stack().rename("target").reset_index()
    long.columns = ["Date", "Ticker", "target"]
    long.to_parquet(tmp_path / "actual.parquet", index=False)
    sig = long.rename(columns={"target": "value"})
    sig["value"] = 1.0                                   # always long == bar
    sig.to_parquet(tmp_path / "signal.parquet", index=False)

    rc = main(["score", str(tmp_path / "signal.parquet"), "--actual", str(tmp_path / "actual.parquet"),
               "--n-boot", "100"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert abs(out["net_edge"]) < 1e-9
    assert out["clears_bar"] is False
    assert out["days"] == 300


def test_cli_members(capsys):
    assert main(["members", "2020-03-16"]) == 0
    text = capsys.readouterr().out
    assert "members on 2020-03-16" in text
    assert "AAPL" in text
