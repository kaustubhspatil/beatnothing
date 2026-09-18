import json

import numpy as np
import pandas as pd
import pytest

from beatnothing.canary import hindsight_universe, peek
from beatnothing.validate import validate_submission


def _actual(n=400, k=30, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(rng.normal(0.0004, 0.014, (n, k)), index=idx, columns=[f"S{i}" for i in range(k)])


def _write(tmp_path, folder_name, wide, **meta_extra):
    d = tmp_path / folder_name
    d.mkdir(parents=True, exist_ok=True)
    # melt, not stack: older pandas drops missing values when stacking, which would quietly
    # remove the very hole that test_missing_values_are_refused is checking for
    long = wide.rename_axis("Date").reset_index().melt(id_vars="Date", var_name="Ticker", value_name="value")
    long.to_parquet(d / "signal.parquet", index=False)
    meta = {"name": folder_name, "kind": "predictions", "description": "a test entry",
            "registered": "2026-09-18", "model_sha256": "0" * 64}
    meta.update(meta_extra)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def test_an_honest_submission_passes(tmp_path):
    a = _actual()
    rng = np.random.default_rng(9)
    honest = pd.DataFrame(rng.normal(size=a.shape), index=a.index, columns=a.columns) * 0.001
    rep = validate_submission(_write(tmp_path, "honest", honest), a)
    assert rep.ok and not rep.errors


def test_the_peek_canary_is_rejected(tmp_path):
    """The loudest leak in the harness must not be able to reach the board."""
    a = _actual()
    rep = validate_submission(_write(tmp_path, "peek", peek(a)), a)
    assert not rep.ok
    assert any("information coefficient" in e for e in rep.errors)
    assert any("Sharpe" in e for e in rep.errors)


def test_the_hindsight_universe_canary_is_caught(tmp_path):
    a = _actual(n=600, k=40)
    rep = validate_submission(_write(tmp_path, "hindsight", hindsight_universe(a, top_n=4)), a)
    assert rep.warnings or not rep.ok       # picking the winners in advance shows up as an implausible record


def test_missing_values_are_refused(tmp_path):
    a = _actual()
    sig = pd.DataFrame(1.0, index=a.index, columns=a.columns)
    sig.iloc[10, 3] = np.nan
    rep = validate_submission(_write(tmp_path, "holes", sig), a)
    assert not rep.ok and any("missing or infinite" in e for e in rep.errors)


def test_a_short_record_is_refused(tmp_path):
    a = _actual(n=60)
    sig = pd.DataFrame(1.0, index=a.index, columns=a.columns)
    rep = validate_submission(_write(tmp_path, "brief", sig), a)
    assert not rep.ok and any("trading days" in e for e in rep.errors)


def test_leveraged_weights_are_refused(tmp_path):
    a = _actual()
    w = pd.DataFrame(0.2, index=a.index, columns=a.columns)     # 30 names at 0.2 is six times the book
    rep = validate_submission(_write(tmp_path, "levered", w, kind="weights"), a)
    assert not rep.ok and any("gross exposure" in e for e in rep.errors)


def test_short_weights_need_the_right_rule(tmp_path):
    a = _actual()
    w = pd.DataFrame(-0.01, index=a.index, columns=a.columns)
    rep = validate_submission(_write(tmp_path, "sneaky_short", w, kind="weights"), a)
    assert not rep.ok and any("long_short" in e for e in rep.errors)
    ok = validate_submission(_write(tmp_path, "declared_short", w, kind="weights", rule="long_short"), a)
    assert ok.ok


def test_metadata_must_be_complete_and_consistent(tmp_path):
    a = _actual()
    sig = pd.DataFrame(1.0, index=a.index, columns=a.columns)
    rep = validate_submission(_write(tmp_path, "mislabelled", sig, name="something_else"), a)
    assert not rep.ok and any("does not match the folder" in e for e in rep.errors)
    rep = validate_submission(_write(tmp_path, "no_rule", sig, rule="long_only_please"), a)
    assert not rep.ok and any("rule" in e for e in rep.errors)
    rep = validate_submission(_write(tmp_path, "undated", sig, registered=None), a)
    assert not rep.ok


def test_structure_is_checked_without_the_returns_file(tmp_path):
    a = _actual()
    sig = pd.DataFrame(1.0, index=a.index, columns=a.columns)
    rep = validate_submission(_write(tmp_path, "structure_only", sig), None)
    assert rep.ok and "information_coefficient" not in rep.facts


def test_a_missing_folder_fails_cleanly(tmp_path):
    rep = validate_submission(tmp_path / "nothing_here", None)
    assert not rep.ok and len(rep.errors) == 2
