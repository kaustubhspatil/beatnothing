"""
Check a submission before it reaches the leaderboard.

Anyone can open a pull request, so the board is only worth reading if a machine checks
every entry. This module is that check, and it runs in continuous integration on any
change under a submissions folder.

Structure comes first: the right columns and types, one value per date and ticker, no
missing or infinite numbers, weights inside the rules the engine enforces, and metadata
complete enough that the entry can be audited later. Then plausibility, which is the
part that catches a leak. A cross sectional signal on daily equity returns has an
information coefficient of roughly 0.02 to 0.05; that is what decades of published work
report and what every honest contestant here produces. A submission whose rank
correlation with the next day return is an order of magnitude larger has not found
something the field missed, it has seen the answer. The same logic applies to the net
Sharpe ratio itself: above about three, on a long only book of large cap equities, the
explanation is almost never skill.

These thresholds reject nothing that is real and catch the accidents that matter, which
is the trade a benchmark should want. A flagged submission is not banned; it is held
for a human to look at, and the reason is printed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import RULES

REQUIRED_META = ("name", "kind", "description", "registered")
IC_WARN, IC_FAIL = 0.08, 0.15
SHARPE_WARN, SHARPE_FAIL = 2.5, 5.0
MIN_DAYS = 120


class Report:
    """Errors block a submission, warnings ask a human to look."""

    def __init__(self, name: str):
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.facts: dict = {}

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "errors": self.errors,
                "warnings": self.warnings, "facts": self.facts}

    def render(self) -> str:
        head = f"{'PASS' if self.ok else 'FAIL'}  {self.name}"
        lines = [head] + [f"   error:   {e}" for e in self.errors] + [f"   warning: {w}" for w in self.warnings]
        for k, v in self.facts.items():
            lines.append(f"   {k}: {v}")
        return "\n".join(lines)


def _information_coefficient(signal: pd.DataFrame, actual: pd.DataFrame) -> float:
    """Mean daily cross sectional rank correlation between the signal and the next day return."""
    s, a = signal.align(actual, join="inner")
    both = s.notna() & a.notna()
    ics = []
    for day in s.index:
        row_s, row_a = s.loc[day][both.loc[day]], a.loc[day][both.loc[day]]
        if len(row_s) >= 10 and row_s.nunique() > 1:
            ics.append(row_s.rank().corr(row_a.rank()))
    return float(np.nanmean(ics)) if ics else float("nan")


def validate_submission(folder: Path, actual: pd.DataFrame | None = None) -> Report:
    """Check one submissions/<name>/ folder. Pass `actual` to enable the plausibility checks."""
    folder = Path(folder)
    rep = Report(folder.name)

    meta_path, sig_path = folder / "meta.json", folder / "signal.parquet"
    if not meta_path.exists():
        rep.error("meta.json is missing")
    if not sig_path.exists():
        rep.error("signal.parquet is missing")
    if not rep.ok:
        return rep

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        rep.error(f"meta.json is not valid JSON: {exc}")
        return rep

    for key in REQUIRED_META:
        if not meta.get(key):
            rep.error(f"meta.json is missing '{key}'")
    if meta.get("kind") not in ("predictions", "weights"):
        rep.error("meta.json 'kind' must be 'predictions' or 'weights'")
    rule = meta.get("rule", "long_flat")
    if rule not in RULES:
        rep.error(f"meta.json 'rule' must be one of {RULES}")
    q = float(meta.get("quantile", 0.1))
    if not 0 < q <= 0.5:
        rep.error("meta.json 'quantile' must be above zero and at most one half")
    if meta.get("name") and meta["name"] != folder.name:
        rep.error(f"meta.json name '{meta['name']}' does not match the folder '{folder.name}'")
    if not (meta.get("model_sha256") or meta.get("source_sha256")):
        rep.warn("no model_sha256 or source_sha256: the entry cannot be tied to a frozen artifact later")

    # Frozen means frozen. The signal file's own hash pins the content, and because an
    # entry arrives by pull request, the public git history dates it: the two together
    # are a registration nobody can move afterwards, with no third party involved.
    digest = hashlib.sha256(sig_path.read_bytes()).hexdigest()
    rep.facts["signal_sha256"] = digest[:16] + "..."
    claimed = meta.get("signal_sha256")
    if claimed and claimed != digest:
        rep.error(f"signal.parquet does not match the signal_sha256 in meta.json "
                  f"(file {digest[:16]}..., claimed {str(claimed)[:16]}...)")
    elif not claimed:
        rep.warn("no signal_sha256 in meta.json: run scripts/validate_submissions.py --write-hash to pin it")
    try:
        pd.Timestamp(meta.get("registered"))
    except (ValueError, TypeError):
        rep.error("meta.json 'registered' is not a date")

    try:
        sig = pd.read_parquet(sig_path)
    except Exception as exc:                                    # noqa: BLE001
        rep.error(f"signal.parquet cannot be read: {exc}")
        return rep

    if list(sig.columns) != ["Date", "Ticker", "value"]:
        rep.error(f"signal.parquet columns must be exactly Date, Ticker, value; found {list(sig.columns)}")
        return rep
    if not pd.api.types.is_datetime64_any_dtype(sig["Date"]):
        rep.error("Date must be a datetime column")
    if not pd.api.types.is_numeric_dtype(sig["value"]):
        rep.error("value must be numeric")
    if not rep.ok:
        return rep

    if sig.duplicated(["Date", "Ticker"]).any():
        n = int(sig.duplicated(["Date", "Ticker"]).sum())
        rep.error(f"{n:,} duplicated (Date, Ticker) rows")
    if not np.isfinite(sig["value"]).all():
        rep.error(f"{int((~np.isfinite(sig['value'])).sum()):,} values are missing or infinite")
    if sig["Date"].nunique() < MIN_DAYS:
        rep.error(f"only {sig['Date'].nunique()} trading days; at least {MIN_DAYS} are needed to say anything")

    rep.facts["rows"] = f"{len(sig):,}"
    rep.facts["days"] = f"{sig['Date'].nunique():,}"
    rep.facts["tickers"] = f"{sig['Ticker'].nunique():,}"
    rep.facts["window"] = f"{sig['Date'].min().date()} to {sig['Date'].max().date()}"
    rep.facts["rule"] = rule

    wide = sig.pivot(index="Date", columns="Ticker", values="value").sort_index()
    if meta.get("kind") == "weights":
        if rule != "long_short" and (wide.fillna(0) < -1e-12).any().any():
            rep.error("negative weights need rule 'long_short'")
        gross = wide.abs().sum(axis=1)
        if (gross > 1.0 + 1e-6).any():
            rep.error(f"gross exposure reaches {gross.max():.3f}; the engine allows at most one")

    if rule in ("long_top", "long_short"):
        # a decile rule needs the signal to actually separate names; an early stopped tree
        # model can emit two or three distinct values a day and rank nothing
        distinct = wide.nunique(axis=1)
        names = wide.notna().sum(axis=1)
        share = float((distinct / names.replace(0, np.nan)).median())
        rep.facts["distinct_values_per_day"] = f"{distinct.median():.0f} of {names.median():.0f} names"
        if share < 2 * q:
            rep.warn(f"the signal takes only {distinct.median():.0f} distinct values across "
                     f"{names.median():.0f} names on a typical day, which is too coarse to pick a "
                     f"{q:.0%} tail cleanly; ties are broken by ticker order")

    if actual is None or not rep.ok:
        return rep

    ic = _information_coefficient(wide, actual)
    rep.facts["information_coefficient"] = f"{ic:+.4f}"
    if np.isfinite(ic):
        if abs(ic) >= IC_FAIL:
            rep.error(f"information coefficient {ic:+.3f} is far beyond anything published for daily equity "
                      f"returns (0.02 to 0.05). This is what a signal that has seen the answer looks like.")
        elif abs(ic) >= IC_WARN:
            rep.warn(f"information coefficient {ic:+.3f} is high enough to be worth a second look")

    from .engine import Backtest
    kw = {"predictions": wide, "rule": rule, "quantile": q} if meta["kind"] == "predictions" \
        else {"weights": wide, "allow_short": rule == "long_short"}
    try:
        stats = Backtest(actual, **kw).stats()
    except ValueError as exc:
        rep.error(f"the engine refused this submission: {exc}")
        return rep
    rep.facts["net_sharpe"] = f"{stats['net_sharpe']:+.2f}"
    rep.facts["annual_turnover"] = f"{stats['annual_turnover']:.1f}x"
    if stats["net_sharpe"] >= SHARPE_FAIL:
        rep.error(f"net Sharpe {stats['net_sharpe']:.2f} on large cap equities is not a result, it is a bug")
    elif stats["net_sharpe"] >= SHARPE_WARN:
        rep.warn(f"net Sharpe {stats['net_sharpe']:.2f} is high enough to deserve an explanation in the pull request")
    return rep


def validate_all(submissions_dir: Path, actual_path: Path | None = None) -> list[Report]:
    submissions_dir = Path(submissions_dir)
    actual = None
    if actual_path and Path(actual_path).exists():
        from .leaderboard import load_actual
        actual = load_actual(Path(actual_path))
    folders = sorted(p for p in submissions_dir.iterdir() if p.is_dir())
    return [validate_submission(f, actual) for f in folders]
