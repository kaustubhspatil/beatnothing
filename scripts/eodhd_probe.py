"""
Probe how much of the survivorship gap a paid source would close, before paying for it.

Reads the EODHD API key from the environment variable EODHD_API_KEY or from the file
~/.secrets/eodhd_api_key.txt (never from inside this repository), then:

  1. lists the delisted US symbols the account can see,
  2. checks which of the index leavers that Yahoo no longer serves are among them,
  3. downloads one full daily history as a sample and prints its span.

Nothing is written into the repository except an optional JSON report under leaderboard/.

    python scripts/eodhd_probe.py            # probe only
    python scripts/eodhd_probe.py --report   # also write leaderboard/eodhd_coverage.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = Path.home() / ".secrets" / "eodhd_api_key.txt"
BASE = "https://eodhd.com/api"


def api_key() -> str:
    key = os.environ.get("EODHD_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not key or key.startswith("PASTE_"):
        raise SystemExit(f"no key: set EODHD_API_KEY or paste the key into {KEY_FILE}")
    return key


def get(path: str, key: str, **params):
    params.update({"api_token": key, "fmt": "json"})
    r = requests.get(f"{BASE}/{path}", params=params, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"{path}: HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def main(write_report: bool) -> None:
    key = api_key()
    gone = json.loads((ROOT / "leaderboard" / "coverage_report.json").read_text(encoding="utf-8"))["left_and_unavailable_tickers"]
    print(f"index leavers with no Yahoo prices: {len(gone)}")

    delisted = pd.DataFrame(get("exchange-symbol-list/US", key, delisted=1))
    print(f"delisted US symbols visible to this account: {len(delisted):,}")
    codes = set(delisted["Code"].astype(str)) if len(delisted) else set()
    covered = sorted(t for t in gone if t in codes)
    missing = sorted(t for t in gone if t not in codes)
    print(f"covered by EODHD delisted list: {len(covered)} of {len(gone)}")
    print("still missing:", missing)

    sample = covered[0] if covered else None
    span = None
    if sample:
        hist = pd.DataFrame(get(f"eod/{sample}.US", key, period="d"))
        if len(hist):
            span = [str(hist["date"].min()), str(hist["date"].max()), int(len(hist))]
            print(f"sample {sample}: {span[2]:,} daily rows, {span[0]} to {span[1]}")

    if write_report:
        out = {"leavers_unavailable_on_yahoo": len(gone), "covered_by_eodhd": covered, "still_missing": missing,
               "delisted_symbols_visible": int(len(delisted)), "sample": {"ticker": sample, "span": span}}
        (ROOT / "leaderboard" / "eodhd_coverage.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("wrote leaderboard/eodhd_coverage.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    main(ap.parse_args().report)
