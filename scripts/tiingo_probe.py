"""
Does Tiingo's free tier serve daily history for the index leavers Yahoo has deleted?

Reads the token from TIINGO_API_KEY or ~/.secrets/tiingo_api_key.txt (never from inside
this repository). Spends at most a handful of requests: the ticker metadata and the daily
history of a few known delisted names, then reports coverage against the 47 missing
leavers listed in leaderboard/coverage_report.json.

    python scripts/tiingo_probe.py            # probe five names
    python scripts/tiingo_probe.py --all      # probe every missing leaver (47 requests, one hour of the free quota)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = Path.home() / ".secrets" / "tiingo_api_key.txt"
BASE = "https://api.tiingo.com"


def token() -> str:
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not key or key.startswith("PASTE_"):
        raise SystemExit(f"no token: set TIINGO_API_KEY or paste the token into {KEY_FILE}")
    return key


def get(path: str, key: str, **params):
    r = requests.get(f"{BASE}{path}", params=params, headers={"Authorization": f"Token {key}"}, timeout=60)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:200])


def main(probe_all: bool) -> None:
    key = token()
    gone = json.loads((ROOT / "leaderboard" / "coverage_report.json").read_text(encoding="utf-8"))["left_and_unavailable_tickers"]
    names = gone if probe_all else ["SIVB", "ATVI", "TWTR", "PXD", "FRC"]
    rows = []
    for t in names:
        code, meta = get(f"/tiingo/daily/{t}", key)
        code2, hist = get(f"/tiingo/daily/{t}/prices", key, startDate="2005-01-01", format="json")
        n = len(hist) if isinstance(hist, list) else 0
        first = hist[0]["date"][:10] if n else ""
        last = hist[-1]["date"][:10] if n else ""
        rows.append({"ticker": t, "meta_status": code, "end_date": (meta.get("endDate") if isinstance(meta, dict) else None),
                     "rows": n, "first": first, "last": last})
        print(f"{t:6s} meta {code} history {code2} rows {n:5d} {first} -> {last}", flush=True)
        time.sleep(1.5)                      # stay under 50 req/hour
    df = pd.DataFrame(rows)
    covered = df[df["rows"] > 250]["ticker"].tolist()
    print(f"\nfull histories returned for {len(covered)} of {len(names)} probed names")
    (ROOT / "leaderboard" / "tiingo_coverage.json").write_text(
        json.dumps({"probed": names, "covered": covered, "detail": rows}, indent=2), encoding="utf-8")
    print("wrote leaderboard/tiingo_coverage.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    main(ap.parse_args().all)
