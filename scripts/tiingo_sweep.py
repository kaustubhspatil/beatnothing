"""
Paced sweep of Tiingo's daily history for every index leaver that Yahoo has deleted.

Free tier: 50 requests an hour, 1,000 a day, 500 unique symbols a month. One request
per name (prices only), spaced so that a full hour never exceeds the quota, with a
long pause and retry on a 429. Results are appended to
leaderboard/tiingo_coverage_full.json after every name, so a killed run loses nothing.

    python scripts/tiingo_sweep.py                    # every missing leaver not yet probed
    python scripts/tiingo_sweep.py --spacing 100      # seconds between requests (default 100)
    python scripts/tiingo_sweep.py --save-prices      # also cache each history under data/raw/tiingo/
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
OUT = ROOT / "leaderboard" / "tiingo_coverage_full.json"
CACHE = ROOT / "data" / "raw" / "tiingo"
# names Tiingo keeps under the symbol they traded under after leaving the index
ALIASES = {"FRC": "FRCB"}


def token() -> str:
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not key or key.startswith("PASTE_"):
        raise SystemExit(f"no token: set TIINGO_API_KEY or paste the token into {KEY_FILE}")
    return key


def fetch(t: str, key: str):
    r = requests.get(f"https://api.tiingo.com/tiingo/daily/{t}/prices",
                     params={"startDate": "2005-01-01", "format": "json"},
                     headers={"Authorization": f"Token {key}"}, timeout=60)
    if r.status_code == 429:
        return 429, None
    if r.status_code != 200:
        return r.status_code, None
    return 200, r.json()


def main(spacing: float, save_prices: bool, tickers: str | None = None) -> None:
    key = token()
    gone = ([t.strip() for t in tickers.split(",") if t.strip()] if tickers else
            json.loads((ROOT / "leaderboard" / "coverage_report.json").read_text(encoding="utf-8"))["left_and_unavailable_tickers"])
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    seed = ROOT / "leaderboard" / "tiingo_coverage.json"
    if seed.exists():                       # fold in the five name probe so they are not fetched twice
        for row in json.loads(seed.read_text(encoding="utf-8"))["detail"]:
            done.setdefault(row["ticker"], {"status": 200 if row["rows"] else 404, "rows": row["rows"],
                                            "first": row["first"], "last": row["last"]})
    todo = [t for t in gone if t not in done or done[t].get("status") == 429 or (save_prices and not (CACHE / f"{t}.csv").exists() and done[t]["rows"] >= 250)]
    print(f"{len(gone)} missing leavers, {len(done)} already probed, {len(todo)} to fetch, {spacing:.0f}s apart", flush=True)
    if save_prices:
        CACHE.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(todo, 1):
        while True:                      # a rate limit is never a failure, only a wait
            code, rows = fetch(ALIASES.get(t, t), key)
            if code != 429:
                break
            print(f"{t}: rate limited, sleeping 16 minutes", flush=True)
            time.sleep(16 * 60)
        n = len(rows) if rows else 0
        first = rows[0]["date"][:10] if n else ""
        last = rows[-1]["date"][:10] if n else ""
        done[t] = {"status": code, "rows": n, "first": first, "last": last}
        if save_prices and n:
            pd.DataFrame(rows).to_csv(CACHE / f"{t}.csv", index=False)
        OUT.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(f"[{i}/{len(todo)}] {t:6s} {code} rows {n:5d} {first} -> {last}", flush=True)
        if i < len(todo):
            time.sleep(spacing)
    covered = sorted(t for t, d in done.items() if d["rows"] >= 250)
    print(f"\nfull histories for {len(covered)} of {len(gone)}: missing {sorted(set(gone) - set(covered))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=100.0)
    ap.add_argument("--save-prices", action="store_true")
    ap.add_argument("--tickers", default=None, help="comma separated list instead of the coverage report")
    a = ap.parse_args()
    main(a.spacing, a.save_prices, a.tickers)
