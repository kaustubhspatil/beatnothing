"""
Check every submission, or one, against the rules. Exits non zero if any fails, so it
can gate a pull request.

    python scripts/validate_submissions.py                       # both tracks
    python scripts/validate_submissions.py --track pit
    python scripts/validate_submissions.py --folder submissions_pit/momentum_12_1_long_short
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from beatnothing.leaderboard import TRACKS, load_actual
from beatnothing.validate import validate_all, validate_submission

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=list(TRACKS) + ["all"], default="all")
    ap.add_argument("--folder", default=None, help="check a single submission folder")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    reports = []
    if args.folder:
        folder = Path(args.folder)
        track = next((t for t, c in TRACKS.items() if c["submissions"] in folder.parts), "survivor48")
        actual_path = ROOT / TRACKS[track]["actual"]
        actual = load_actual(actual_path) if actual_path.exists() else None
        reports.append(validate_submission(folder, actual))
    else:
        for track, cfg in TRACKS.items():
            if args.track not in ("all", track):
                continue
            sub_dir = ROOT / cfg["submissions"]
            if not sub_dir.exists():
                continue
            actual_path = ROOT / cfg["actual"]
            if not actual_path.exists():
                print(f"skipping {track}: {cfg['actual']} is not built, so only structure can be checked")
            for rep in validate_all(sub_dir, actual_path if actual_path.exists() else None):
                rep.name = f"{track}/{rep.name}"
                reports.append(rep)

    if args.as_json:
        print(json.dumps([r.to_dict() for r in reports], indent=1))
    else:
        for rep in reports:
            print(rep.render())
    failed = [r.name for r in reports if not r.ok]
    warned = [r.name for r in reports if r.ok and r.warnings]
    print(f"\n{len(reports)} submissions checked, {len(failed)} failed, {len(warned)} with warnings")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
