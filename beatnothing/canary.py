"""
Leak canaries: contestants that cheat on purpose.

Run them through your evaluation pipeline. If they do not score absurdly well,
your pipeline is not measuring what you think it is. Verify the verifier.

Also here: the truncation test that proves features are trailing only. Delete
everything after a cutoff, recompute, and the surviving rows must be identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def peek(actual: pd.DataFrame) -> pd.DataFrame:
    """Uses the return of t+1 as the prediction on t. The loudest possible leak."""
    return actual.copy()


def off_by_one(feature: pd.DataFrame) -> pd.DataFrame:
    """A trailing feature shifted the wrong way (the classic misaligned index):
    the value observed on t+1 is used on t."""
    return feature.shift(-1)


def hindsight_universe(actual: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Always long the `top_n` names with the highest realised return over the whole
    window. Survivorship and selection bias in one line."""
    total = (1 + actual.fillna(0)).prod()
    winners = total.sort_values(ascending=False).index[:top_n]
    pred = pd.DataFrame(0.0, index=actual.index, columns=actual.columns)
    pred[winners] = 1.0
    return pred


def truncation_test(build_features, prices: pd.DataFrame, cutoff, keys=("Date", "Ticker"),
                    atol: float = 1e-9) -> dict:
    """
    Prove features are trailing only. `build_features(prices)` must return a long
    frame keyed by `keys`. Rows on or before `cutoff` must be identical whether or
    not data after `cutoff` exists. The target column is excluded from the verdict
    because it is the one deliberate forward look (it needs t+1 to exist).
    """
    full = build_features(prices)
    trunc = build_features(prices[prices["Date"] <= pd.Timestamp(cutoff)])
    full = full[full["Date"] <= pd.Timestamp(cutoff)]
    merged = full.merge(trunc, on=list(keys), suffixes=("_full", "_trunc"))
    cols = [c for c in full.columns if c not in keys]
    worst = {}
    for c in cols:
        a = merged[f"{c}_full"].to_numpy(float)
        b = merged[f"{c}_trunc"].to_numpy(float)
        mask = np.isfinite(a) & np.isfinite(b)
        worst[c] = float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0
    return {"rows_compared": int(len(merged)), "max_abs_diff": worst,
            "passed": bool(all(v <= atol for c, v in worst.items() if c != "target"))}
