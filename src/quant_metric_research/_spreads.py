from __future__ import annotations

import pandas as pd


def fractional_quantile_spread(
    scores: pd.Series, returns: pd.Series, *, quantiles: int
) -> float:
    """Compute a descriptive top-minus-bottom spread from validated finite pairs.

    Callers enforce cross-section size, score variation, and valid quantiles.
    Each bucket retains floor(n / quantiles) units of mass. If a score group
    crosses a boundary, every tied observation receives the same fraction of
    that group's bucket mass, independent of row order or security name.
    """
    groups = (
        pd.DataFrame({"score": scores.to_numpy(), "return": returns.to_numpy()})
        .groupby("score", sort=True)["return"]
        .agg(["mean", "size"])
    )
    count = int(groups["size"].sum())
    bucket_size = max(1, count // quantiles)
    group_end = groups["size"].cumsum()
    group_start = group_end - groups["size"]
    bottom_mass = (bucket_size - group_start).clip(lower=0).clip(upper=groups["size"])
    top_mass = (
        (group_end - (count - bucket_size)).clip(lower=0).clip(upper=groups["size"])
    )
    return float(((top_mass - bottom_mass) * groups["mean"]).sum() / bucket_size)
