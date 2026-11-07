#!/usr/bin/env python
"""Data for the weekly-shape animation.

What the animation illustrates
------------------------------
The construction of the embedding: a daily series becomes a vector of 7 numbers (the
share of a week carried by each day), then a point in a 2D plane where families of
shapes separate.

**The families are the planted archetypes of notebook 10, step 1** -- the positive
control -- and not shapes invented for the video. That is what makes the animation
honest: it shows what a discrete structure *would* look like, at a measured separation
above the detection floor, while the post states what the real data turns out to be. A
base profile taken from M5 carries the archetypes, the weekly totals are the real ones,
and the counting noise is redrawn as it is in the audit.

Three files, the contract the scene reads
-----------------------------------------
    weekly_series.parquet        series_id, date, value      the series explained on screen
    archetype_embeddings.parquet series_id, archetype, v0..v6
    archetype_umap.parquet       series_id, archetype, umap_x, umap_y

Invariant checked here: the stored vector of the explained series is **exactly** the one
the scene obtains by stacking the weeks it displays. Otherwise the animation would show
one thing and print the numbers of another.

    uv run --extra viz python scripts/make_manim_data.py
"""

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from m5_chronos2.data import ARTIFACTS_DIR, roster_name
from m5_chronos2.evaluation import make_windows
from m5_chronos2.shapes import (WEEKDAYS, positive_control, separation, shape_matrix,
                                umap_layout, weekly_share_array, weekly_shapes)

SAMPLE_SIZE = 500
SEED = 42

# How many weeks the scene counts through. The displayed vector is the aggregate of
# those weeks, so this number belongs as much to the data as to the animation.
N_WEEKS = 20

N_ARCHETYPES = 3          # one colour, one pass per archetype in the animation
SEPARATION = 0.8          # well above the detection floor measured in notebook 10
MIN_WEEKS = 26

ARCHETYPE_NAMES = ["weekend", "weekday", "midweek"]


def base_profile_and_totals(roster: str, last_day):
    """The real mean profile and the real weekly totals: what the simulation rests on."""
    sales = pl.read_parquet(ARTIFACTS_DIR / f"sales_flags_{roster}.parquet")
    shapes_table = weekly_shapes(sales, last_day=last_day, min_weeks=MIN_WEEKS)
    ids, blocks, totals = weekly_share_array(sales, last_day=last_day)

    position = {series_id: index for index, series_id in enumerate(ids)}
    rows = [position[series_id] for series_id in shapes_table["id"].to_list()]
    totals = [totals[row] for row in rows]

    base = shape_matrix(shapes_table).mean(axis=0)
    return base / base.sum(), totals


def example_series(profile: np.ndarray, weekly_totals: np.ndarray,
                   generator: np.random.Generator):
    """Draw N_WEEKS weeks of daily sales, and the vector they give.

    The vector is recomputed from the drawn weeks rather than copied from the latent
    profile: it is the same double normalisation the study uses everywhere, applied to
    the numbers the scene will actually stack.
    """
    totals = np.maximum(np.rint(weekly_totals[:N_WEEKS]), 1).astype(np.int64)
    if len(totals) < N_WEEKS:                      # short series: loop its totals
        totals = np.resize(totals, N_WEEKS)
    counts = generator.multinomial(totals, profile)          # N_WEEKS x 7

    shares = counts / counts.sum(axis=1, keepdims=True)      # L1 per week
    vector = shares.mean(axis=0)                             # then the mean, which sums to 1
    return counts, vector / vector.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="viz/data",
                        help="directory the scene reads its parquets from")
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()

    out = Path(arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(arguments.seed)

    roster = roster_name(SAMPLE_SIZE, SEED)
    last_day = make_windows().filter(pl.col("role") == "validation")["origin"][0]
    base, totals = base_profile_and_totals(roster, last_day)
    print(f"real base profile: {np.round(base, 3)}")
    print(f"{len(totals)} series, median {np.median([t.mean() for t in totals]):.1f} units/week")

    simulation = positive_control(base, totals, n_clusters=N_ARCHETYPES,
                                  sep=SEPARATION, seed=arguments.seed)
    shapes = simulation.shapes.copy()
    print(f"{N_ARCHETYPES} archetypes, measured separation delta = {simulation.separation:.2f}")

    # The explained series: its weeks are drawn, so its vector has to be rebuilt from
    # them and put back into the matrix BEFORE the projection, otherwise its point would
    # not be the point of the series shown.
    example = int(np.flatnonzero(simulation.labels == 0)[0])
    counts, vector = example_series(simulation.profiles[example], totals[example], generator)
    shapes[example] = vector

    coordinates, method = umap_layout(shapes, seed=arguments.seed)
    print(f"projection: {method}")

    ids = [f"S{index:04d}" for index in range(len(shapes))]
    names = [ARCHETYPE_NAMES[label] for label in simulation.labels]
    example_id = ids[example]

    # --- the invariant --------------------------------------------------------------
    redone = counts / counts.sum(axis=1, keepdims=True)
    redone = redone.mean(axis=0)
    redone = redone / redone.sum()
    error = float(np.abs(redone - shapes[example]).max())
    assert error < 1e-12, f"the stored vector does not come from the weeks written ({error:.1e})"
    assert abs(shapes[example].sum() - 1) < 1e-12
    print(f"invariant: stored vector == aggregate of the {N_WEEKS} weeks written "
          f"(difference {error:.1e})")

    # --- writing --------------------------------------------------------------------
    start = last_day - timedelta(days=N_WEEKS * WEEKDAYS - 1)
    dates = pl.date_range(start, last_day, interval="1d", eager=True)
    assert len(dates) == N_WEEKS * WEEKDAYS
    series = pl.DataFrame({
        "series_id": [example_id] * (N_WEEKS * WEEKDAYS),
        "date": dates,
        "value": counts.reshape(-1).astype(np.int64),
    })
    series.write_parquet(out / "weekly_series.parquet")

    embeddings = pl.DataFrame(
        {"series_id": ids, "archetype": names}
        | {f"v{d}": shapes[:, d] for d in range(WEEKDAYS)}
    )
    embeddings.write_parquet(out / "archetype_embeddings.parquet")

    pl.DataFrame({
        "series_id": ids, "archetype": names,
        "umap_x": coordinates[:, 0].astype(float),
        "umap_y": coordinates[:, 1].astype(float),
    }).write_parquet(out / "archetype_umap.parquet")

    print(f"\nwritten to {out}:")
    print(f"  weekly_series.parquet         {series.height} days of {example_id} "
          f"({names[example]})")
    print(f"  archetype_embeddings.parquet  {embeddings.height} series x {WEEKDAYS}")
    print(f"  archetype_umap.parquet        {method}, "
          f"{', '.join(f'{n}={names.count(n)}' for n in ARCHETYPE_NAMES[:N_ARCHETYPES])}")
    print(f"\nthe scene reads SERIES_ID = {example_id}")


if __name__ == "__main__":
    main()
