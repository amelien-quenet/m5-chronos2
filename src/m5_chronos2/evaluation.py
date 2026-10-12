"""The evaluation protocol: where the forecasts start, and which series are scored.

Decided once, saved to disk, and never changed afterwards: every model of the
project is evaluated on exactly the same (series, window) pairs.
"""

from datetime import date, timedelta

import polars as pl

from m5_chronos2.data import HORIZON

# Last day of history before the final 28-day test window (d_1913).
TEST_ORIGIN = date(2016, 4, 24)

# One window earlier, for every choice we make (context length, fine-tuning...).
# The test window is only used for the final comparison.
VALIDATION_ORIGIN = TEST_ORIGIN - timedelta(days=HORIZON)


def make_windows() -> pl.DataFrame:
    """The forecast windows: one row per origin, with its role and its first/last forecast day."""
    windows = pl.DataFrame(
        {
            "role": ["validation", "test"],
            "origin": [VALIDATION_ORIGIN, TEST_ORIGIN],
        }
    )
    return windows.with_columns(
        (pl.col("origin") + pl.duration(days=1)).cast(pl.Date).alias("first_forecast_day"),
        (pl.col("origin") + pl.duration(days=HORIZON)).cast(pl.Date).alias("last_forecast_day"),
    )


def check_series(sales: pl.DataFrame, origin: date, min_history_days: int) -> pl.DataFrame:
    """For one origin, check each series against the three scoring conditions.

    Needs the columns 'id', 'date', 'sales', 'is_missing'.
    Returns one row per series with one boolean column per condition and 'is_scored'.

    The three conditions:
    1. enough history: at least `min_history_days` on-sale days up to the origin
       (otherwise there is too little to forecast from);
    2. no missing day in the forecast window (otherwise part of the truth is unknown);
    3. at least one sale in the forecast window (an all-zero window makes WMAPE and
       WQL divide by zero for that series).
    """
    window_end = origin + timedelta(days=HORIZON)
    history = sales.filter(pl.col("date") <= origin)
    window = sales.filter((pl.col("date") > origin) & (pl.col("date") <= window_end))

    history_stats = history.group_by("id").agg(
        (~pl.col("is_missing")).sum().alias("history_days_on_sale")
    )
    window_stats = window.group_by("id").agg(
        pl.col("is_missing").any().alias("window_has_missing"),
        pl.col("sales").sum().alias("window_total_sales"),
    )

    checks = history_stats.join(window_stats, on="id", how="inner")
    checks = checks.with_columns(
        (pl.col("history_days_on_sale") >= min_history_days).alias("enough_history"),
        (~pl.col("window_has_missing")).alias("window_fully_observed"),
        (pl.col("window_total_sales") > 0).alias("window_not_all_zero"),
    )
    checks = checks.with_columns(
        (
            pl.col("enough_history")
            & pl.col("window_fully_observed")
            & pl.col("window_not_all_zero")
        ).alias("is_scored"),
        pl.lit(origin).alias("origin"),
    )
    return checks.sort("id")


def actuals(sales: pl.DataFrame) -> pl.DataFrame:
    """The truth to compare forecasts with: 'id', 'date', 'y' (units sold), 'is_missing'."""
    return sales.select("id", "date", pl.col("sales").alias("y"), "is_missing")
