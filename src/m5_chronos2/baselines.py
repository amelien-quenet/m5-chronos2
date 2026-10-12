"""Three naive forecasts. They need no training and set the floor any real model must beat.

All return the standard forecast format (see metrics.py). They only give a point
forecast, so every quantile column is a copy of y_pred.
"""

from datetime import date, timedelta

import polars as pl

from m5_chronos2.data import HORIZON
from m5_chronos2.metrics import QUANTILE_LEVELS, quantile_column


def forecast_grid(series_ids: list[str], origin: date) -> pl.DataFrame:
    """Every (series, forecast day) pair of one window, with the horizon (1 to HORIZON)."""
    forecast_days = pl.date_range(
        origin + timedelta(days=1), origin + timedelta(days=HORIZON), interval="1d", eager=True
    ).alias("date")
    grid = pl.DataFrame({"id": series_ids}).join(forecast_days.to_frame(), how="cross")
    return grid.with_columns(
        pl.lit(origin).alias("origin"),
        (pl.col("date") - pl.lit(origin)).dt.total_days().alias("horizon"),
    )


def copy_point_to_quantiles(forecasts: pl.DataFrame) -> pl.DataFrame:
    """For point forecasts: every quantile column equals y_pred, then keep the standard columns."""
    forecasts = forecasts.with_columns(
        [pl.col("y_pred").alias(quantile_column(level)) for level in QUANTILE_LEVELS]
    )
    columns = ["id", "origin", "date", "y_pred"] + [quantile_column(level) for level in QUANTILE_LEVELS]
    return forecasts.select(columns).sort("id", "date")


def zero_forecast(series_ids: list[str], origin: date) -> pl.DataFrame:
    """Always predict 0 units."""
    grid = forecast_grid(series_ids, origin)
    grid = grid.with_columns(pl.lit(0.0).alias("y_pred"))
    return copy_point_to_quantiles(grid)


def seasonal_naive_forecast(sales: pl.DataFrame, series_ids: list[str], origin: date) -> pl.DataFrame:
    """Repeat the last observed week: each forecast day copies the same weekday of the last week."""
    grid = forecast_grid(series_ids, origin)

    # Horizon 1..7 -> look 7 days back, 8..14 -> 14 days back, etc.: always the last observed week.
    grid = grid.with_columns(
        (7 * ((pl.col("horizon") - 1) // 7 + 1)).alias("days_back")
    )
    grid = grid.with_columns(
        (pl.col("date") - pl.duration(days=pl.col("days_back"))).cast(pl.Date).alias("source_date")
    )

    source = sales.select("id", pl.col("date").alias("source_date"), pl.col("sales").cast(pl.Float64).alias("y_pred"))
    grid = grid.join(source, on=["id", "source_date"], how="left")
    return copy_point_to_quantiles(grid)


def moving_average_forecast(
    sales: pl.DataFrame, series_ids: list[str], origin: date, window_days: int = 28
) -> pl.DataFrame:
    """Predict, for every forecast day, the mean daily sales of the last `window_days` days."""
    recent = sales.filter(
        (pl.col("date") > origin - timedelta(days=window_days)) & (pl.col("date") <= origin)
    )
    recent_mean = recent.group_by("id").agg(pl.col("sales").mean().alias("y_pred"))

    grid = forecast_grid(series_ids, origin)
    grid = grid.join(recent_mean, on="id", how="left")
    return copy_point_to_quantiles(grid)
