"""Scoring forecasts.

Every model writes its forecasts in the same format, one row per (series, day):
    id, origin, date, y_pred, q0.1, q0.2, ..., q0.9
y_pred is the point forecast (the median for probabilistic models); q0.1 ... q0.9
are the quantile forecasts. A model that only gives a point forecast copies
y_pred into every quantile column.

All metrics are pooled: errors are summed over every scored (series, day) pair.
"""

import polars as pl

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Horizon buckets, in days after the origin (both ends included).
# Zero-padded names, so that sorting the names alphabetically keeps them in time order.
HORIZON_BUCKETS = {
    "days 01-07": (1, 7),
    "days 08-14": (8, 14),
    "days 15-21": (15, 21),
    "days 22-28": (22, 28),
}


def quantile_column(level: float) -> str:
    """Name of the column holding the forecast of one quantile level, e.g. 0.1 -> 'q0.1'."""
    return f"q{level:.1f}"


def quantile_loss(level: float, column: str | None = None) -> pl.Expr:
    """Quantile (pinball) loss, row by row, of the forecast held in `column`.

    Under-prediction (y above the forecast) costs `level` per unit,
    over-prediction costs `1 - level` per unit.

    `column` defaults to that level's own quantile forecast, which is what WQL scores:
    the loss is then a proper scoring rule, minimised by the true quantile. Pass
    "y_pred" to price the same asymmetry on the point forecast instead.
    """
    error = pl.col("y") - pl.col(column or quantile_column(level))
    return pl.when(error >= 0).then(level * error).otherwise((level - 1) * error)


def metric_expressions() -> list[pl.Expr]:
    """The five metrics, plus the number of series they are computed on."""
    absolute_error = (pl.col("y") - pl.col("y_pred")).abs()
    total_actual = pl.col("y").abs().sum()

    # WQL: for each quantile level, 2 * total loss / total actual; then the mean over levels.
    wql_per_level = [2 * quantile_loss(level).sum() / total_actual for level in QUANTILE_LEVELS]

    return [
        pl.col("id").n_unique().alias("n_series"),
        absolute_error.mean().alias("MAE"),
        (100 * absolute_error.sum() / total_actual).alias("WMAPE"),
        absolute_error.quantile(0.9).alias("P90"),
        (pl.sum_horizontal(wql_per_level) / len(QUANTILE_LEVELS)).alias("WQL"),
        # RQL: the 0.2 asymmetry applied to the POINT forecast, so over-prediction costs
        # 4x under-prediction on the number one would actually act on.
        #
        # Scored on y_pred, not on q0.2. Scoring it on q0.2 made the metric mean two
        # different things: for a point-only model (every quantile column a copy of
        # y_pred) it was the asymmetric cost of the forecast, but for a model emitting
        # real quantiles it was a proper scoring rule for the 20th percentile -- a
        # deliberately low number that can hardly over-predict. On this data that gap is
        # a factor of ~2.75 at identical skill, so the column was not comparable down
        # its own rows. Corrected 2026-11-07; RQL figures produced before that date are
        # not comparable with these.
        (100 * quantile_loss(0.2, "y_pred").sum() / total_actual).alias("RQL"),
    ]


def add_horizon(scored: pl.DataFrame) -> pl.DataFrame:
    """Add 'horizon' (days after the origin, 1 = first forecast day) and 'horizon_bucket'."""
    scored = scored.with_columns(
        (pl.col("date") - pl.col("origin")).dt.total_days().alias("horizon")
    )
    bucket = pl.lit(None, dtype=pl.String)
    for name, (first_day, last_day) in HORIZON_BUCKETS.items():
        bucket = (
            pl.when(pl.col("horizon").is_between(first_day, last_day))
            .then(pl.lit(name))
            .otherwise(bucket)
        )
    return scored.with_columns(bucket.alias("horizon_bucket"))


def join_truth(predictions: pl.DataFrame, truth: pl.DataFrame, scored_series: pl.DataFrame) -> pl.DataFrame:
    """Put forecasts and actual values side by side, keeping only what must be scored.

    - truth: output of evaluation.actuals ('id', 'date', 'y', 'is_missing');
    - scored_series: the frozen evaluation roster ('origin', 'id' of every scored pair).
    """
    scored = predictions.join(scored_series.select("origin", "id"), on=["origin", "id"], how="inner")
    scored = scored.join(truth, on=["id", "date"], how="inner")
    scored = scored.filter(~pl.col("is_missing"))  # never score a day without truth
    return add_horizon(scored)


def score(scored: pl.DataFrame, by: str | list[str] | None = None) -> pl.DataFrame:
    """Compute the metrics on the output of join_truth, overall or per group (e.g. by='horizon_bucket')."""
    if by is None:
        return scored.select(metric_expressions())
    return scored.group_by(by).agg(metric_expressions()).sort(by)


METRICS = ["MAE", "WMAPE", "P90", "WQL", "RQL"]


def compare(scored: dict[str, pl.DataFrame], by: str | list[str] | None = None,
            reference: str | None = None) -> pl.DataFrame:
    """Scores of several models side by side: one row per model (and per group of `by`).

    scored: {model name -> output of join_truth}. With `reference`, adds for every metric
    the change relative to that model, in % (negative = lower error than the reference).
    """
    tables = []
    for model, table in scored.items():
        result = score(table, by=by).with_columns(pl.lit(model).alias("model"))
        tables.append(result.select(["model"] + result.columns[:-1]))
    table = pl.concat(tables)
    if reference is None:
        return table

    keys = [] if by is None else ([by] if isinstance(by, str) else list(by))
    reference_values = table.filter(pl.col("model") == reference).select(
        keys + [pl.col(metric).alias(f"{metric}_reference") for metric in METRICS]
    )
    table = table.join(reference_values, on=keys, how="left") if keys else table.join(reference_values, how="cross")
    table = table.with_columns(
        [(100 * (pl.col(m) / pl.col(f"{m}_reference") - 1)).alias(f"{m}_change_%") for m in METRICS]
    )
    return table.drop([f"{m}_reference" for m in METRICS])
