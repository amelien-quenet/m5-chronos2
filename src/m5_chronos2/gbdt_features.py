"""Features of the gradient-boosting baseline, reproduced from a public M5 solution.

Source: btrotta, "kaggle-m5" (top 4% of the M5 Accuracy competition),
https://github.com/btrotta/kaggle-m5 , file predict.py. Same features, rewritten in polars.
The few deliberate differences are listed in notebook 04.

Vocabulary
----------
- day: the M5 day number (d_1 = 2011-01-29 is day 1). Integer days make lags simple.
- origin: the last day of history. We forecast days origin+1 ... origin+28.
- offset: the "direct" strategy trains one model per forecast day. The model for
  day origin+1+k may only use sales up to the origin, so all its sales-based
  features are shifted k extra days into the past. k is called the offset.
- group level: features are computed on three daily series per row:
    * item_store: the series itself (one item in one store),
    * dept_store: the mean over all items of its department in its store,
    * item:       the mean of its item over all 10 stores.
  Averaging over many series removes noise, which is why the solution uses them.
"""

import polars as pl

from m5_chronos2.data import ID_COLUMNS, PRICES_FILE, SALES_FILE, load_calendar

# The solution keeps the days from d_1123 on (about 2.2 years before the end of the data).
FIRST_DAY = 1123
TRAIN_DAYS = 365  # each model is trained on the last year before the origin
LAG_DAYS = [1, 2, 3, 4, 5, 6, 7]
ROLLING_WINDOWS = [7, 14, 28]
EVENT_DAYS_AFTER = range(7)  # the event uplift looks at the 7 days following each event

GROUP_LEVELS = {
    "item_store": ["item_id", "store_id"],
    "dept_store": ["dept_id", "store_id"],
    "item": ["item_id"],
}


# --- Calendar -------------------------------------------------------------------

def calendar_by_day() -> pl.DataFrame:
    """One row per day: day number, date, week, day of week (0 = Monday), month, SNAP flags, events."""
    calendar = load_calendar()
    return calendar.with_columns(
        pl.col("d").str.slice(2).cast(pl.Int32).alias("day"),
        (pl.col("date").dt.weekday() - 1).alias("day_of_week"),
        pl.col("date").dt.month().alias("month"),
    )


# --- Sales of every series -------------------------------------------------------

def load_all_series_long() -> pl.DataFrame:
    """Daily sales of ALL 30,490 series from FIRST_DAY on, with a masked target 'value'.

    The dept_store and item features average over series that are not in our sample,
    so they need the whole data. Columns: id columns, 'day', 'sales', 'value'.

    'value' is the sales, set to null (unknown) on two kinds of days, as in the solution:
    - before the item is on sale (no price yet): there was nothing to sell;
    - when the whole store sold nothing (stores closed on Christmas day).
    """
    calendar = calendar_by_day()
    last_day = int(pl.scan_csv(SALES_FILE).collect_schema().names()[-1][2:])  # 1941
    day_labels = [f"d_{day}" for day in range(FIRST_DAY, last_day + 1)]

    wide = pl.scan_csv(SALES_FILE).select(ID_COLUMNS + day_labels).collect()

    # First week with a price, per series (computed on the wide table: 30,490 rows only).
    first_price_week = (
        pl.scan_csv(PRICES_FILE)
        .group_by("store_id", "item_id")
        .agg(pl.col("wm_yr_wk").min().alias("first_price_week"))
        .collect()
    )
    wide = wide.join(first_price_week, on=["store_id", "item_id"], how="left")

    # Categorical ids: 25 million rows repeat the same few strings, categories store them once.
    wide = wide.with_columns(pl.col(ID_COLUMNS).cast(pl.Categorical))

    long = wide.unpivot(
        index=ID_COLUMNS + ["first_price_week"], variable_name="d", value_name="sales"
    )
    long = long.with_columns(
        pl.col("d").str.slice(2).cast(pl.Int32).alias("day"),
        pl.col("sales").cast(pl.Float32),
    ).drop("d")
    long = long.join(calendar.select("day", "wm_yr_wk"), on="day", how="left")

    # Mask 1: not on sale yet.
    on_sale = pl.col("wm_yr_wk") >= pl.col("first_price_week")
    # Mask 2: the whole store sold nothing that day.
    store_closed = pl.col("sales").sum().over("store_id", "day") == 0

    long = long.with_columns(
        pl.when(on_sale & ~store_closed).then(pl.col("sales")).otherwise(None).alias("value")
    )
    return long.drop("first_price_week", "wm_yr_wk").sort("id", "day")


# --- Daily series per group level -----------------------------------------------

def group_daily_means(all_long: pl.DataFrame, group_columns: list[str], origin: int) -> pl.DataFrame:
    """Mean daily 'value' of each group (nulls ignored), one row per (group, day).

    Values after the origin are set to null: they are the future we forecast, so no
    feature may see them. Group columns are turned back into plain strings, so they
    can be joined with our sample.
    """
    means = (
        all_long.group_by(group_columns + ["day"])
        .agg(pl.col("value").mean().alias("value"))
        .with_columns(pl.col(group_columns).cast(pl.String))
        .with_columns(pl.when(pl.col("day") <= origin).then(pl.col("value")).otherwise(None).alias("value"))
    )
    calendar = calendar_by_day().select("day", "day_of_week")
    means = means.join(calendar, on="day", how="left")
    return means.sort(group_columns + ["day"])


# --- Features computed once per origin --------------------------------------------

def expanding_stats(means: pl.DataFrame, group_columns: list[str], level: str) -> pl.DataFrame:
    """Mean and variance of the group's daily series over all days so far (not the current one).

    Four features per group level:
    - {level}_mean, {level}_var: over all past days;
    - {level}_day_of_week_mean, {level}_day_of_week_var: over past days with the same
      day of week, divided by {level}_mean (so they measure the weekly shape, not the level).
    Past the origin, values are null, so the statistics stay frozen at their origin value.
    """

    def running_mean_and_var(partition: list[str], name: str) -> list[pl.Expr]:
        x = pl.col("value").fill_null(0)
        n = pl.col("value").is_not_null().cast(pl.Int64).cum_sum().over(partition)  # days seen so far
        sum_x = x.cum_sum().over(partition)
        sum_x2 = (x * x).cum_sum().over(partition)
        mean = pl.when(n > 0).then(sum_x / n).otherwise(None)
        # sample variance (divided by n - 1), as pandas computes it; undefined below 2 days
        var = pl.when(n > 1).then((sum_x2 - sum_x * sum_x / n) / (n - 1)).otherwise(None)
        return [mean.alias(f"{name}_mean"), var.alias(f"{name}_var")]

    stats = means.with_columns(running_mean_and_var(group_columns, level))
    stats = stats.with_columns(running_mean_and_var(group_columns + ["day_of_week"], f"{level}_day_of_week"))

    # "Over the past days only": take the value of the previous row (previous day, or
    # previous same weekday), then fill gaps with the last known value.
    overall = [f"{level}_mean", f"{level}_var"]
    weekly = [f"{level}_day_of_week_mean", f"{level}_day_of_week_var"]
    stats = stats.with_columns(
        [pl.col(c).shift(1).forward_fill().over(group_columns) for c in overall]
        + [pl.col(c).shift(1).forward_fill().over(group_columns + ["day_of_week"]) for c in weekly]
    )

    # Weekly shape relative to the overall level (null when the level is 0).
    level_mean = pl.col(f"{level}_mean")
    stats = stats.with_columns(
        [pl.when(level_mean > 0).then(pl.col(c) / level_mean).otherwise(None).alias(c) for c in weekly]
    )
    return stats.select(group_columns + ["day"] + overall + weekly)


def event_uplift(item_means: pl.DataFrame, origin: int) -> pl.DataFrame:
    """For each day: how much sales usually change in the days following a calendar event.

    For each event e (Super Bowl, Easter, ...) and each n = 0..6:
        uplift(e, n) = mean sales on days n days after e
                       - mean sales on days n + 7 days after e (same weekday, one week later)
    measured on the item-level daily means, over the history (up to the origin).
    The feature of a day is the sum of the uplifts of the (e, n) pairs it belongs to.
    """
    calendar = calendar_by_day()
    daily_sales = item_means.group_by("day").agg(pl.col("value").mean().alias("sales")).sort("day")

    event_days = calendar.select(
        "day", pl.concat_list("event_name_1", "event_name_2").list.drop_nulls().alias("events")
    ).explode("events").drop_nulls("events")

    rows = []
    for event in event_days["events"].unique().sort().to_list():
        days_of_event = event_days.filter(pl.col("events") == event)["day"]
        for n in EVENT_DAYS_AFTER:
            during = daily_sales.filter(pl.col("day").is_in(days_of_event + n), pl.col("day") <= origin)
            week_after = daily_sales.filter(pl.col("day").is_in(days_of_event + n + 7), pl.col("day") <= origin)
            mean_during, mean_week_after = during["sales"].mean(), week_after["sales"].mean()
            if mean_during is None or mean_week_after is None:  # not seen in the history
                continue
            uplift = mean_during - mean_week_after
            for day in (days_of_event + n).to_list():
                rows.append({"day": day, "event_uplift": uplift})

    uplift_by_day = pl.DataFrame(rows, schema={"day": pl.Int32, "event_uplift": pl.Float64})
    uplift_by_day = uplift_by_day.group_by("day").agg(pl.col("event_uplift").sum())
    all_days = calendar.select("day")
    return all_days.join(uplift_by_day, on="day", how="left").with_columns(pl.col("event_uplift").fill_null(0))


# --- Features that depend on the offset --------------------------------------------

def lag_features(means: pl.DataFrame, group_columns: list[str], level: str, offset: int) -> pl.DataFrame:
    """Past values of the group's daily series, seen from a model with the given offset.

    - lag_{n}_{level}: the value n + offset days before, for n = 1..7;
    - last_{w}_days_avg_{level}: the mean of the w days ending offset + 1 days before, w = 7, 14, 28.
    """
    value = pl.col("value")
    features = [value.shift(n + offset).over(group_columns).alias(f"lag_{n}_{level}") for n in LAG_DAYS]
    features += [
        value.rolling_mean(window_size=w).shift(1 + offset).over(group_columns).alias(f"last_{w}_days_avg_{level}")
        for w in ROLLING_WINDOWS
    ]
    return means.select(group_columns + ["day"] + features)


# --- Putting it together -----------------------------------------------------------

def series_features(sample_long: pl.DataFrame) -> pl.DataFrame:
    """Features of our sample that do not depend on the origin or the offset.

    sample_long: one row per (series, day) with id columns, 'day', 'value', 'sell_price'.
    Adds: day_of_week, month, snap (SNAP day in the item's state), price features.
    """
    calendar = calendar_by_day().select("day", "day_of_week", "month", "snap_CA", "snap_TX", "snap_WI")
    table = sample_long.join(calendar, on="day", how="left")
    table = table.with_columns(
        pl.when(pl.col("state_id") == "CA").then(pl.col("snap_CA"))
        .when(pl.col("state_id") == "TX").then(pl.col("snap_TX"))
        .otherwise(pl.col("snap_WI"))
        .alias("snap")
    ).drop("snap_CA", "snap_TX", "snap_WI")

    # Price momentum: mean price over the previous n days, minus today's price.
    table = table.sort("id", "day").with_columns(
        [
            (pl.col("sell_price").rolling_mean(window_size=n).shift(1).over("id") - pl.col("sell_price"))
            .alias(f"sell_price_lag_{n}")
            for n in (7, 14)
        ]
    )
    return table


def days_since_first_sale(sample_long: pl.DataFrame, origin: int) -> pl.DataFrame:
    """For each (series, day): days since the series' first sale (known at the origin)."""
    first_sale = (
        sample_long.filter((pl.col("value") > 0) & (pl.col("day") <= origin))
        .group_by("id")
        .agg(pl.col("day").min().alias("first_sale_day"))
    )
    table = sample_long.select("id", "day").join(first_sale, on="id", how="left")
    return table.with_columns(
        (pl.col("day") - pl.col("first_sale_day")).alias("days_since_first_sale")
    ).select("id", "day", "days_since_first_sale")
