"""Finding blank periods: long runs of consecutive days with zero sales.

Vocabulary used in the whole project:
- a day is "silent" when the item is on sale but sold nothing;
- a "blank period" is a run of at least `min_days` consecutive silent days.

Days when the item is not on sale are NOT silent: there was nothing to sell,
so they say nothing about demand. They are handled separately (missing data).
"""

import numpy as np
import polars as pl


def add_silent_runs(sales: pl.DataFrame) -> pl.DataFrame:
    """For each day, the length of the run of consecutive silent days it belongs to.

    Needs the columns 'id', 'date', 'sales' and 'is_on_sale'.
    Adds 'is_silent' and 'silent_run_length' (0 on days that are not silent).

    How it works, on one series:
        is_silent         F F T T T F T F
        run starts here?  T F T F F T T T   (each time is_silent changes value)
        run_id            1 1 2 2 2 3 4 5   (running count of the starts)
        run length        - - 3 3 3 - 1 -   (number of days sharing the run_id)
    """
    sales = sales.sort("id", "date")
    sales = sales.with_columns(
        ((pl.col("sales") == 0) & pl.col("is_on_sale")).alias("is_silent")
    )

    run_starts = pl.col("is_silent") != pl.col("is_silent").shift(1)
    sales = sales.with_columns(
        # .over("id") makes shift and cum_sum restart for every series
        run_starts.fill_null(True).cum_sum().over("id").alias("run_id")
    )

    sales = sales.with_columns(
        pl.when(pl.col("is_silent"))
        .then(pl.col("date").count().over("id", "run_id"))
        .otherwise(0)
        .alias("silent_run_length")
    )
    return sales.drop("run_id")


def add_blank_flags(sales: pl.DataFrame, min_days: int, long_days: int) -> pl.DataFrame:
    """Flag the days that belong to a blank period, and split blank periods by length.

    Needs 'silent_run_length' (see add_silent_runs).
    Adds:
    - 'is_blank': the day is inside a run of at least `min_days` silent days;
    - 'blank_kind': "short" (less than `long_days`), "long" (`long_days` or more), or null.
    """
    run_length = pl.col("silent_run_length")
    sales = sales.with_columns((run_length >= min_days).alias("is_blank"))
    sales = sales.with_columns(
        pl.when(~pl.col("is_blank"))
        .then(pl.lit(None, dtype=pl.String))
        .when(run_length < long_days)
        .then(pl.lit("short"))
        .otherwise(pl.lit("long"))
        .alias("blank_kind")
    )
    return sales


# --- Are blank periods shared within groups? ----------------------------------
# These functions work on numpy matrices: one row per series, one column per day.


def tolerate_shifts(blank: np.ndarray, days: int) -> np.ndarray:
    """Widen every blank period by `days` on each side.

    Two series whose blank periods start 2 days apart share no blank day at all,
    yet they clearly react to the same event. After widening by 2 days, they overlap.
    """
    widened = blank.copy()
    for shift in range(1, days + 1):
        widened[:, shift:] |= blank[:, :-shift]   # extend each period to the right
        widened[:, :-shift] |= blank[:, shift:]   # and to the left
    return widened


def pairwise_blank_correlation(blank: np.ndarray, on_sale: np.ndarray, min_common_days: int) -> np.ndarray:
    """Correlation between the blank-period flags of every pair of series.

    For each pair, only the days when BOTH series are on sale are used, and the
    correlation is the Pearson correlation of two 0/1 vectors (the "phi coefficient").
    Unlike a plain dot product, it subtracts the overlap expected by chance: two series
    that are blank 50% of the time share many blank days even if they are unrelated.

    Returns a symmetric matrix; NaN where the pair has fewer than `min_common_days`
    common on-sale days, or where one series has no blank day (or only blank days)
    on the common days (the correlation is undefined).
    """
    n_series = blank.shape[0]
    correlation = np.full((n_series, n_series), np.nan)
    for i in range(n_series):
        for j in range(i + 1, n_series):
            common_days = on_sale[i] & on_sale[j]
            if common_days.sum() < min_common_days:
                continue
            x = blank[i, common_days]
            y = blank[j, common_days]
            if x.std() == 0 or y.std() == 0:
                continue
            correlation[i, j] = correlation[j, i] = np.corrcoef(x, y)[0, 1]
    return correlation


def within_vs_between(correlation: np.ndarray, labels: np.ndarray) -> dict:
    """Mean pairwise correlation inside groups, between groups, and over all pairs."""
    upper = np.triu_indices(len(labels), k=1)        # every pair once
    values = correlation[upper]
    same_group = (labels[:, None] == labels[None, :])[upper]
    defined = ~np.isnan(values)
    return {
        "all_pairs": values[defined].mean(),
        "within": values[defined & same_group].mean(),
        "between": values[defined & ~same_group].mean(),
    }


def grouping_permutation_test(correlation: np.ndarray, labels: np.ndarray,
                              n_permutations: int, seed: int) -> dict:
    """Is the within-group correlation higher than chance would give?

    Statistic: within - between. Under the hypothesis "the grouping does not matter",
    shuffling the labels between series (group sizes unchanged) should give values as
    large as the observed one. The p-value is the share of shuffles that do.
    """
    observed = within_vs_between(correlation, labels)
    observed_gap = observed["within"] - observed["between"]

    rng = np.random.default_rng(seed)
    null_gaps = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(labels)
        result = within_vs_between(correlation, shuffled)
        null_gaps[k] = result["within"] - result["between"]

    return {
        **observed,
        "gap": observed_gap,
        "null_gaps": null_gaps,
        "z_score": (observed_gap - null_gaps.mean()) / null_gaps.std(),
        "p_value": (1 + (null_gaps >= observed_gap).sum()) / (1 + n_permutations),
    }
