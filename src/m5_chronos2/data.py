"""Loading the M5 data.

Everything the notebooks need to go from the raw Kaggle CSVs to one tidy table:
one row per (series, day), with the number of units sold.

The raw sales file is "wide": one row per series, one column per day (d_1 ... d_1941).
We turn it "long", which is the format both polars and Chronos-2 prefer.
"""

from pathlib import Path

import polars as pl

# --- Where things are -------------------------------------------------------
# This file lives in <project>/src/m5_chronos2/, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SALES_FILE = RAW_DIR / "sales_train_evaluation.csv"  # d_1 ... d_1941
CALENDAR_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE = RAW_DIR / "sell_prices.csv"  # one row per (store, item, week) the item was on sale

# --- The problem, in constants ------------------------------------------------
# Columns that describe a series (its place in the Walmart hierarchy).
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# We forecast 28 days ahead, like the M5 competition.
HORIZON = 28


def load_calendar() -> pl.DataFrame:
    """One row per day: the M5 day label (d_1, d_2, ...), its real date, events, SNAP days."""
    calendar = pl.read_csv(CALENDAR_FILE)
    calendar = calendar.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    return calendar


def list_all_series() -> pl.DataFrame:
    """One row per series with its hierarchy columns, without the sales (cheap to load)."""
    return pl.scan_csv(SALES_FILE).select(ID_COLUMNS).collect()


def roster_name(sample_size: int | None, seed: int) -> str:
    """A short name for a sample, used in file names. Example: 'n500_seed42' or 'full'."""
    if sample_size is None:
        return "full"
    return f"n{sample_size}_seed{seed}"


def get_roster(all_series: pl.DataFrame, sample_size: int | None, seed: int) -> pl.DataFrame:
    """The list of series we study ("roster").

    The random sample is drawn once, then saved to disk. Every later run (and every
    later notebook) reads the saved file instead of sampling again, so all our
    experiments are evaluated on exactly the same series.

    sample_size=None means: take every series (for when more compute is available).
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    roster_path = ARTIFACTS_DIR / f"roster_{roster_name(sample_size, seed)}.parquet"

    if roster_path.exists():
        print(f"Reading the frozen roster: {roster_path.name}")
        return pl.read_parquet(roster_path)

    if sample_size is None:
        roster = all_series
    else:
        roster = all_series.sample(n=sample_size, seed=seed)
    roster = roster.sort("id")

    roster.write_parquet(roster_path)
    print(f"New roster sampled and saved: {roster_path.name}")
    return roster


def load_sales_long(series_ids: list[str] | pl.Series) -> pl.DataFrame:
    """Sales of the chosen series, as a long table: one row per (series, day).

    Columns: the ID_COLUMNS, then 'date' and 'sales' (units sold that day).
    """
    # 1. Read only the rows (series) we asked for.
    wide = pl.scan_csv(SALES_FILE).filter(pl.col("id").is_in(series_ids)).collect()

    # 2. Wide -> long: the 1,941 day columns become two columns, 'd' and 'sales'.
    long = wide.unpivot(index=ID_COLUMNS, variable_name="d", value_name="sales")

    # 3. Replace the day label (d_1, ...) by the real date.
    calendar = load_calendar().select("d", "date")
    long = long.join(calendar, on="d", how="left")

    return long.select(ID_COLUMNS + ["date", "sales"]).sort("id", "date")


def add_launch_flag(sales: pl.DataFrame) -> pl.DataFrame:
    """Flag the days before each series' first sale.

    Most items are not sold from day 1: the series starts with a run of zeros.
    Our working hypothesis is that the item was not on the shelf yet, so those
    zeros are not real observations of demand. To be checked (prices file).

    Adds two columns: 'first_sale_date' and 'is_launched' (True from the first sale on).
    """
    first_sale = (
        sales.filter(pl.col("sales") > 0)
        .group_by("id")
        .agg(pl.col("date").min().alias("first_sale_date"))
    )
    sales = sales.join(first_sale, on="id", how="left")
    sales = sales.with_columns(
        (pl.col("date") >= pl.col("first_sale_date")).alias("is_launched")
    )
    # A join does not guarantee row order: sort again so each series stays in date order.
    return sales.sort("id", "date")


def load_prices(roster: pl.DataFrame) -> pl.DataFrame:
    """Weekly selling prices of the series in the roster.

    One row per (store, item, week) in which the item had a price, i.e. was on sale.
    A missing row means the item was not on sale that week.
    """
    wanted = roster.select("store_id", "item_id").lazy()
    prices = (
        pl.scan_csv(PRICES_FILE)
        # keep only the (store, item) pairs of our roster
        .join(wanted, on=["store_id", "item_id"], how="semi")
        .collect()
    )
    return prices


def add_price(sales: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Attach the weekly price to each day, and flag the days the item was on sale.

    Adds 'sell_price' (null when not on sale) and 'is_on_sale'.
    """
    # Prices are weekly: find the Walmart week (wm_yr_wk) of each day first.
    calendar = load_calendar().select("date", "wm_yr_wk")
    sales = sales.join(calendar, on="date", how="left")

    sales = sales.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    sales = sales.with_columns(pl.col("sell_price").is_not_null().alias("is_on_sale"))

    return sales.drop("wm_yr_wk").sort("id", "date")
