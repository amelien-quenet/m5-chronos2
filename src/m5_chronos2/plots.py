"""One shared look for every figure, so the blog stays visually consistent."""

import matplotlib.pyplot as plt
import numpy as np

# One color per product category. Fixed: a category keeps its color in every figure.
CATEGORY_COLORS = {
    "FOODS": "#2a78d6",  # blue
    "HOUSEHOLD": "#eb6834",  # orange
    "HOBBIES": "#1baf7a",  # aqua
}

# Color for single-series plots.
MAIN_COLOR = "#2a78d6"

TEXT_COLOR = "#0b0b0b"
MUTED_TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def use_blog_style() -> None:
    """Light, quiet axes so the data stands out. Call once at the top of a notebook."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED_TEXT_COLOR,
            "axes.labelcolor": MUTED_TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titlesize": 11,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED_TEXT_COLOR,
            "ytick.color": MUTED_TEXT_COLOR,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def index_plot(ax, per_series, metric, name_a, name_b, order_by="volume"):
    """Compare two models series by series, with a symmetric relative difference.

    per_series: one row per series, with columns f"{metric}_{name_a}", f"{metric}_{name_b}"
    and `order_by`. For each series:

        delta = 2 * (error_a - error_b) / (error_a + error_b)     (between -2 and 2)

    delta > 0: model A has the larger error, so B is better (and vice versa).
    Unlike a plain ratio, halving and doubling the error move delta by the same amount.
    Series are sorted by `order_by`, largest first.
    """
    import polars as pl  # local import: the rest of this module does not need polars

    error_a = pl.col(f"{metric}_{name_a}")
    error_b = pl.col(f"{metric}_{name_b}")
    table = per_series.with_columns(
        pl.when(error_a + error_b > 0)
        .then(2 * (error_a - error_b) / (error_a + error_b))
        .otherwise(None)
        .alias("delta")
    ).sort(order_by, descending=True)

    positions = range(table.height)
    ax.axhline(0, color=MUTED_TEXT_COLOR, linewidth=1)
    ax.scatter(positions, table["delta"].to_numpy(), s=10, color=MAIN_COLOR)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlabel(f"series, sorted by {order_by} (largest first)")
    ax.set_ylabel(f"relative difference in {metric}")
    ax.text(0.99, 0.97, f"{name_b} better", transform=ax.transAxes, ha="right", va="top", color=MUTED_TEXT_COLOR)
    ax.text(0.99, 0.03, f"{name_a} better", transform=ax.transAxes, ha="right", va="bottom", color=MUTED_TEXT_COLOR)

    share_b_better = (table["delta"] > 0).mean()
    return share_b_better


# One fixed color per real model, used in every comparison figure.
# Naive baselines stay gray, so the eye goes to the real models.
MODEL_COLORS = {
    "chronos2": "#2a78d6",  # blue
    "chronos2_vanilla": "#2a78d6",  # blue: Chronos-2 out of the box
    "xgboost_btrotta": "#eb6834",  # orange
    "chronos2_best": "#1baf7a",  # aqua: Chronos-2 after context-length choice and fine-tuning
}
BASELINE_STYLES = {"zero": ":", "seasonal_naive": "--", "moving_average_28d": "-"}


def plot_by_bucket(by_bucket, metrics, models=None, figsize=(13, 3.2)):
    """One panel per metric, x = horizon bucket, one line per model.

    by_bucket: output of metrics.compare(..., by="horizon_bucket"). Real models take
    their MODEL_COLORS color; naive baselines stay gray, with BASELINE_STYLES.
    """
    import polars as pl

    from m5_chronos2.metrics import HORIZON_BUCKETS

    models = models or by_bucket["model"].unique(maintain_order=True).to_list()
    bucket_names = list(HORIZON_BUCKETS)
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    for model in models:
        rows = by_bucket.filter(pl.col("model") == model).sort("horizon_bucket")
        for ax, metric in zip(axes, metrics):
            ax.plot(range(len(bucket_names)), rows[metric].to_numpy(),
                    color=MODEL_COLORS.get(model, MUTED_TEXT_COLOR),
                    linestyle=BASELINE_STYLES.get(model, "-"), marker="o", markersize=4, label=model)
    for ax, metric in zip(axes, metrics):
        ax.set_title(metric, loc="left")
        ax.set_xticks(range(len(bucket_names)))
        ax.set_xticklabels([f"{first}-{last}" for first, last in HORIZON_BUCKETS.values()])
        ax.set_xlabel("days ahead")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=len(models))
    return fig, axes


def plot_fan(ax, past, forecast, actual, color=MAIN_COLOR, label="forecast"):
    """History, actual values over the window, and the forecast's quantile fan.

    past / actual: 'date' and 'y'. forecast: standard format (q0.1 ... q0.9, y_pred).
    Light band: 10%-90% quantiles; dark band: 20%-80%; line: median.
    """
    dates = forecast["date"].to_numpy()
    ax.plot(past["date"].to_numpy(), past["y"].to_numpy(), color=MUTED_TEXT_COLOR, linewidth=0.8)
    ax.fill_between(dates, forecast["q0.1"].to_numpy(), forecast["q0.9"].to_numpy(),
                    color=color, alpha=0.15, linewidth=0, label=f"{label}, 10-90%")
    ax.fill_between(dates, forecast["q0.2"].to_numpy(), forecast["q0.8"].to_numpy(),
                    color=color, alpha=0.30, linewidth=0, label=f"{label}, 20-80%")
    ax.plot(dates, forecast["y_pred"].to_numpy(), color=color, label=f"{label}, median")
    ax.plot(actual["date"].to_numpy(), actual["y"].to_numpy(), color=TEXT_COLOR, linestyle="none",
            marker="o", markersize=3, label="actual")
    ax.set_ylabel("units sold")


def plot_change_bars(ax, table, metric, label_column="model"):
    """Horizontal bars: change of `metric` vs the reference, in %. Left of 0 = better."""
    column = f"{metric}_change_%"
    labels = table[label_column].to_list()
    values = table[column].to_numpy()
    positions = range(len(labels))
    ax.barh(positions, values, color=MAIN_COLOR, height=0.6)
    ax.axvline(0, color=MUTED_TEXT_COLOR, linewidth=1)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} change vs reference (%), lower is better")
    for position, value in zip(positions, values):
        ax.text(value, position, f" {value:+.1f}% ", va="center",
                ha="left" if value >= 0 else "right", color=TEXT_COLOR, fontsize=8)


def plot_shape_vs_alpha(stats, weekday_names=None, figsize=(11, 3.4)):
    """The permutation null against the observed increment, plus the per-weekday ranks.

    `stats` is what `persistence.shape_explains_alpha` returns, and it is the only input:
    the figure cannot drift out of step with the population the test was run on, which is
    the sub-panel of series carrying both a panel row and a weekly shape.
    """
    if weekday_names is None:
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    ax.hist(stats["null_increments"], bins=30, color=GRID_COLOR,
            label="shape shuffled within volume strata")
    ax.axvline(stats["increment"], color=MAIN_COLOR, linewidth=2)
    ax.text(stats["increment"], ax.get_ylim()[1] * 0.92, " observed ",
            ha="left", color=MAIN_COLOR, fontsize=9)
    if stats.get("null_p95") is not None:
        ax.axvline(stats["null_p95"], color=MUTED_TEXT_COLOR, linestyle="--", linewidth=1)
        ax.text(stats["null_p95"], ax.get_ylim()[1] * 0.60, " null 95th ",
                ha="left", va="center", color=MUTED_TEXT_COLOR, fontsize=8)
    ax.set_xlabel("share of the stable alpha variance that shape adds")
    ax.set_ylabel("permutations")
    ax.legend(fontsize=8, loc="upper right")

    # Which weekday, if any, tracks difficulty. Descriptive only: the test is joint, and
    # these correlations are taken before volume is removed.
    ax = axes[1]
    ranks = stats["spearman_per_weekday"]
    ax.bar(range(len(ranks)), ranks, color=MAIN_COLOR, width=0.6)
    ax.axhline(0, color=MUTED_TEXT_COLOR, linewidth=1)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(weekday_names[:len(ranks)])
    ax.set_ylabel("Spearman with alpha")
    ax.set_title(f"raw, before volume is removed ({stats['n_series']} series)",
                 loc="left", fontsize=9)
    fig.tight_layout()
    return fig, axes


def plot_context_sweep(axes, sweep, metrics, colormap="Blues"):
    """The context-length sweep: per metric, one line per context length, x = horizon bucket.

    sweep: one row per (context_length, horizon_bucket) with the metric columns.
    Color = context length (light = short, dark = long). A line that is best on the
    left and worst on the right shows up as a crossing: the optimum depends on the horizon.
    """
    import matplotlib as mpl
    import polars as pl

    from m5_chronos2.metrics import HORIZON_BUCKETS

    lengths = sorted(sweep["context_length"].unique().to_list())
    norm = mpl.colors.LogNorm(vmin=min(lengths), vmax=max(lengths))
    base = mpl.colormaps[colormap]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("ramp", base(np.linspace(0.3, 1.0, 256)))
    bucket_names = list(HORIZON_BUCKETS)
    for ax, metric in zip(axes, metrics):
        for length in lengths:
            rows = sweep.filter(pl.col("context_length") == length).sort("horizon_bucket")
            ax.plot(range(len(bucket_names)), rows[metric].to_numpy(), color=cmap(norm(length)),
                    linewidth=1.2, marker="o", markersize=3)
        ax.set_title(metric, loc="left")
        ax.set_xticks(range(len(bucket_names)))
        ax.set_xticklabels([f"{first}-{last}" for first, last in HORIZON_BUCKETS.values()])
        ax.set_xlabel("days ahead")
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return mappable
