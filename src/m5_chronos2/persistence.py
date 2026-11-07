"""Is forecast difficulty a property of the series, or of the week?

The first half of the last question of the study: have we hit a ceiling? Every
remaining lever -- fine-tuning per group, cross-learning inside a group, one model
per group -- needs difficulty to be a *stable property of a series*. If the series
that are hard this week are not the ones that were hard last week, then no static
grouping of series can help, whatever a clustering algorithm reports. That is a
ceiling argument, and it is cheap: a handful of extra forecast runs.

The measurement
---------------
Forecast repeatedly from many origins and study the per-(series, origin) log error
``x[f, o]``, then double-centre it:

    x[f, o] = mu + alpha_f + beta_o + eps[f, o]

- ``alpha_f`` is the series. By construction the *only* part a static grouping
  could exploit.
- ``beta_o`` is the origin: weeks that are hard for everyone.
- ``eps[f, o]`` is everything left, and it gets split further into a part that
  survives from one origin to the next (persistent), a part that does not
  (transient), and the measurement noise of the cell itself.

Three decisions that make the numbers mean something
----------------------------------------------------
- **Logs.** Error is positive and right-skewed, and difficulty is multiplicative:
  "30% worse than average" compares across origins of unequal difficulty,
  "+0.2 WQL" does not.
- **Zero-shot model only.** The study's best model is a stitch involving a LoRA
  checkpoint fine-tuned on history up to the origin. Sweeping the origin backwards
  with a checkpoint trained near the *last* origin would let the model see the
  future of every earlier window, and re-fine-tuning 24 times is not affordable.
  So the sweep uses the vanilla model at the context length chosen for days 1-7,
  on the 7 days that segment owns.
- **Non-overlapping scored windows.** Origins are SEGMENT_DAYS apart and only the
  first SEGMENT_DAYS forecast days are scored, so no day is scored twice and the
  lag-1 autocovariance across origins is not overlap arithmetic.

Layout of this module: the polars half turns forecasts into one number per
(series, origin); the numpy half does the statistics and never touches polars, so
every estimator below can be checked against `simulate_panel`, a panel with known
components.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from m5_chronos2.metrics import QUANTILE_LEVELS, add_horizon, quantile_loss

# The horizon segment the zero-shot model owns (notebook 07's stitch boundary), and
# therefore both the scored window and the spacing between origins.
SEGMENT_DAYS = 7

# 24 origins: what 7-day steps allow over the available history without the earliest
# origin running short of context.
N_ORIGINS = 24

# Same as the scoring protocol of notebook 03.
MIN_HISTORY_DAYS = 56


# --- The polars half: forecasts -> one log error per (series, origin) ----------------

def origin_grid(last_origin: date, n_origins: int = N_ORIGINS,
                step_days: int = SEGMENT_DAYS) -> list[date]:
    """`n_origins` origins, `step_days` apart, ending at `last_origin`, in time order.

    `last_origin` is the *validation* origin, not the test one: this is a diagnostic,
    and there is no reason to let it read the window the final benchmark is scored on.
    """
    return [last_origin - timedelta(days=step_days * k) for k in reversed(range(n_origins))]


def sweep_context(sales: pl.DataFrame, origin: date,
                  min_history_days: int = MIN_HISTORY_DAYS) -> pl.DataFrame:
    """History up to `origin` for the series that have enough of it at that origin.

    Series are dropped here rather than at scoring time for two reasons: a series
    with four days of history is not a forecast anyone should pay for, and the
    earliest origins would otherwise hand the model a batch of near-empty series.
    The filter is the same `>= min_history_days on-sale days` condition the scoring
    protocol applies, so no series is forecast that could not be scored.
    """
    from m5_chronos2.chronos2 import history

    observed = sales.filter((pl.col("date") <= origin) & ~pl.col("is_missing"))
    long_enough = (
        observed.group_by("id").len()
        .filter(pl.col("len") >= min_history_days)["id"]
        .sort().to_list()
    )
    return history(sales, origin, long_enough)


def scored_cells(sales: pl.DataFrame, origins: list[date],
                 horizon_days: int = SEGMENT_DAYS,
                 min_history_days: int = MIN_HISTORY_DAYS) -> pl.DataFrame:
    """The (series, origin) pairs that may be scored, over a `horizon_days` window.

    The same three conditions as `evaluation.check_series` -- enough on-sale history,
    no missing day in the window, at least one sale in the window -- applied to a
    7-day window instead of the protocol's 28. Written here rather than by
    generalising `evaluation.py`, which holds the frozen protocol of the benchmark
    and should not grow a parameter for a diagnostic.
    """
    parts = []
    for origin in origins:
        window_end = origin + timedelta(days=horizon_days)
        history_stats = (
            sales.filter(pl.col("date") <= origin)
            .group_by("id")
            .agg((~pl.col("is_missing")).sum().alias("history_days_on_sale"))
        )
        window_stats = (
            sales.filter((pl.col("date") > origin) & (pl.col("date") <= window_end))
            .group_by("id")
            .agg(
                pl.col("is_missing").any().alias("window_has_missing"),
                pl.col("sales").sum().alias("window_total_sales"),
            )
        )
        checks = history_stats.join(window_stats, on="id", how="inner")
        parts.append(
            checks.filter(
                (pl.col("history_days_on_sale") >= min_history_days)
                & ~pl.col("window_has_missing")
                & (pl.col("window_total_sales") > 0)
            )
            .select("id")
            .with_columns(pl.lit(origin).alias("origin"))
        )
    return pl.concat(parts).sort("origin", "id")


def _wql_by_cell(scored: pl.DataFrame, name: str) -> pl.DataFrame:
    """WQL per (series, origin), same formula as metrics.py but not pooled over series."""
    total_actual = pl.col("y").abs().sum()
    per_level = [2 * quantile_loss(level).sum() / total_actual for level in QUANTILE_LEVELS]
    cells = scored.group_by("id", "origin").agg(
        (pl.sum_horizontal(per_level) / len(QUANTILE_LEVELS)).alias(name),
        pl.col("y").abs().sum().alias("total_actual"),
        pl.len().alias("n_days"),
    )
    # A window with no sales has no WQL: the denominator is zero. The window-level
    # filter already removed those, but the odd/even halves can still be empty.
    return cells.filter(pl.col("total_actual") > 0).drop("total_actual")


def cell_wql(predictions: pl.DataFrame, truth: pl.DataFrame, cells: pl.DataFrame,
             horizon_days: int = SEGMENT_DAYS) -> pl.DataFrame:
    """One WQL per (series, origin), plus the odd-day and even-day halves of each cell.

    The halves are the within-cell replicate the noise floor is built from. Without
    them, the sampling noise of a 7-day window lands in the transient term and the
    conclusion becomes "unpredictable shocks" when it should be "thin data".

    Returns 'id', 'origin', 'wql', 'n_days', 'wql_odd', 'wql_even' (the last two null
    when that half of the window has no sales).
    """
    scored = (
        predictions.join(cells, on=["id", "origin"], how="inner")
        .join(truth, on=["id", "date"], how="inner")
        .filter(~pl.col("is_missing"))
    )
    scored = add_horizon(scored).filter(pl.col("horizon") <= horizon_days)

    whole = _wql_by_cell(scored, "wql")
    odd = _wql_by_cell(scored.filter(pl.col("horizon") % 2 == 1), "wql_odd").drop("n_days")
    even = _wql_by_cell(scored.filter(pl.col("horizon") % 2 == 0), "wql_even").drop("n_days")
    return (
        whole.join(odd, on=["id", "origin"], how="left")
        .join(even, on=["id", "origin"], how="left")
        .sort("id", "origin")
    )


@dataclass
class Panel:
    """A balanced (series x origin) grid of log errors, and the two half-window reads.

    Balanced on purpose: double-centring a grid with holes is not the same estimator,
    and a series missing from half the origins would contribute an `alpha_f` measured
    on a different set of weeks than its neighbours'. `dropped` records what that
    costs, because a filter that quietly removes half the roster changes the question.
    """

    x: np.ndarray            # F x O, log WQL
    odd: np.ndarray          # F x O, log WQL on the odd days of each window (may be NaN)
    even: np.ndarray         # F x O, idem on the even days
    ids: list[str]
    origins: list[date]
    dropped: dict[str, int]

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape


def build_panel(cells: pl.DataFrame) -> Panel:
    """Keep the series scored at *every* origin, and take logs."""
    origins = sorted(cells["origin"].unique().to_list())
    usable = cells.filter(pl.col("wql") > 0)

    complete = (
        usable.group_by("id").len()
        .filter(pl.col("len") == len(origins))["id"]
        .sort().to_list()
    )
    dropped = {
        "series_seen": cells["id"].n_unique(),
        "series_kept": len(complete),
        "cells_with_zero_wql": cells.height - usable.height,
    }
    if not complete:
        raise ValueError("no series is scored at every origin: widen the roster or the grid")

    table = usable.filter(pl.col("id").is_in(complete)).sort("id", "origin")
    index = {origin: position for position, origin in enumerate(origins)}
    rows = {series: position for position, series in enumerate(complete)}

    shape = (len(complete), len(origins))
    x = np.full(shape, np.nan)
    odd = np.full(shape, np.nan)
    even = np.full(shape, np.nan)
    for series, origin, wql, wql_odd, wql_even in zip(
        table["id"], table["origin"], table["wql"], table["wql_odd"], table["wql_even"]
    ):
        i, j = rows[series], index[origin]
        x[i, j] = np.log(wql)
        if wql_odd is not None and wql_odd > 0:
            odd[i, j] = np.log(wql_odd)
        if wql_even is not None and wql_even > 0:
            even[i, j] = np.log(wql_even)

    assert not np.isnan(x).any(), "the panel is not balanced"
    return Panel(x=x, odd=odd, even=even, ids=complete, origins=origins, dropped=dropped)


# --- The numpy half: the decomposition ----------------------------------------------

@dataclass
class Decomposition:
    """The variance budget of the panel. Everything here is a sum of squares or a ratio."""

    mu: float
    alpha: np.ndarray        # F, the series effect
    beta: np.ndarray         # O, the origin effect
    eps: np.ndarray          # F x O, what neither explains

    ss_frame: float
    ss_origin: float
    ss_interaction: float
    total: float

    ss_frame_adjusted: float
    v_interaction: float
    v_noise: float
    v_persistent: float      # raw, NOT clipped: see `shares`
    v_transient: float

    n_series: int
    n_origins: int
    n_noise_cells: int

    def shares(self, clip: bool = True) -> dict[str, float]:
        """Each component as a share of the total sum of squares.

        `clip=False` leaves a negative persistence as it is. That matters: clipping a
        small negative to zero turns "indistinguishable from zero" into "exactly zero",
        which is a much stronger claim than the data supports. Clip only for a stacked
        bar chart, where the shares have to add up, and say that you did.

        'series' + 'origin' + 'persistent' + 'transient' + 'noise' adds to 1 (the raw,
        unadjusted series share is the one that balances the budget). 'stable' is not
        part of the budget: it is the sum of the two components a grouping could use,
        reported because the boundary between them is not identified -- see `decompose`.
        """
        cells = self.n_series * self.n_origins
        persistent = max(self.v_persistent, 0.0) if clip else self.v_persistent
        transient = self.v_interaction - self.v_noise - max(self.v_persistent, 0.0)
        adjusted = self.ss_frame_adjusted / self.total
        return {
            "series": self.ss_frame / self.total,
            "series_adjusted": adjusted,
            "origin": self.ss_origin / self.total,
            "interaction": self.ss_interaction / self.total,
            "persistent": cells * persistent / self.total,
            "transient": cells * transient / self.total,
            "noise": cells * self.v_noise / self.total,
            "stable": adjusted + cells * max(self.v_persistent, 0.0) / self.total,
        }


def _noise_variance(odd: np.ndarray, even: np.ndarray) -> tuple[float, int]:
    """Measurement variance of a cell, from its own two halves.

    The two halves of a window are the same series over the same week, so their
    difference is pure sampling noise. Half that difference has the variance of the
    noise carried by their mean -- which is what the cell's error estimate is.
    """
    valid = np.isfinite(odd) & np.isfinite(even)
    if valid.sum() < 2:
        return 0.0, int(valid.sum())
    half_difference = 0.5 * (odd[valid] - even[valid])
    return float(np.mean(half_difference ** 2)), int(valid.sum())


def _lag1_autocovariance(eps: np.ndarray) -> float:
    """Average over series of the lag-1 product of consecutive origins."""
    if eps.shape[1] < 2:
        return 0.0
    return float(np.mean(eps[:, :-1] * eps[:, 1:]))


def decompose(panel: Panel) -> Decomposition:
    """Double-centre the panel and split the residual into persistent / transient / noise.

    Two corrections that are easy to leave out and change the answer:

    - **`+ v_interaction / n_origins` on the persistence term.** Double-centring makes
      the residual sum to zero along each row, which induces a *negative* lag-1
      autocovariance even for pure white noise. Without the correction, persistence is
      systematically under-reported.
    - **`-(F - 1) * v_interaction` on the series sum of squares.** Each `alpha_f` is a
      mean of noisy cells, so it carries noise of its own and `ss_frame` is inflated
      by about that much even when no series is genuinely harder than another. Both
      the raw and the adjusted number are reported; the adjusted one is the one to
      quote, since it answers "how much of this is really the series".

    One limit of this decomposition, measured rather than assumed. A difficulty that
    drifts slowly is partly indistinguishable from a constant one over a panel of
    finite span: the mean of a highly autocorrelated residual lands in `alpha_f` and
    not in the interaction. Sweeping the autocorrelation of a planted drift through
    `simulate_panel` (rho from 0 to 0.97, planted stable share held near 19%) moves it
    between the two terms -- series 0.6% / persistent 1.4% at rho = 0, series 14.9% /
    persistent 4.6% at rho = 0.97 -- while their **sum** tracks the planted value to
    within a point or two throughout. So the boundary between `series` and
    `persistent` is a statement about the timescale of the drift rather than a clean
    partition, and `shares()["stable"]` reports the quantity the two jointly estimate.
    The verdict rule stays on `series_adjusted`, because that is the part a *static*
    grouping -- fitted once and left alone -- can actually use.
    """
    x = panel.x
    n_series, n_origins = x.shape
    cells = n_series * n_origins

    mu = float(x.mean())
    alpha = x.mean(axis=1) - mu
    beta = x.mean(axis=0) - mu
    eps = x - mu - alpha[:, None] - beta[None, :]

    ss_frame = float(n_origins * np.sum(alpha ** 2))
    ss_origin = float(n_series * np.sum(beta ** 2))
    ss_interaction = float(np.sum(eps ** 2))
    total = float(np.sum((x - mu) ** 2))

    v_interaction = ss_interaction / cells
    v_noise, n_noise_cells = _noise_variance(panel.odd, panel.even)
    v_persistent = _lag1_autocovariance(eps) + v_interaction / n_origins
    v_transient = v_interaction - v_noise - max(v_persistent, 0.0)

    return Decomposition(
        mu=mu, alpha=alpha, beta=beta, eps=eps,
        ss_frame=ss_frame, ss_origin=ss_origin, ss_interaction=ss_interaction, total=total,
        ss_frame_adjusted=max(ss_frame - (n_series - 1) * v_interaction, 0.0),
        v_interaction=v_interaction, v_noise=v_noise,
        v_persistent=v_persistent, v_transient=v_transient,
        n_series=n_series, n_origins=n_origins, n_noise_cells=n_noise_cells,
    )


def _resample(panel: Panel, rows: np.ndarray) -> Panel:
    return Panel(
        x=panel.x[rows], odd=panel.odd[rows], even=panel.even[rows],
        ids=[panel.ids[i] for i in rows], origins=panel.origins, dropped=panel.dropped,
    )


def bootstrap_shares(panel: Panel, n_draws: int = 1000, seed: int = 0,
                     level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Confidence interval of each share, resampling *series* with replacement.

    Series, not cells: the uncertainty that matters is "would another sample of series
    give the same answer", and cells within a series are not independent -- that
    dependence is the very thing being measured.
    """
    generator = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(n_draws):
        rows = generator.integers(0, panel.shape[0], size=panel.shape[0])
        for key, value in decompose(_resample(panel, rows)).shares(clip=False).items():
            collected.setdefault(key, []).append(value)
    tail = 100 * (1 - level) / 2
    return {key: (float(np.percentile(values, tail)),
                  float(np.percentile(values, 100 - tail)))
            for key, values in collected.items()}


# --- Rank agreement: the same question without any variance model -------------------

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged. Written out rather than imported: six lines, and the
    tie rule is part of the statistic rather than a library default."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of the average ranks."""
    ra, rb = _average_ranks(np.asarray(a, float)), _average_ranks(np.asarray(b, float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.sum(ra * rb) / denominator) if denominator > 0 else float("nan")


def rank_agreement(panel: Panel, decile: float = 0.1) -> dict:
    """Does the ranking of series by difficulty survive from one origin to the next?

    Deliberately free of the variance model above: if the two disagree, the
    decomposition is wrong somewhere. `worst_decile_survival` is the share of the
    hardest `decile` of series at one origin that are still in it at the next, against
    the `decile` a coin flip would give.
    """
    x = panel.x
    n_series, n_origins = x.shape
    consecutive = np.array([spearman(x[:, o], x[:, o + 1]) for o in range(n_origins - 1)])

    size = max(int(round(decile * n_series)), 1)
    survivals = []
    for o in range(n_origins - 1):
        worst_now = set(np.argsort(-x[:, o])[:size].tolist())
        worst_next = set(np.argsort(-x[:, o + 1])[:size].tolist())
        survivals.append(len(worst_now & worst_next) / size)

    return {
        "consecutive": consecutive,
        "consecutive_median": float(np.median(consecutive)),
        "first_vs_last": spearman(x[:, 0], x[:, -1]),
        "worst_decile_survival": float(np.mean(survivals)),
        "worst_decile_random": float(size / n_series),
    }


# --- Validating the residual of the residual ----------------------------------------

def transient_ratio(panel: Panel, result: Decomposition) -> np.ndarray:
    """Per-cell transient magnitude, in units of the noise floor.

    `v_transient` is a difference of differences, so *any* leak -- a broken origin, a
    series going dark, a join artefact -- lands in it. This is how to look at the
    cells it is built from: predict each residual from its neighbouring origins using
    the autocovariances already estimated (nothing new is fitted), and square what is
    left over the noise floor.

    A large ratio should be one origin jumping off an otherwise flat trace with a
    visible event underneath. If instead it is a near-zero-volume series, the
    transient share is measuring data holes and the ceiling argument has to be re-run
    on a cleaner panel.
    """
    eps = result.eps
    n_origins = eps.shape[1]
    g0 = result.v_interaction
    g1 = _lag1_autocovariance(eps)
    if g0 <= 0:
        return np.zeros_like(eps)
    g2 = float(np.mean(eps[:, :-2] * eps[:, 2:])) if n_origins > 2 else 0.0

    # Best linear prediction of eps[:, o] from its neighbours. By symmetry both
    # neighbours take the same weight, so the 2x2 system collapses to one division.
    weight_both = g1 / (g0 + g2) if (g0 + g2) != 0 else 0.0
    weight_one = g1 / g0

    predicted = np.zeros_like(eps)
    predicted[:, 0] = weight_one * eps[:, 1]
    predicted[:, -1] = weight_one * eps[:, -2]
    if n_origins > 2:
        predicted[:, 1:-1] = weight_both * (eps[:, :-2] + eps[:, 2:])

    floor = result.v_noise if result.v_noise > 0 else g0
    return (eps - predicted) ** 2 / floor


# --- Is the stable part anything more than volume? ----------------------------------

def series_noise(panel: Panel) -> np.ndarray:
    """Per-series measurement variance of one cell, from that series' own two halves.

    `_noise_variance` pools every cell into a single number, which is what the variance
    budget needs. Here the number is wanted **per series**, because the sampling noise of
    a short window is strongly volume-dependent: a window selling three units carries far
    more of it than one selling three hundred. Since `alpha_f` is a mean of noisy cells,
    low-volume series have theirs inflated *more* than high-volume ones -- a bias that
    differs per series, which the global adjustment of `decompose` cannot remove.

    A series whose own noise is not estimable (fewer than two origins with a sale in both
    halves of the window) falls back to the pooled value rather than to zero, which would
    claim its `alpha_f` is measured perfectly.
    """
    pooled, _ = _noise_variance(panel.odd, panel.even)
    noise = np.full(panel.x.shape[0], pooled, dtype=float)
    for i in range(panel.x.shape[0]):
        value, n_valid = _noise_variance(panel.odd[i], panel.even[i])
        if n_valid >= 2:
            noise[i] = value
    return noise


def _explained_variance(y: np.ndarray, columns: np.ndarray) -> float:
    """Variance of `y` explained by a least-squares fit on `columns`, bias-adjusted.

    Two points matter for reading the result.

    **Measurement error in `y` does not bias this.** `y` here is `alpha_f`, which carries
    noise of its own, but noise in the *dependent* variable leaves `cov(y, z)` and the
    slope unbiased -- it inflates `var(y)`, which is corrected separately. So the explained
    variance needs no errors-in-variables treatment; only the denominator does.

    **A least-squares fit explains some variance by construction**, about
    `k * var(residual) / n` of it. That is subtracted here, the same adjustment an adjusted
    R-squared makes, so a covariate that carries nothing scores about zero rather than a
    small positive number.
    """
    n = len(y)
    k = columns.shape[1]
    design = np.column_stack([np.ones(n), columns])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    explained = float(np.var(fitted))
    residual = float(np.var(y)) - explained
    if n - k - 1 > 0:
        explained -= k * residual / (n - k - 1)
    return max(explained, 0.0)


def explain_alpha(panel: Panel, result: Decomposition,
                  covariates: dict[str, np.ndarray]) -> dict:
    """What is left of the stable difficulty once its own noise and `covariates` are out.

    A large series share is the start of an argument, not the end of one. WQL is
    scale-free, so a dependence on volume would not be an artefact of units -- it would be
    a statement about predictability: an item selling a third of a unit a day has an
    irreducibly high relative error, and that is a permanent property of the item. If
    `alpha_f` is a function of volume then "group the series by difficulty" means "bucket
    them by volume", which needs no clustering and which the forecaster can read off the
    history unaided. What survives is the part a grouping would have to earn its keep on.

    Three shares are reported, all as a fraction of the panel's total sum of squares so
    they sit on the same axis as `Decomposition.shares()`:

    - `share_raw` -- `ss_frame / total`, identical to `shares()["series"]`.
    - `share_corrected_global` -- `shares()["series_adjusted"]`, taken from `result` so
      the two agree by construction. It subtracts the *whole* interaction variance from
      every series equally.
    - `share_corrected` -- subtracts each series' **own measured noise** instead. It is
      therefore the larger of the two: it removes only what the odd/even halves prove is
      measurement error, and leaves real week-to-week movement inside `alpha_f`.

    `per_covariate[name]` gives the Spearman correlation with `alpha_f`, the share of the
    noise-corrected `alpha` variance that covariate explains (`r2`), and the share of the
    budget still on the series after removing it. `all_covariates` does the same with every
    covariate at once.

    **One bias this cannot remove, and the control that exposes it.** Subtracting variances
    fixes an inflated spread; it does not fix an inflated *level*. `x` is a log, so a
    noisier estimate of the same underlying error need not have the same expectation, and
    the noise level is itself a function of volume -- which could put a volume slope into
    `alpha_f` that no variance correction touches. There is no clean closed form for it
    here, so the per-series noise is added to `per_covariate` as `"noise level (control)"`.
    If `alpha_f` correlates with its own measurement noise about as strongly as it
    correlates with volume, the volume result is not safe to read, and the panel needs
    longer windows rather than a better estimator.
    """
    alpha = result.alpha
    n_series, n_origins = panel.shape
    if alpha.shape != (n_series,):
        raise ValueError("the decomposition does not match the panel it is being read on")

    cell_noise = series_noise(panel)
    alpha_noise = cell_noise / n_origins          # the noise carried by alpha_f itself

    var_hat = float(np.mean(alpha ** 2))          # alpha is mean-centred by construction
    var_noise = float(np.mean(alpha_noise))
    var_alpha = max(var_hat - var_noise, 0.0)

    def as_share(variance: float) -> float:
        return n_series * n_origins * variance / result.total

    def removing(columns: np.ndarray) -> dict:
        explained = _explained_variance(alpha, columns)
        return {
            "r2": explained / var_alpha if var_alpha > 0 else float("nan"),
            "share_corrected": as_share(max(var_alpha - explained, 0.0)),
            "explained_variance": explained,
        }

    prepared: dict[str, np.ndarray] = {}
    for name, values in covariates.items():
        column = np.asarray(values, dtype=float)
        if column.shape != alpha.shape:
            raise ValueError(f"covariate {name!r} has {column.shape}, expected {alpha.shape}")
        if not np.isfinite(column).all():
            raise ValueError(f"covariate {name!r} has missing or infinite values")
        prepared[name] = column

    per_covariate = {}
    for name, column in prepared.items():
        per_covariate[name] = {"spearman": spearman(alpha, column),
                               **removing(column[:, None])}

    control = np.log(np.maximum(cell_noise, 1e-12))
    if "noise level (control)" not in per_covariate and np.ptp(control) > 0:
        per_covariate["noise level (control)"] = {"spearman": spearman(alpha, control),
                                                  **removing(control[:, None])}

    all_covariates = (removing(np.column_stack(list(prepared.values())))
                      if prepared else {"r2": float("nan"),
                                        "share_corrected": as_share(var_alpha),
                                        "explained_variance": 0.0})

    return {
        "share_raw": result.ss_frame / result.total,
        "share_corrected_global": result.ss_frame_adjusted / result.total,
        "share_corrected": as_share(var_alpha),
        "per_covariate": per_covariate,
        "all_covariates": all_covariates,
        "series_noise": cell_noise,
        "var_alpha_hat": var_hat,
        "var_alpha": var_alpha,
        "var_noise": var_noise,
        "n_series": n_series,
    }


def select_series(panel: Panel, keep: np.ndarray) -> Panel:
    """A sub-panel of the series `keep` selects, for a question only some of them can answer.

    Needed because the panel's membership rule and a shape's membership rule are different
    rules. A series joins the panel by being scorable at every origin, which asks for 56
    on-sale days before the first one; a weekly shape asks for `min_weeks` *complete* weeks.
    An item that launched a few months before the panel starts satisfies the first and not
    the second, so a test that needs both has a smaller population than the panel -- and
    that population has to be named and counted rather than assumed away.

    The sub-panel is a real panel: re-decompose it rather than reusing the full panel's
    `alpha`, so `mu`, the origin effects and the variance budget are all computed on the
    series the question is actually asked about. `dropped` records what the selection cost.
    """
    keep = np.asarray(keep)
    n_series = panel.x.shape[0]
    if keep.dtype == bool:
        if keep.shape != (n_series,):
            raise ValueError(f"a boolean mask must have {n_series} entries, got {keep.shape}")
        rows = np.flatnonzero(keep)
    else:
        rows = keep.astype(int)
        if rows.size and (rows.min() < 0 or rows.max() >= n_series):
            raise ValueError("a row index is outside the panel")
    if rows.size == 0:
        raise ValueError("the selection is empty")

    return Panel(
        x=panel.x[rows], odd=panel.odd[rows], even=panel.even[rows],
        ids=[panel.ids[i] for i in rows], origins=panel.origins,
        dropped={**panel.dropped,
                 "series_before_selection": n_series,
                 "series_selected": int(rows.size)},
    )


def shape_explains_alpha(
    panel: Panel,
    result: Decomposition,
    shapes: np.ndarray,
    covariates: dict[str, np.ndarray],
    *,
    shape_reliability: float | None = None,
    strata_from: str | None = None,
    n_strata: int = 10,
    n_permutations: int = 500,
    seed: int = 0,
) -> dict:
    """Does the weekly shape carry any of the difficulty that volume does not explain?

    `explain_alpha` answers what the stable item effect *is*; on this panel most of it is
    volume. This answers the question that follows, and it is the one that decides whether
    a clustering of the weekly shape is worth repairing: **once volume and sparsity are
    out, does an item's position in the 7-dimensional shape space predict anything about
    how hard it is to forecast?** If it does not, no verdict about types in that space --
    however well measured -- can reach the difficulty that is left.

    `shapes` is `(n_series, 7)`, one row per entry of `panel.ids`, in that order; aligning
    them is the caller's job and a mismatch raises. `covariates` is the nuisance block, the
    same one `explain_alpha` is given. Both should be built from history **before the first
    origin of the panel**, so they are prior properties of the item rather than summaries
    of the windows being scored.

    Three things make the answer readable rather than merely computed.

    **The shape vector sums to 1**, so its seven coordinates are collinear with the
    regression's own intercept. Six are used and the seventh is implied; which one is
    dropped does not change the fitted subspace.

    **Six extra regressors explain variance by construction** -- about `k/n` of what is
    left, which is 4% here. `_explained_variance` removes that in expectation, and a
    **stratified permutation test** gives the distribution rather than the expectation.
    The permutation shuffles the shape rows *within strata of `strata_from`* (volume, in
    practice), so the null keeps whatever relationship shape has with volume and destroys
    only its link to `alpha_f`. Permuting freely instead would break the shape-volume
    relationship too, making the null easier to beat and biasing the test toward finding
    an effect -- the same mistake that makes an over-diffuse continuum null unreadable.

    **Shape is itself measured with error**, and error in a *predictor* attenuates the
    variance it explains toward zero -- the mirror image of the attenuation in
    `rank_agreement`. So a null result here is confounded with the reliability of the
    shape. Pass `shape_reliability` (the split-half reliability of the pooled profile, on
    these series) and `increment_disattenuated` reports the increment divided by it: an
    approximate upper bound on what a perfectly measured shape could contribute. Dividing
    a multivariate increment by a single scalar is an approximation, and it is the right
    way round -- it can only make a negative result weaker, never stronger. It can also
    exceed 1 when the increment is large relative to the reliability, which means the
    scalar approximation has broken down rather than that shape explains everything.

    One reading note on the output: `share_after_shape` can come out *above*
    `share_baseline`. That is not a bug. The explained variance is adjusted for the
    degrees of freedom the six shape columns cost, so a shape block that contributes less
    than its own degrees of freedom leaves less explained than the nuisance block alone --
    and a share that goes up when shape is added is itself the answer to the question.
    """
    alpha = result.alpha
    n_series, n_origins = panel.shape
    shapes = np.asarray(shapes, dtype=float)
    if shapes.shape[0] != n_series:
        raise ValueError(
            f"shapes has {shapes.shape[0]} rows, the panel has {n_series}: align the "
            "shape table on panel.ids before calling this"
        )
    if not np.isfinite(shapes).all():
        raise ValueError("the shape matrix has missing or infinite values")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    nuisance_names = list(covariates)
    nuisance = np.column_stack([np.asarray(covariates[name], dtype=float)
                                for name in nuisance_names]) if covariates else None
    if nuisance is not None:
        if nuisance.shape[0] != n_series:
            raise ValueError("a covariate does not match the panel")
        if not np.isfinite(nuisance).all():
            raise ValueError("a covariate has missing or infinite values")

    # The seventh coordinate is implied by the other six and the intercept.
    shape_columns = shapes[:, :-1]

    var_alpha = max(float(np.mean(alpha ** 2)) - float(np.mean(series_noise(panel)))
                    / n_origins, 0.0)
    if var_alpha <= 0:
        raise ValueError("no stable item effect is left to explain once noise is removed")

    def explained(columns: np.ndarray | None) -> float:
        if columns is None or columns.shape[1] == 0:
            return 0.0
        return _explained_variance(alpha, columns)

    def with_shape(block: np.ndarray) -> np.ndarray:
        return block if nuisance is None else np.column_stack([nuisance, block])

    explained_baseline = explained(nuisance)
    explained_full = explained(with_shape(shape_columns))
    increment = max(explained_full - explained_baseline, 0.0) / var_alpha

    # Strata that keep the shape-volume relationship intact under permutation.
    if strata_from is not None:
        if strata_from not in covariates:
            raise ValueError(f"strata_from={strata_from!r} is not one of the covariates")
        key = np.asarray(covariates[strata_from], dtype=float)
        ranks = np.argsort(np.argsort(key))
        strata = np.minimum(ranks * n_strata // n_series, n_strata - 1)
    else:
        strata = np.zeros(n_series, dtype=int)

    generator = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for draw in range(n_permutations):
        order = np.arange(n_series)
        for stratum in np.unique(strata):
            members = np.flatnonzero(strata == stratum)
            order[members] = generator.permutation(members)
        shuffled = explained(with_shape(shape_columns[order]))
        null[draw] = max(shuffled - explained_baseline, 0.0) / var_alpha

    n_at_least = int(np.sum(null >= increment))
    return {
        "n_series": n_series,
        "n_shape_columns": int(shape_columns.shape[1]),
        "nuisance": nuisance_names,
        "share_baseline": n_series * n_origins
                          * max(var_alpha - explained_baseline, 0.0) / result.total,
        "share_after_shape": n_series * n_origins
                             * max(var_alpha - explained_full, 0.0) / result.total,
        "increment": increment,
        "increment_disattenuated": (increment / shape_reliability
                                    if shape_reliability else None),
        "shape_reliability": shape_reliability,
        "null_median": float(np.median(null)),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": (1 + n_at_least) / (1 + n_permutations),
        "p_value_floor": 1 / (1 + n_permutations),
        "beat_all_nulls": n_at_least == 0,
        "n_permutations": n_permutations,
        "n_strata": int(len(np.unique(strata))),
        "null_increments": null,
        "spearman_per_weekday": [spearman(alpha, shapes[:, d])
                                 for d in range(shapes.shape[1])],
    }


# --- The verdict, written down so it cannot be read wishfully -----------------------

def verdict(result: Decomposition, ranks: dict,
            intervals: dict[str, tuple[float, float]] | None = None) -> str:
    """The reading rule, fixed in advance.

    Thresholds from the published literature: a Spearman of 0.6 or more and a third of the
    variance on the series means a static grouping has something real to hold; 0.3 or
    less, or under a fifth of the variance, means it has not. Between the two, the
    honest answer is "partial, weigh it against the complexity".
    """
    computed = result.shares()
    share = computed["series_adjusted"]
    agreement = ranks["consecutive_median"]
    lines = []

    if intervals is not None:
        low, high = intervals["persistent"]
        if high - low > 0.20:
            lines.append(
                f"UNDERPOWERED: the persistence interval spans {100 * (high - low):.0f} "
                "points -- the verdict below is provisional."
            )

    if agreement >= 0.6 and share >= 0.35:
        lines.append("Difficulty IS persistent: a static grouping has something real to hold.")
    elif agreement <= 0.3 or share < 0.20:
        lines.append("Difficulty is NOT stable: no static grouping can help. Ceiling.")
    else:
        lines.append("Partial persistence: weigh the gain against the complexity.")

    lines.append(
        f"  series (bias-adjusted) {100 * share:.1f}% of the variance, "
        f"median consecutive-origin Spearman {agreement:+.2f}, "
        f"first vs last {ranks['first_vs_last']:+.2f}"
    )
    lines.append(
        f"  stable at all (series + persistent) {100 * computed['stable']:.1f}%, "
        f"of which persistent {100 * computed['persistent']:.1f}%"
    )
    if computed["persistent"] > 2 * max(share, 1e-9):
        lines.append(
            "  Most of what is stable is drifting rather than fixed: a grouping would "
            "have to be refitted, not fixed once."
        )
    return "\n".join(lines)


# --- A panel whose answer is known, to check the estimators above --------------------

def simulate_panel(n_series: int = 250, n_origins: int = N_ORIGINS, mu: float = -1.0,
                   sd_series: float = 0.30, sd_origin: float = 0.20,
                   sd_persistent: float = 0.25, rho: float = 0.85,
                   sd_transient: float = 0.60, sd_noise: float = 0.40,
                   seed: int = 0) -> tuple[Panel, dict[str, float]]:
    """A panel built from known components, and the variances it was built with.

    The positive control of this whole section: a negative result on real data is
    uninterpretable unless the estimator is known to recover a planted answer at the
    same panel size. Returns the panel and the *realised* variances, computed from the
    arrays actually drawn rather than from the parameters, so the comparison is not
    confounded by sampling in the simulation itself.

    One subtlety worth stating: the persistent component is an AR(1) in the origin, and
    `v_persistent` estimates a lag-1 autocovariance, so the quantity it targets is
    `rho * var(persistent)` -- not the variance of that component.
    """
    generator = np.random.default_rng(seed)

    alpha = generator.normal(0, sd_series, n_series)
    beta = generator.normal(0, sd_origin, n_origins)

    persistent = np.zeros((n_series, n_origins))
    innovation = sd_persistent * np.sqrt(1 - rho ** 2)
    persistent[:, 0] = generator.normal(0, sd_persistent, n_series)
    for o in range(1, n_origins):
        persistent[:, o] = rho * persistent[:, o - 1] + generator.normal(0, innovation, n_series)

    transient = generator.normal(0, sd_transient, (n_series, n_origins))
    truth = mu + alpha[:, None] + beta[None, :] + persistent + transient

    # Each cell is read twice -- the odd days and the even days of its window -- and
    # the cell's own value is their mean, exactly as `cell_wql` produces them.
    odd = truth + generator.normal(0, sd_noise, (n_series, n_origins))
    even = truth + generator.normal(0, sd_noise, (n_series, n_origins))
    x = 0.5 * (odd + even)

    centred = x - x.mean()
    total = float(np.sum(centred ** 2))
    cells = n_series * n_origins
    realised = {
        "series": float(n_origins * np.sum((alpha - alpha.mean()) ** 2) / total),
        "origin": float(n_series * np.sum((beta - beta.mean()) ** 2) / total),
        "persistent": float(cells * rho * np.var(persistent) / total),
        "transient": float(cells * np.var(transient) / total),
        "noise": float(cells * np.var(0.5 * (odd - even)) / total),
    }
    panel = Panel(
        x=x, odd=odd, even=even,
        ids=[f"sim_{i:04d}" for i in range(n_series)],
        origins=[date(2016, 3, 27) - timedelta(days=SEGMENT_DAYS * k)
                 for k in reversed(range(n_origins))],
        dropped={"series_seen": n_series, "series_kept": n_series, "cells_with_zero_wql": 0},
    )
    return panel, realised
