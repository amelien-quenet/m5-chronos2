"""The weekly shape of a series, and whether the population has any structure.

The second half of the last question of the study. Notebook 09 shows that difficulty
is not a stable property of a series; this module asks the independent question
underneath it: does the population contain **discrete types** at all, or is it one
continuum? If it is a continuum, then cross-learning had no neighbour to learn from
and a grouping key was always an arbitrary cut -- which turns two observed failures
into one explained fact.

The object being clustered
--------------------------
Not the level of a series -- that would sort items by how much they sell -- but its
**shape**: the share of a week that falls on each day of the week. The original study
worked on hourly data and used a 24-dimensional hour-of-day profile; M5 is daily, so
the strict analogue swaps "day/hour" for "week/weekday" and the profile has 7
dimensions. For a series `f` observed over weeks `w = 1..W`:

    alpha[w, d] = y[w, d] / sum_d y[w, d]                   each week normalised to 1
    v[d]        = sum_w alpha[w, d] / sum_d sum_w alpha[w, d]

The second normalisation is a division by `W`, so `v` is the mean over weeks of the
weekly shares, and it sums to 1 by construction. Written as two normalisations rather
than one average because that is what the construction means: every week gets an equal
say regardless of its volume, which is the point of looking at shape.

Why an audit, and not just a clustering
---------------------------------------
Any clustering algorithm returns clusters. The six questions below are what separate
a finding from a plot, and each can kill the next:

- **step 0** -- is the object stable enough to cluster at all? (ICC, split-half
  reliability, and the counting-noise floor, because normalising to shares makes a
  low-volume series pure noise -- the most likely source of a spurious cluster.)
- **step 1** -- could the pipeline find structure if there were any? Plant K
  archetypes at this sample size and this noise, and measure the separation below
  which absence of evidence is not evidence of absence.
- **step 2** -- does the pipeline hallucinate structure? Run it on a continuum fitted
  to the real cloud's own geometry, and report an empirical p-value.
- **step 4** -- is the metric measuring shape, or clock offset? (Numbered as in the
  playbook; steps 3, 5 and 6 are out of scope and named in the notebook.)

As in `persistence.py`, the polars half is thin and the statistics are numpy, so every
estimator here can be checked against data whose answer is known.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

WEEKDAYS = 7

# Tibshirani-Walther's threshold for a reproducible cluster count, and the adjusted
# Rand index above which the split itself -- not just the count -- is right.
PREDICTION_STRENGTH_THRESHOLD = 0.80
ARI_THRESHOLD = 0.80


# --- The embedding --------------------------------------------------------------------

def complete_weeks(sales: pl.DataFrame, last_day=None, week_start: int = 1) -> pl.DataFrame:
    """Tag each day with the week it belongs to, keeping only usable weeks.

    A week is usable when all 7 of its days are observed and on sale, and the week sold
    something. The first condition keeps a partial week from pretending to be a shape;
    the second is forced by the normalisation, which divides by the week's total.

    `week_start` is the weekday a week begins on, as polars numbers them (1 = Monday).
    It only rotates the labels, never the partition into weeks, so it cannot change any
    conclusion -- but it decides which day is drawn first in the figures.
    """
    table = sales.filter(~pl.col("is_missing"))
    if last_day is not None:
        table = table.filter(pl.col("date") <= last_day)

    anchor = table["date"].min()
    # Shift the anchor back to the previous `week_start`, so week boundaries fall on
    # that weekday rather than wherever the data happens to begin.
    offset = (anchor.isoweekday() - week_start) % WEEKDAYS
    table = table.with_columns(
        (((pl.col("date") - pl.lit(anchor)).dt.total_days() + offset) // WEEKDAYS).alias("week"),
        (((pl.col("date").dt.weekday() - week_start) % WEEKDAYS)).alias("weekday"),
    )

    weeks = table.group_by("id", "week").agg(
        pl.len().alias("n_days"),
        pl.col("sales").sum().alias("week_total"),
    )
    usable = weeks.filter((pl.col("n_days") == WEEKDAYS) & (pl.col("week_total") > 0))
    return table.join(usable.select("id", "week", "week_total"), on=["id", "week"], how="inner")


def weekly_shapes(sales: pl.DataFrame, last_day=None, min_weeks: int = 8,
                  statistic: str = "mean", week_start: int = 1) -> pl.DataFrame:
    """One 7-dimensional shape per series: 'id', 'v0'..'v6', 'n_weeks', 'total'.

    `statistic` selects how the weekly shares are pooled. 'mean' is the construction of
    the report, and the two normalisations then collapse to a single division by the
    number of weeks. 'median' is the more robust variant -- worth running once as a
    check, since M5 weeks are bursty -- but a vector of medians does not sum to 1, so it
    is renormalised afterwards and is no longer exactly the stated construction.
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")

    weeks = complete_weeks(sales, last_day=last_day, week_start=week_start)
    shares = weeks.with_columns((pl.col("sales") / pl.col("week_total")).alias("share"))

    pooled = getattr(pl.col("share"), statistic)()
    table = (
        shares.group_by("id", "weekday")
        .agg(pooled.alias("v"), pl.col("week").n_unique().alias("n_weeks"),
             pl.col("sales").sum().alias("total"))
        .sort("id", "weekday")
    )
    wide = table.pivot(on="weekday", index="id", values="v").sort("id")
    wide.columns = ["id"] + [f"v{int(name)}" for name in wide.columns[1:]]

    counts = table.group_by("id").agg(
        pl.col("n_weeks").max().alias("n_weeks"), pl.col("total").sum().alias("total"))
    wide = wide.join(counts, on="id", how="left").filter(pl.col("n_weeks") >= min_weeks)

    columns = [f"v{d}" for d in range(WEEKDAYS)]
    row_sum = pl.sum_horizontal(columns)
    return wide.with_columns([(pl.col(c) / row_sum).alias(c) for c in columns]).sort("id")


def weekly_share_array(sales: pl.DataFrame, last_day=None, week_start: int = 1):
    """Per-week shares, for the reliability work of step 0: it needs single weeks, not
    the pooled profile.

    Returns (series ids, list of W_f x 7 arrays, list of W_f weekly totals).
    """
    weeks = complete_weeks(sales, last_day=last_day, week_start=week_start)
    shares = weeks.with_columns((pl.col("sales") / pl.col("week_total")).alias("share"))
    shares = shares.sort("id", "week", "weekday")

    ids, blocks, totals = [], [], []
    for (series_id,), part in shares.group_by("id", maintain_order=True):
        values = part["share"].to_numpy().reshape(-1, WEEKDAYS)
        week_totals = part["week_total"].to_numpy().reshape(-1, WEEKDAYS)[:, 0]
        ids.append(series_id)
        blocks.append(values)
        totals.append(week_totals)
    return ids, blocks, totals


def shape_matrix(shapes: pl.DataFrame) -> np.ndarray:
    """The 'v0'..'v6' columns as an (n_series, 7) array, in the table's row order."""
    return shapes.select([f"v{d}" for d in range(WEEKDAYS)]).to_numpy()


# --- Step 0: is the object stable? ----------------------------------------------------

def icc1(blocks: list[np.ndarray]) -> np.ndarray:
    """ICC(1) per coordinate: the share of variance that sits *between* series.

    One number per weekday, computed on single-week profiles. A low value means a single
    week says little about its series -- which is survivable, because clustering consumes
    the pooled profile, not single weeks. That is what `split_half_reliability` measures,
    and it is the number that actually decides whether to go on.
    """
    counts = np.array([len(block) for block in blocks], dtype=float)
    means = np.array([block.mean(axis=0) for block in blocks])
    grand = np.average(means, axis=0, weights=counts)

    between = np.sum(counts[:, None] * (means - grand) ** 2, axis=0)
    within = np.sum([np.sum((block - block.mean(axis=0)) ** 2, axis=0) for block in blocks], axis=0)

    n_series, total = len(blocks), counts.sum()
    between_ms = between / max(n_series - 1, 1)
    within_ms = within / max(total - n_series, 1)
    # Unequal group sizes: the usual n0 correction.
    n0 = (total - np.sum(counts ** 2) / total) / max(n_series - 1, 1)
    variance = (between_ms - within_ms) / max(n0, 1e-12)
    return np.clip(variance / np.maximum(variance + within_ms, 1e-12), 0.0, 1.0)


def split_half_reliability(blocks: list[np.ndarray], n_draws: int = 20,
                           seed: int = 0) -> float:
    """Spearman-Brown corrected reliability of the pooled profile.

    Split each series' weeks in two at random, pool each half, and correlate the two
    halves across series. Each half is built from W/2 weeks while the real profile uses
    W, so the raw correlation understates it; Spearman-Brown undoes exactly that.

    This is the number that matters: clustering sees the pooled profile. A population
    whose single weeks are noise can still have perfectly reliable profiles.
    """
    generator = np.random.default_rng(seed)
    usable = [block for block in blocks if len(block) >= 4]
    if len(usable) < 3:
        return float("nan")

    scores = []
    for _ in range(n_draws):
        first, second = [], []
        for block in usable:
            order = generator.permutation(len(block))
            middle = len(block) // 2
            first.append(block[order[:middle]].mean(axis=0))
            second.append(block[order[middle: 2 * middle]].mean(axis=0))
        first, second = np.array(first), np.array(second)
        # One correlation per coordinate, across series, then averaged.
        per_coordinate = []
        for d in range(first.shape[1]):
            a, b = first[:, d] - first[:, d].mean(), second[:, d] - second[:, d].mean()
            denominator = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
            if denominator > 0:
                per_coordinate.append(float(np.sum(a * b) / denominator))
        raw = float(np.mean(per_coordinate)) if per_coordinate else 0.0
        scores.append(2 * raw / (1 + raw) if raw > -1 else 0.0)   # Spearman-Brown, halves
    return float(np.mean(scores))


def counting_noise_ratio(blocks: list[np.ndarray], totals: list[np.ndarray]) -> np.ndarray:
    """Observed within-series variance over the multinomial counting floor, per series.

    Normalising counts into shares makes a low-volume series look wildly variable for
    purely arithmetic reasons: a week selling 3 units can only produce a handful of
    distinct shape vectors. This is the ratio that tells "this series genuinely changes
    shape week to week" apart from "this series sells too little for the question to
    mean anything". A value near 1 is indistinguishable from pure counting noise.
    """
    ratios = []
    for block, week_totals in zip(blocks, totals):
        if len(block) < 2:
            ratios.append(np.nan)
            continue
        observed = float(np.mean(np.var(block, axis=0, ddof=1)))
        profile = block.mean(axis=0)
        # Var of a multinomial share at total n is p(1-p)/n, averaged over weeks and days.
        expected = float(np.mean([
            np.mean(profile * (1 - profile)) / max(total, 1) for total in week_totals
        ]))
        ratios.append(observed / expected if expected > 0 else np.nan)
    return np.array(ratios)


# --- Clustering, and whether to believe it --------------------------------------------

def _fit_gmm(x: np.ndarray, k: int, seed: int = 0, covariance_type: str = "full"):
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(n_components=k, covariance_type=covariance_type,
                           random_state=seed, n_init=3).fit(x)


def prediction_strength(x: np.ndarray, k: int, n_draws: int = 10, seed: int = 0,
                        covariance_type: str = "full") -> float:
    """Tibshirani-Walther prediction strength at K = `k`.

    Split the sample in two, cluster each half, then use the *training* half's model to
    label the test half and ask: within each cluster the test half found on its own,
    what fraction of pairs the training model also puts together? The score is the worst
    cluster's fraction, which is why it is strict -- one unreproducible cluster sinks it.

    It asks reproducibility rather than fit, and that is the whole point: every
    criterion that rewards fit (likelihood, BIC, silhouette) will happily split a
    continuum.
    """
    generator = np.random.default_rng(seed)
    if k <= 1:
        return 1.0
    scores = []
    for draw in range(n_draws):
        order = generator.permutation(len(x))
        middle = len(x) // 2
        train, test = x[order[:middle]], x[order[middle:]]
        if min(len(train), len(test)) <= k:
            continue
        try:
            train_model = _fit_gmm(train, k, seed=seed + draw, covariance_type=covariance_type)
            test_model = _fit_gmm(test, k, seed=seed + draw, covariance_type=covariance_type)
        except ValueError:
            continue
        by_train = train_model.predict(test)
        by_test = test_model.predict(test)

        worst = 1.0
        for cluster in range(k):
            members = np.flatnonzero(by_test == cluster)
            if len(members) < 2:
                continue
            labels = by_train[members]
            same = (labels[:, None] == labels[None, :])
            np.fill_diagonal(same, False)
            worst = min(worst, same.sum() / (len(members) * (len(members) - 1)))
        scores.append(worst)
    return float(np.mean(scores)) if scores else float("nan")


def k_hat(x: np.ndarray, max_k: int = 6, n_draws: int = 10, seed: int = 0,
          threshold: float = PREDICTION_STRENGTH_THRESHOLD,
          covariance_type: str = "full") -> tuple[int, dict[int, float]]:
    """The selection rule, biased toward "no structure": the largest K that clears the
    threshold, and 1 if none does.

    Returns (K_hat, the prediction strength at each K). Reporting the whole curve
    matters -- a K_hat of 1 because every K scored 0.79 is a different situation from
    one because every K scored 0.3.
    """
    strengths = {
        k: prediction_strength(x, k, n_draws=n_draws, seed=seed, covariance_type=covariance_type)
        for k in range(2, max_k + 1)
    }
    clearing = [k for k, score in strengths.items() if np.isfinite(score) and score > threshold]
    return (max(clearing) if clearing else 1), strengths


def separation(x: np.ndarray, labels: np.ndarray) -> float:
    """delta = smallest gap between centroids, over the pooled cloud radius.

    The knob a simulation turns is arbitrary; this is the separation it actually
    achieved, and it is comparable across settings in a way the knob is not.
    """
    groups = np.unique(labels)
    centroids = np.array([x[labels == group].mean(axis=0) for group in groups])
    radius = float(np.sqrt(np.mean([
        np.mean(np.sum((x[labels == group] - centroid) ** 2, axis=1))
        for group, centroid in zip(groups, centroids)
    ])))
    gaps = [
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i in range(len(groups)) for j in range(i + 1, len(groups))
    ]
    return min(gaps) / radius if radius > 0 and gaps else float("inf")


# --- Step 1: the positive control -----------------------------------------------------

@dataclass
class Simulation:
    shapes: np.ndarray          # n_series x 7, pooled profiles
    labels: np.ndarray          # n_series, the planted archetype (all zeros for a null)
    separation: float


def _resample_weeks(profiles: np.ndarray, totals: list[np.ndarray],
                    generator: np.random.Generator) -> np.ndarray:
    """Redraw each series' weeks as multinomial counts at its real weekly totals.

    This is what makes a simulation comparable to the real data rather than merely
    shaped like it: the counting noise of a 3-unit week is reproduced, not assumed away.
    """
    pooled = []
    for profile, week_totals in zip(profiles, totals):
        profile = np.clip(profile, 1e-9, None)
        profile = profile / profile.sum()
        weeks = []
        for total in week_totals:
            counts = generator.multinomial(max(int(round(total)), 1), profile)
            if counts.sum() > 0:
                weeks.append(counts / counts.sum())
        pooled.append(np.mean(weeks, axis=0) if weeks else profile)
    return np.array(pooled)


def positive_control(base_profile: np.ndarray, totals: list[np.ndarray], n_clusters: int = 4,
                     sep: float = 1.0, scatter: float = 0.25, seed: int = 0) -> Simulation:
    """K genuinely distinct archetypes, at this dataset's sample size and noise.

    The archetypes ride on a **base profile taken from the real data**, so the real
    envelope and its quirks are present rather than invented. Each series is scattered
    around its archetype in log space -- the continuum component, which is what keeps
    the control from being trivially easy -- and its weeks are redrawn as multinomial
    counts at its real totals.

    `sep` scales the archetype offsets; the separation actually achieved is measured and
    returned, because that is the comparable quantity.
    """
    generator = np.random.default_rng(seed)
    n_series = len(totals)
    base = np.log(np.clip(base_profile, 1e-9, None))

    # Directions that are contrasts between parts of the week: a weekend-heavy profile,
    # a weekday-heavy one, a mid-week one, a flat one. Orthogonalised so that `sep`
    # moves each archetype an equal distance from the others.
    raw = np.array([
        [-1, -1, -1, -1, -1, 2.5, 2.5],      # weekend
        [1, 1, 1, 1, 1, -2.5, -2.5],         # weekday
        [-1, 1, 2, 1, -1, -1, -1],           # mid-week
        [0, 0, 0, 0, 0, 0, 0],               # the base profile itself
    ], dtype=float)[:n_clusters]
    raw = raw - raw.mean(axis=1, keepdims=True)

    labels = generator.integers(0, n_clusters, n_series)
    offsets = sep * raw[labels]
    noise = scatter * generator.normal(size=(n_series, WEEKDAYS))
    # A smooth harmonic decay makes the scatter look like a continuum rather than static.
    decay = np.exp(-0.35 * np.arange(WEEKDAYS))
    noise = noise * decay / decay.mean()

    profiles = np.exp(base + offsets + noise)
    profiles = profiles / profiles.sum(axis=1, keepdims=True)
    pooled = _resample_weeks(profiles, totals, generator)
    return Simulation(shapes=pooled, labels=labels, separation=separation(pooled, labels))


# --- Step 2: the null that counts -----------------------------------------------------

def fitted_null(shapes: np.ndarray, totals: list[np.ndarray], n_neighbours: int = 60,
                seed: int = 0) -> Simulation:
    """A continuum fitted to the real cloud's own geometry.

    This is the null that carries evidence, and it is worth being clear about why a
    hand-written one does not. Invented curves interpolated into a family are fine for
    planting structure of a known size (step 1), but nothing about their geometry comes
    from the data, so "the real data scores higher than they do" says nothing.

    Construction: find the dominant axis of variation by SVD, estimate a smooth manifold
    along it as a local mean over `n_neighbours` nearest series, and scatter series
    around it with Gaussian residuals carrying the **real residual covariance**. Weeks
    are then redrawn at the real weekly totals. The result is the real cloud with
    multimodality removed *by construction* -- Gaussian residuals about a smooth
    manifold cannot be multimodal -- while keeping its spread, its dimensions and its
    counting noise.
    """
    generator = np.random.default_rng(seed)
    log_shapes = np.log(np.clip(shapes, 1e-9, None))
    centre = log_shapes.mean(axis=0)
    centred = log_shapes - centre

    # Dominant axis, and each series' position along it.
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    axis = components[0]
    position = centred @ axis

    # The manifold: a local mean over the nearest neighbours *along that axis*.
    order = np.argsort(position)
    ranks = np.empty(len(position), dtype=int)
    ranks[order] = np.arange(len(position))
    half = max(n_neighbours // 2, 1)
    manifold = np.array([
        centred[order[max(rank - half, 0): min(rank + half, len(order))]].mean(axis=0)
        for rank in ranks
    ])

    residual = centred - manifold
    covariance = np.cov(residual, rowvar=False)
    covariance = covariance + 1e-9 * np.eye(WEEKDAYS)
    drawn = generator.multivariate_normal(np.zeros(WEEKDAYS), covariance, size=len(shapes))

    profiles = np.exp(centre + manifold + drawn)
    profiles = profiles / profiles.sum(axis=1, keepdims=True)
    pooled = _resample_weeks(profiles, totals, generator)
    return Simulation(shapes=pooled, labels=np.zeros(len(shapes), dtype=int), separation=0.0)


def null_envelope(shapes: np.ndarray, totals: list[np.ndarray], n_nulls: int = 20,
                  max_k: int = 6, n_draws: int = 6, seed: int = 0) -> dict:
    """Real prediction strength against the null's, with an empirical p-value.

    The headline test is on **max over K**, which needs no multiplicity correction: the
    question "is there any K at which this data clusters more reproducibly than a
    continuum would" is one question, however many K are tried.
    """
    real_k, real_strengths = k_hat(shapes, max_k=max_k, n_draws=n_draws, seed=seed)
    real_max = max(value for value in real_strengths.values() if np.isfinite(value))

    null_maxima, null_curves, null_k = [], [], []
    for draw in range(n_nulls):
        simulated = fitted_null(shapes, totals, seed=seed + 1000 + draw)
        estimate, strengths = k_hat(simulated.shapes, max_k=max_k, n_draws=n_draws,
                                    seed=seed + draw)
        finite = [value for value in strengths.values() if np.isfinite(value)]
        if not finite:
            continue
        null_maxima.append(max(finite))
        null_curves.append(strengths)
        null_k.append(estimate)

    null_maxima = np.array(null_maxima)
    # +1 in both places: the observed value is itself one draw from the null under H0,
    # which keeps the p-value from ever being exactly 0 on a finite number of nulls.
    p_value = (1 + np.sum(null_maxima >= real_max)) / (1 + len(null_maxima))
    return {
        "real_k_hat": real_k,
        "real_strengths": real_strengths,
        "real_max": real_max,
        "null_maxima": null_maxima,
        "null_curves": null_curves,
        "null_k_hat": np.array(null_k),
        "p_value": float(p_value),
        "n_nulls": len(null_maxima),
        # The floor the number of nulls imposes. Beating every null gives exactly this
        # and nothing smaller, so a p-value sitting on its floor means "more nulls
        # needed to say more", not "significant at that level".
        "min_p_value": 1 / (1 + len(null_maxima)),
        "beat_all_nulls": bool(np.all(null_maxima < real_max)) if len(null_maxima) else False,
    }


def geometry_comparison(real: np.ndarray, null: np.ndarray, seed: int = 0) -> dict:
    """Show the null resembles the real cloud, rather than asserting it.

    On the three statistics that actually drive a clustering: the spread of each
    coordinate, the distribution of pairwise distances, and the distribution of
    nearest-neighbour distances. If these disagree, the null is not a null for this
    data and its p-value means nothing.
    """
    def distances(x):
        gram = np.sum(x ** 2, axis=1)
        squared = np.maximum(gram[:, None] + gram[None, :] - 2 * x @ x.T, 0.0)
        matrix = np.sqrt(squared)
        upper = matrix[np.triu_indices(len(x), k=1)]
        np.fill_diagonal(matrix, np.inf)
        return upper, matrix.min(axis=1)

    real_pairs, real_nn = distances(real)
    null_pairs, null_nn = distances(null)
    spread_real, spread_null = real.std(axis=0), null.std(axis=0)

    # Scalar summaries, so the notebook can refuse to read a p-value rather than
    # eyeball three histograms and decide they look fine. The direction matters: a null
    # that is *more* spread out than the real cloud clusters less reproducibly, which
    # makes the real data beat it too easily and biases the p-value downward -- toward
    # finding structure. That is the dangerous direction, so it is the one to check.
    ratios = spread_null / np.maximum(spread_real, 1e-12)
    return {
        "spread_real": spread_real,
        "spread_null": spread_null,
        "pairwise_real": real_pairs,
        "pairwise_null": null_pairs,
        "nearest_real": real_nn,
        "nearest_null": null_nn,
        "spread_ratio": ratios,
        "worst_spread_ratio": float(max(ratios.max(), 1 / ratios.min())),
        "pairwise_ratio": float(np.median(null_pairs) / max(np.median(real_pairs), 1e-12)),
        "nearest_ratio": float(np.median(null_nn) / max(np.median(real_nn), 1e-12)),
    }


# --- Step 4: is the metric measuring shape, or the position of the week? --------------

def _shift_statistics(shapes: np.ndarray) -> dict:
    n_series = len(shapes)
    best_shift, gains = [], []
    for i in range(n_series):
        for j in range(i + 1, n_series):
            base = float(np.linalg.norm(shapes[i] - shapes[j]))
            if base == 0:
                continue
            distances = [float(np.linalg.norm(np.roll(shapes[i], tau) - shapes[j]))
                         for tau in range(WEEKDAYS)]
            tau = int(np.argmin(distances))
            best_shift.append(tau if tau <= WEEKDAYS // 2 else tau - WEEKDAYS)
            gains.append(1 - (min(distances) / base) ** 2)

    best_shift, gains = np.array(best_shift), np.array(gains)
    return {
        "shift": best_shift,
        "gain": gains,
        "share_aligned": float(np.mean(best_shift == 0)) if len(best_shift) else float("nan"),
        "share_within_one": float(np.mean(np.abs(best_shift) <= 1)) if len(best_shift) else float("nan"),
        "median_gain": float(np.median(gains)) if len(gains) else float("nan"),
    }


def shift_test(shapes: np.ndarray, seed: int = 0) -> dict:
    """Would a cyclic rotation of the week bring two series closer together?

    Euclidean distance on a 7-vector is not shift-invariant: two series with the same
    shape one day apart land far apart. That may be exactly right -- Saturday is
    Saturday, and the forecast is made in calendar days -- but it raises a fair
    objection. If a one-day offset is larger than the typical between-series difference,
    the clustering is sorting series by *where their week sits* rather than by shape.

    For every pair, find the rotation that minimises the distance:

        d(tau) = || roll(p_i, tau) - p_j ||,   tau* = argmin over the 7 rotations
        gain   = 1 - (d_best / d_0)^2          share of squared distance removed

    Two notes that make the test exact: use the raw 7-vector, not a truncated
    representation -- truncating discards exactly the fast-rotating components that make
    a one-day shift visible; and no per-coordinate standardisation, which is anisotropic
    and so does not commute with a rotation, making "which series you rotate" an
    arbitrary convention.

    **A week has only 7 rotations, so `tau* = 0` happens 14% of the time by luck alone.**
    The original study worked on 24 hourly positions, where the chance level is 4% and
    the raw share could be read directly; here it cannot. So the test also runs itself on
    a copy of the data with every series randomly rotated -- same cloud, alignment
    destroyed -- and the comparison between the two is the result. The absolute number is
    not.
    """
    observed = _shift_statistics(shapes)

    generator = np.random.default_rng(seed)
    rotated = np.array([np.roll(row, generator.integers(WEEKDAYS)) for row in shapes])
    reference = _shift_statistics(rotated)

    return observed | {
        "chance_level": 1 / WEEKDAYS,
        "reference": reference,
        "share_aligned_reference": reference["share_aligned"],
        "median_gain_reference": reference["median_gain"],
        # > 1 means the real data is more aligned than a deliberately misaligned copy,
        # which is the direction that says the coordinates mean something.
        "alignment_lift": (observed["share_aligned"] / reference["share_aligned"]
                           if reference["share_aligned"] > 0 else float("inf")),
    }


# --- Display only ----------------------------------------------------------------------

def umap_layout(x: np.ndarray, seed: int = 0, n_neighbors: int = 15,
                min_dist: float = 0.1) -> np.ndarray:
    """2-D layout for the figures and for the video. Display only: no conclusion in this
    module or in notebook 10 rests on it, because a projection can separate what is not
    separated and merge what is. The decisions are taken in the full 7 dimensions.
    """
    import umap

    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors,
                        min_dist=min_dist)
    return reducer.fit_transform(x)
