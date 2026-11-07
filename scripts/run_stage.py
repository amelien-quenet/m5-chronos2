#!/usr/bin/env python
"""Stage runner: the study's heavy computation, outside the notebooks.

Principle
---------
A *stage* is a pure function `configuration -> artifacts/<name>.parquet`. It is
self-contained (everything is on the command line, nothing in a kernel's state),
it writes its result to disk, and it is resumable: run again, it skips what
already exists. That is the contract `chronos2.cached()` already implements --
this script only exposes it on the command line, so that a six-hour computation
can be launched and then forgotten.

The stages call exactly the same functions as the notebooks, to exactly the same
paths. So after a run on the tower and an rsync of the artifacts, the notebooks
re-execute end to end in seconds: every `cached()` finds its parquet. The
notebooks stay the narrative layer -- scores, figures, prose -- and compute
nothing.

Usage
-----
    python scripts/run_stage.py --list
    python scripts/run_stage.py --stages vanilla,covariates
    python scripts/run_stage.py --all
    python scripts/run_stage.py --stages context_coarse --force
    python scripts/run_stage.py --stages finetune_test --finetune-context 1829

Routing
-------
A stage marked `needs_gpu` refuses to run on CPU: the measured factor is about
24x, so launching it on the laptop is not a choice but an oversight.
`--allow-slow` lifts the refusal when it is deliberate.

Traceability
------------
Every artifact produced gets a `<name>.parquet.meta.json`: machine, device,
library versions, seed, roster, git commit, duration. A CPU and a GPU do not
return bit-identical numbers; without that record, a comparison table is not
defensible.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import polars as pl

from m5_chronos2.chronos2 import (
    ALL_KNOWN_COVARIATES,
    KNOWN_COVARIATE_BLOCKS,
    cached,
    checkpoint_dir,
    device_report,
    experiment_path,
    finetune,
    finetune_inputs,
    forecast,
    future_window,
    history,
    known_covariate_table,
    past_blank_flags,
    pipeline_dtype,
    get_pipeline,
    resolve_device,
)
from m5_chronos2.data import ARTIFACTS_DIR, HORIZON, roster_name
from m5_chronos2.evaluation import actuals, make_windows
from m5_chronos2.metrics import join_truth, score

# --- The study's configuration, identical to the notebooks -------------------------

SAMPLE_SIZE = 500
SEED = 42

CONTEXT_LENGTH = None          # the model's maximum, as in 05 and 06
CROSS_LEARNING_BATCH = 100     # series per attention group (the vendor's own setting)
SHUFFLE_SEED = 0               # series order for cross-learning

COARSE_GRID = [7, 14, 21, 28, 42, 56, 84, 112, 140, 168, 224, 280, 364,
               448, 560, 728, 896, 1092, 1288, 1456, 1680]
FINE_STEP = 7
FINE_HALF_WIDTH = 56
SELECTION_METRIC = "WQL"
EARLY_LAST_DAY = 7

NUM_STEPS = 1000
LEARNING_RATE = 1e-5
BATCH_SIZE = 32

TRACKED_PACKAGES = ["chronos-forecasting", "torch", "transformers", "peft",
                    "polars", "numpy", "pyarrow"]


# --- Traceability -------------------------------------------------------------------

def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    return versions


def write_meta(path: Path, stage: str, device: str, seconds: float, extra: dict | None = None):
    """The sidecar that says where this file came from. Never an artifact without its meta."""
    meta = {
        "stage": stage,
        "artifact": path.name,
        "produced_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(seconds, 1),
        "roster": roster_name(SAMPLE_SIZE, SEED),
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "git_commit": git_commit(),
        "packages": package_versions(),
        **device_report(device),
    }
    if extra:
        meta |= extra
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2, default=str))


# --- The data, loaded once ----------------------------------------------------------

class Study:
    """Everything the stages share. Loaded lazily: `--list` reads nothing."""

    def __init__(self, device: str):
        self.device = device
        self.name = roster_name(SAMPLE_SIZE, SEED)
        self._cache: dict = {}

    def _get(self, key: str, build: Callable):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def sales(self) -> pl.DataFrame:
        return self._get("sales", lambda: pl.read_parquet(
            ARTIFACTS_DIR / f"sales_flags_{self.name}.parquet"))

    @property
    def windows(self) -> pl.DataFrame:
        return self._get("windows", make_windows)

    @property
    def validation_origin(self) -> date:
        return self.windows.filter(pl.col("role") == "validation")["origin"][0]

    @property
    def test_origin(self) -> date:
        return self.windows.filter(pl.col("role") == "test")["origin"][0]

    @property
    def series_ids(self) -> list[str]:
        return self._get("ids", lambda: self.sales["id"].unique().sort().to_list())

    @property
    def validation_context(self) -> pl.DataFrame:
        return self._get("ctx", lambda: history(self.sales, self.validation_origin))

    @property
    def max_history(self) -> int:
        return self.validation_context.group_by("id").len()["len"].max()

    @property
    def truth(self) -> pl.DataFrame:
        return self._get("truth", lambda: actuals(self.sales))

    @property
    def validation_roster(self) -> pl.DataFrame:
        return self._get("vroster", lambda: pl.read_parquet(
            ARTIFACTS_DIR / f"eval_roster_{self.name}.parquet"
        ).filter(pl.col("role") == "validation"))

    @property
    def choice_path(self) -> Path:
        return ARTIFACTS_DIR / f"context_choice_{self.name}.json"

    @property
    def choice(self) -> dict:
        if not self.choice_path.exists():
            raise SystemExit(
                f"\n{self.choice_path.name} is missing.\n"
                "Choosing the context lengths is not a computation step: it is a decision\n"
                "made by looking at the sweep, in notebook 07, on the Mac.\n"
                "Run context_coarse and context_fine first, pull the artifacts, run\n"
                "notebook 07, then come back."
            )
        return json.loads(self.choice_path.read_text())

    def coarse_grid(self) -> list[int]:
        """Coarse grid, bounded by the longest history.

        Any length >= max_history reads the whole history, so it is equivalent to the
        vanilla model: max_history is kept as the single upper bound.
        """
        return sorted({length for length in COARSE_GRID if length < self.max_history}
                      | {self.max_history})

    def context_run(self, length: int) -> pl.DataFrame:
        return cached(
            experiment_path(f"context_{length}", self.name),
            lambda: forecast(self.validation_context, self.validation_origin,
                             context_length=int(length), device=self.device),
        )

    def segment_winners(self, lengths: list[int]) -> list[int]:
        """Best length per segment (days 1-7 / 8-28), on the selection metric.

        This is the only place where the runner scores anything: the fine grid depends
        on the coarse grid's winners.
        """
        segment = (pl.when(pl.col("horizon") <= EARLY_LAST_DAY)
                   .then(pl.lit("days 01-07")).otherwise(pl.lit("days 08-28")))
        parts = []
        for length in lengths:
            scored = join_truth(self.context_run(length), self.truth, self.validation_roster)
            parts.append(score(scored.with_columns(segment.alias("segment")), by="segment")
                         .with_columns(pl.lit(length).alias("context_length")))
        by_segment = pl.concat(parts)
        return [part.sort(SELECTION_METRIC).row(0, named=True)["context_length"]
                for _, part in by_segment.group_by("segment", maintain_order=True)]


# --- The stages ---------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    description: str
    run: Callable[[Study], list[Path]]
    needs_gpu: bool = False
    cost: str = "light"            # "light" < 10 min, "heavy" beyond that
    requires: list[str] = field(default_factory=list)
    note: str = ""


def stage_vanilla(study: Study) -> list[Path]:
    """Chronos-2 with nothing: no covariates, no cross-learning, maximum context."""
    path = ARTIFACTS_DIR / "predictions" / f"chronos2_vanilla_{study.name}.parquet"

    def compute():
        parts = []
        for origin in study.windows["origin"].to_list():
            start = time.time()
            parts.append(forecast(history(study.sales, origin), origin, device=study.device))
            print(f"  origin {origin}: {len(study.series_ids)} series in {time.time() - start:.0f} s")
        return pl.concat(parts)

    predictions = cached(path, compute)
    expected = len(study.series_ids) * HORIZON * study.windows.height
    assert predictions.height == expected, f"{predictions.height} rows, {expected} expected"
    return [path]


def stage_uplift(study: Study) -> list[Path]:
    """Event-uplift table at the validation origin (taken from the XGBoost feature set)."""
    from m5_chronos2.gbdt_features import (calendar_by_day, event_uplift,
                                           group_daily_means, load_all_series_long)

    path = ARTIFACTS_DIR / "experiments" / f"event_uplift_origin{study.validation_origin}.parquet"
    day = calendar_by_day().filter(pl.col("date") == study.validation_origin)["day"][0]

    def compute():
        all_long = load_all_series_long()
        return event_uplift(group_daily_means(all_long, ["item_id"], day), day)

    cached(path, compute)
    return [path]


def _known_inputs(study: Study):
    uplift = pl.read_parquet(
        ARTIFACTS_DIR / "experiments" / f"event_uplift_origin{study.validation_origin}.parquet")
    covariates = known_covariate_table(study.sales, uplift)
    context_known = study.validation_context.join(covariates, on=["id", "date"], how="left")
    future_known = future_window(covariates, study.validation_origin, study.series_ids)
    assert future_known.height == len(study.series_ids) * HORIZON
    missing = future_known.select(ALL_KNOWN_COVARIATES).null_count().sum_horizontal()[0]
    assert missing == 0, "a known covariate is missing inside the window"
    return context_known, future_known


def stage_covariates(study: Study) -> list[Path]:
    """The four blocks of known covariates: calendar, price, SNAP+events, all of them."""
    context_known, future_known = _known_inputs(study)
    blocks = {**KNOWN_COVARIATE_BLOCKS, "all_known": ALL_KNOWN_COVARIATES}
    paths = []
    for block, names in blocks.items():
        path = experiment_path(f"known_{block}", study.name)
        print(f"  {block}: {names}")
        cached(path, lambda names=names: forecast(
            context_known, study.validation_origin, known_covariates=names,
            future=future_known, context_length=CONTEXT_LENGTH, device=study.device))
        paths.append(path)
    return paths


def stage_cross_learning(study: Study) -> list[Path]:
    """Cross-learning on, series shuffled so the groups are not department-homogeneous."""
    shuffled = pl.Series(study.series_ids).shuffle(seed=SHUFFLE_SEED).to_list()
    order = pl.DataFrame({"id": shuffled}).with_row_index("position")
    context_shuffled = (study.validation_context.join(order, on="id")
                        .sort("position", "date").drop("position"))
    path = experiment_path("cross_learning", study.name)
    cached(path, lambda: forecast(
        context_shuffled, study.validation_origin, context_length=CONTEXT_LENGTH,
        cross_learning=True, batch_size=CROSS_LEARNING_BATCH, device=study.device))
    return [path]


def stage_blank(study: Study) -> list[Path]:
    """Three ways to flag blank periods: past covariate, second channel, NaN mask."""
    context_blank = past_blank_flags(study.validation_context).with_columns(
        pl.col("is_blank").cast(pl.Float64))
    context_masked = context_blank.with_columns(
        pl.when(pl.col("is_blank") == 1).then(None).otherwise(pl.col("sales")).alias("sales"))

    paths = []
    for name, compute in [
        ("blank_past_covariate", lambda: forecast(
            context_blank, study.validation_origin, past_covariates=["is_blank"],
            context_length=CONTEXT_LENGTH, device=study.device)),
        ("blank_second_target_both_channels", lambda: forecast(
            context_blank, study.validation_origin, target=["sales", "is_blank"],
            context_length=CONTEXT_LENGTH, device=study.device)),
        ("blank_nan_mask", lambda: forecast(
            context_masked, study.validation_origin,
            context_length=CONTEXT_LENGTH, device=study.device)),
    ]:
        path = experiment_path(name, study.name)
        print(f"  {name}")
        cached(path, compute)
        paths.append(path)
    return paths


def stage_context_coarse(study: Study) -> list[Path]:
    """Coarse grid of context lengths, log-spaced."""
    grid = study.coarse_grid()
    print(f"  coarse grid ({len(grid)} lengths): {grid}")
    paths = []
    for length in grid:
        study.context_run(length)
        paths.append(experiment_path(f"context_{length}", study.name))
    return paths


def stage_context_fine(study: Study) -> list[Path]:
    """Fine grid, one-week steps within +/- 8 weeks of the coarse grid's winners."""
    grid = study.coarse_grid()
    winners = study.segment_winners(grid)
    print(f"  coarse winners: {winners}")
    fine = sorted({
        length
        for winner in winners
        for length in range(max(FINE_STEP, winner - FINE_HALF_WIDTH),
                            min(study.max_history, winner + FINE_HALF_WIDTH) + 1, FINE_STEP)
    } - set(grid))
    print(f"  fine grid ({len(fine)} new lengths): {fine}")
    paths = []
    for length in fine:
        study.context_run(length)
        paths.append(experiment_path(f"context_{length}", study.name))
    return paths


def _finetune_at(study: Study, origin: date, context_length: int, inputs: tuple) -> Path:
    output_dir = checkpoint_dir(study.name, origin, context_length,
                               NUM_STEPS, LEARNING_RATE, BATCH_SIZE)
    print(f"  origin {origin}, fine-tune at context {context_length}")
    finetune(*inputs, context_length=context_length, output_dir=output_dir,
             num_steps=NUM_STEPS, learning_rate=LEARNING_RATE,
             batch_size=BATCH_SIZE, device=study.device)
    return output_dir / "finetuned-ckpt"


def stage_finetune(study: Study) -> list[Path]:
    """LoRA at the two lengths notebook 07 selected, validation origin."""
    choice = study.choice
    contexts = sorted({choice["late_context"], choice["early_context"]},
                      key=lambda length: length != choice["late_context"])   # the long one first
    inputs = finetune_inputs(study.sales, study.validation_origin, study.series_ids)
    return [_finetune_at(study, study.validation_origin, length, inputs) for length in contexts]


def stage_finetune_predict(study: Study) -> list[Path]:
    """Forecasts from the LoRA checkpoints, every fine-tune x predict-context pairing."""
    choice = study.choice
    finetune_contexts = sorted({choice["late_context"], choice["early_context"]})
    predict_contexts = sorted({choice["early_context"], choice["late_context"]})
    origin = study.validation_origin
    paths = []
    for finetune_context in finetune_contexts:
        checkpoint = checkpoint_dir(study.name, origin, finetune_context, NUM_STEPS,
                                    LEARNING_RATE, BATCH_SIZE) / "finetuned-ckpt"
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint missing: {checkpoint}\nrun the finetune stage first")
        for length in predict_contexts:
            name = (f"lora_ctx{finetune_context}_steps{NUM_STEPS}_lr{LEARNING_RATE:g}"
                    f"_bs{BATCH_SIZE}_context_{length}_origin{origin}")
            path = experiment_path(name, study.name)
            print(f"  LoRA@{finetune_context} -> forecast at context {length}")
            cached(path, lambda length=length, checkpoint=checkpoint: forecast(
                history(study.sales, origin), origin, context_length=int(length),
                model_path=checkpoint, device=study.device))
            paths.append(path)
    return paths


def stage_origin_sweep(study: Study) -> list[Path]:
    """Zero-shot Chronos-2 at 24 origins 7 days apart, counting back from validation.

    The raw material notebook 09 needs: is forecast difficulty a property of the series
    or of the week? Answering that takes one error per (series, origin) pair, so as many
    forecasts as there are origins.

    Three choices, all of them dictated by keeping the measurement honest:

    - **the zero-shot model, not the best model.** The final model is a stitch involving
      a LoRA checkpoint trained on the history up to the test origin. Walking it
      backwards would let it see the future of every earlier window, and re-fine-tuning
      24 times is not affordable. So this takes the untuned model at the context length
      selected for days 1-7, over the 7 days that segment owns.
    - **the validation origin as the starting point**, not the test one: this is a
      diagnostic, and it has no business reading the final benchmark's window.
    - **no overlap**: the origins are 7 days apart and only the first 7 days are scored,
      so no day is scored twice and the lag-1 autocovariance between origins is not an
      artefact of overlap.

    The forecast still runs to the standard 28-day horizon -- identical to the rest of
    the study, and the cost is the same -- and only the first 7 days are scored, by
    `persistence.cell_wql`.
    """
    from m5_chronos2.persistence import N_ORIGINS, SEGMENT_DAYS, origin_grid, sweep_context

    context = int(study.choice["early_context"])
    origins = origin_grid(study.validation_origin, N_ORIGINS, SEGMENT_DAYS)
    print(f"  {len(origins)} origins, {SEGMENT_DAYS}-day steps, context {context}")
    print(f"  from {origins[0]} to {origins[-1]}")

    paths = []
    for origin in origins:
        path = experiment_path(f"origin_sweep_ctx{context}_origin{origin}", study.name)
        cached(path, lambda origin=origin: forecast(
            sweep_context(study.sales, origin), origin,
            context_length=context, device=study.device))
        paths.append(path)
    return paths


def make_stage_finetune_test(contexts: list[int] | None):
    def stage_finetune_test(study: Study) -> list[Path]:
        """Everything section 4 of notebook 08 needs at the TEST origin.

        For each length passed to --finetune-context it produces:
          - the LoRA checkpoint re-trained at the test origin;
          - its forecasts at the selected predict lengths;
        and, in every case, the zero-shot forecasts at the test origin at those same
        lengths -- the notebook needs them if a segment's winner is the zero-shot model.

        The lengths must be passed explicitly. That is deliberate: deriving them here
        would amount to selecting on the test window, which notebook 03's protocol
        forbids. Passing both lengths is legitimate -- computing is not choosing, and
        notebook 08 only scores the winner designated on validation -- but the gesture
        stays explicit and on the record.
        """
        if not contexts:
            raise SystemExit(
                "\nfinetune_test requires --finetune-context.\n"
                "  --finetune-context 1829        the single length selected on validation\n"
                "  --finetune-context 1099,1829   both, so notebook 08 runs end to end\n"
                "                                 without computing anything locally\n"
                "Deriving it here would amount to choosing on the test window."
            )
        origin = study.test_origin
        choice = study.choice
        predict_contexts = sorted({choice["early_context"], choice["late_context"]})
        paths = []

        # Zero-shot at the test origin. Same artifact name notebook 08 uses.
        for length in predict_contexts:
            path = experiment_path(f"context_{length}_origin{origin}", study.name)
            print(f"  zero-shot test, context {length}")
            cached(path, lambda length=length: forecast(
                history(study.sales, origin), origin,
                context_length=int(length), device=study.device))
            paths.append(path)

        inputs = finetune_inputs(study.sales, origin, study.series_ids)
        for finetune_context in contexts:
            checkpoint = _finetune_at(study, origin, finetune_context, inputs)
            paths.append(checkpoint)
            for length in predict_contexts:
                name = (f"lora_ctx{finetune_context}_steps{NUM_STEPS}_lr{LEARNING_RATE:g}"
                        f"_bs{BATCH_SIZE}_context_{length}_origin{origin}")
                path = experiment_path(name, study.name)
                print(f"  LoRA@{finetune_context} -> test forecast at context {length}")
                cached(path, lambda length=length, checkpoint=checkpoint: forecast(
                    history(study.sales, origin), origin, context_length=int(length),
                    model_path=checkpoint, device=study.device))
                paths.append(path)
        return paths
    return stage_finetune_test


def build_registry(finetune_contexts: list[int] | None) -> dict[str, Stage]:
    return {s.name: s for s in [
        Stage("vanilla", "zero-shot Chronos-2, both origins", stage_vanilla,
              cost="heavy", note="~2 min GPU / ~45 min CPU"),
        Stage("uplift", "event-uplift table (reads all 30,490 series)", stage_uplift,
              cost="light", note="CPU, no model"),
        Stage("covariates", "4 blocks of known covariates", stage_covariates,
              needs_gpu=True, cost="heavy", requires=["uplift"], note="~4 x 1 min GPU"),
        Stage("cross_learning", "cross-learning on, series shuffled", stage_cross_learning,
              needs_gpu=True, cost="heavy"),
        Stage("blank", "3 blank-period encodings", stage_blank,
              needs_gpu=True, cost="heavy"),
        Stage("context_coarse", "coarse grid of context lengths", stage_context_coarse,
              needs_gpu=True, cost="heavy", note="~21 lengths, ~8 min GPU"),
        Stage("context_fine", "fine grid around the winners", stage_context_fine,
              needs_gpu=True, cost="heavy", requires=["context_coarse"],
              note="~30 lengths, ~12 min GPU"),
        Stage("finetune", "LoRA at both contexts, validation origin", stage_finetune,
              needs_gpu=True, cost="heavy",
              note="~32 min / context GPU; requires context_choice.json"),
        Stage("finetune_predict", "forecasts from the LoRA checkpoints", stage_finetune_predict,
              needs_gpu=True, cost="heavy", requires=["finetune"]),
        Stage("finetune_test", "test origin: zero-shot + re-fine-tune", make_stage_finetune_test(finetune_contexts),
              needs_gpu=True, cost="heavy", note="requires --finetune-context"),
        Stage("origin_sweep", "24 zero-shot origins, 7-day steps", stage_origin_sweep,
              needs_gpu=True, cost="heavy",
              note="~24 x 4 s GPU; requires context_choice.json; for notebook 09"),
    ]}


# Execution order for --all: it stops before finetune, which waits on a human decision.
DEFAULT_ORDER = ["vanilla", "uplift", "covariates", "cross_learning", "blank",
                 "context_coarse", "context_fine"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", help="comma-separated stage names")
    parser.add_argument("--all", action="store_true",
                        help=f"everything up to the decision point: {', '.join(DEFAULT_ORDER)}")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="recompute even if the artifacts exist")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-slow", action="store_true",
                        help="allow a needs_gpu stage to run on CPU")
    parser.add_argument("--finetune-context", default=None,
                        help="length(s) selected on validation, for finetune_test; "
                             "one or more, comma-separated")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    finetune_contexts = ([int(v) for v in args.finetune_context.split(",") if v.strip()]
                         if args.finetune_context else None)
    registry = build_registry(finetune_contexts)

    if args.list:
        print(f"{'stage':20s} {'gpu':4s} {'cost':7s} {'depends on':18s} note")
        for stage in registry.values():
            print(f"{stage.name:20s} {'yes' if stage.needs_gpu else '-':4s} "
                  f"{stage.cost:7s} {','.join(stage.requires) or '-':18s} {stage.note}")
        print(f"\n--all = {' -> '.join(DEFAULT_ORDER)}")
        print("then: notebook 07 on the Mac -> context_choice.json -> finetune")
        return

    if args.all:
        names = DEFAULT_ORDER
    elif args.stages:
        names = [n.strip() for n in args.stages.split(",") if n.strip()]
    else:
        parser.error("pass --stages, --all or --list")

    unknown = [n for n in names if n not in registry]
    if unknown:
        parser.error(f"unknown stage(s): {unknown}. See --list")

    device = resolve_device(args.device)
    report = device_report(device)
    print("=" * 70)
    print(json.dumps(report, indent=2, default=str))

    blocked = [n for n in names if registry[n].needs_gpu and not device.startswith("cuda")]
    if blocked and not args.allow_slow:
        raise SystemExit(
            f"\nstage(s) marked needs_gpu launched on {device}: {blocked}\n"
            "The measured factor is about 24x, so this is probably not a choice.\n"
            "Run it on the tower, or add --allow-slow if it is deliberate."
        )

    study = Study(device)
    print(f"roster {study.name}, device {device}")

    if args.dry_run:
        for name in names:
            print(f"  [dry-run] {name}: {registry[name].description}")
        return

    # The dtype is checked once, before committing hours of computation.
    dtype = pipeline_dtype(get_pipeline(device=device))
    print(f"model dtype: {dtype}")
    if "float32" not in dtype:
        raise SystemExit(
            f"dtype is {dtype}, not float32. On Pascal (sm_61) half precision is a\n"
            "~64x slowdown, not a speed-up. Stopping here."
        )

    total = time.time()
    for name in names:
        stage = registry[name]
        print(f"\n{'-' * 70}\n{name}: {stage.description}")
        start = time.time()
        try:
            produced = stage.run(study)
        except KeyboardInterrupt:
            print(f"\ninterrupted during {name}; the artifacts already written are kept")
            sys.exit(130)
        seconds = time.time() - start
        for path in produced:
            if path.exists() and path.is_file():
                write_meta(path, name, device, seconds, {"dtype": dtype})
        print(f"{name} finished in {seconds / 60:.1f} min, {len(produced)} artifact(s)")

    print(f"\n{'=' * 70}\ntotal: {(time.time() - total) / 60:.1f} min")
    if set(names) & set(DEFAULT_ORDER) and not study.choice_path.exists():
        print("\nNext: pull the artifacts, run notebook 07 on the Mac, choose the context\n"
              "lengths, then launch the finetune stage.")


if __name__ == "__main__":
    main()
