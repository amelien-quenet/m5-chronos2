"""Chronos-2: forecasts in the project's standard format, covariates, fine-tuning.

Never import xgboost in a process that imports torch (this module does, lazily):
on macOS both ship their own OpenMP runtime and the kernel crashes. The XGBoost
baseline is only ever read back from its saved predictions (notebook 04).

Conventions, shared with the rest of the project
------------------------------------------------
- A *context* is one row per (series, day) up to the origin: 'id', 'date', the
  target column(s) and optional covariates.
- Days without a price are missing data (notebook 02). They only occur before an
  item's first price, so they are dropped: each series simply starts later and
  stays gap-free, which Chronos-2 requires.
- Forecasts use the standard format of metrics.py:
  id, origin, date, y_pred, q0.1 ... q0.9   (y_pred is the median, q0.5).

Hardware
--------
Two machines run this code: an Intel MacBook (CPU, torch 2.2.2) and a tower with
a GTX 1080 reached over SSH (CUDA, torch 2.6.0). `resolve_device` is the only
place where that choice is made, and `device_report` is what gets recorded next
to every artifact -- CPU and GPU do not return bit-identical numbers, so a
comparison table must come from a single device.
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from m5_chronos2.blank_periods import add_blank_flags, add_silent_runs
from m5_chronos2.data import ARTIFACTS_DIR, HORIZON, ID_COLUMNS
from m5_chronos2.metrics import QUANTILE_LEVELS, quantile_column

MODEL_PATH = "amazon/chronos-2"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"   # every Chronos-2 run, one parquet per config
FINETUNED_DIR = ARTIFACTS_DIR / "finetuned"       # LoRA checkpoints

QUANTILE_COLUMNS = [quantile_column(level) for level in QUANTILE_LEVELS]
FORECAST_COLUMNS = ["id", "origin", "date", "y_pred"] + QUANTILE_COLUMNS

# Same blank-period definition as notebook 02.
MIN_BLANK_DAYS = 14

_PIPELINES: dict = {}


# --- Which machine are we on -------------------------------------------------------

def resolve_device(device: str | None = None) -> str:
    """'cuda' if a GPU is present, 'cpu' otherwise. M5_DEVICE overrides both.

    The single place where the hardware is chosen: no notebook and no stage script
    hardcodes it, so the same code runs on the laptop and on the tower, and the
    choice can be recorded with the artifact it produced.
    """
    if device is not None:
        return device
    forced = os.environ.get("M5_DEVICE")
    if forced:
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def device_report(device: str | None = None) -> dict:
    """What to print before a long run, and to store beside every artifact."""
    import torch

    device = resolve_device(device)
    report = {"device": device, "torch": torch.__version__}
    if device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(0)
        report |= {
            "gpu": torch.cuda.get_device_name(0),
            "capability": "%d.%d" % torch.cuda.get_device_capability(0),
            "vram_gb": round(properties.total_memory / 1024**3, 1),
            "arch_list": torch.cuda.get_arch_list(),
        }
    return report


# --- The model -----------------------------------------------------------------

def get_pipeline(model_path: str | Path = MODEL_PATH, device: str | None = None):
    """Load a Chronos-2 pipeline once and keep it (loading is the slow part).

    `model_path` is a Hugging Face name or a local fine-tuned checkpoint.

    The dtype is forced to float32, on GPU as well as on CPU. This is deliberate
    and counter-intuitive: half precision is the usual default on CUDA, but the
    GTX 1080 is Pascal (sm_61), which has no bfloat16 hardware at all and runs
    float16 at 1/64 of its float32 rate. Letting the library pick half precision
    there would make the GPU slower than the laptop CPU. Memory is the price;
    `finetune` pays it with gradient accumulation rather than with dtype.

    Call `pipeline_dtype` afterwards: it reads the dtype actually loaded, which is
    the only trustworthy confirmation that this took effect.
    """
    import torch
    from chronos import Chronos2Pipeline

    device = resolve_device(device)
    key = (str(model_path), device)
    if key not in _PIPELINES:
        # transformers renamed `torch_dtype` to `dtype` and warns on the old name;
        # try the current one first and fall back for older versions.
        pipeline = None
        for keyword in ("dtype", "torch_dtype"):
            try:
                pipeline = Chronos2Pipeline.from_pretrained(
                    str(model_path), device_map=device, **{keyword: torch.float32}
                )
                break
            except TypeError:
                continue
        if pipeline is None:
            print("  WARNING: from_pretrained accepts no dtype keyword -- check pipeline_dtype()")
            pipeline = Chronos2Pipeline.from_pretrained(str(model_path), device_map=device)
        _PIPELINES[key] = pipeline
    return _PIPELINES[key]


def pipeline_dtype(pipeline) -> str:
    """Dtype of the loaded weights. Anything other than float32 is a problem on Pascal."""
    import torch

    for attribute in vars(pipeline).values():
        if isinstance(attribute, torch.nn.Module):
            return str(next(attribute.parameters()).dtype)
    return "unknown"


def pipeline_limits(pipeline) -> dict:
    """The numbers to print rather than trust from memory."""
    return {
        "default_context_length": pipeline.default_context_length,
        "model_context_length": pipeline.model_context_length,
        "model_prediction_length": pipeline.model_prediction_length,
        "output_patch_size": pipeline.model_output_patch_size,
        "dtype": pipeline_dtype(pipeline),
    }


# --- What the model sees -----------------------------------------------------------

def history(sales: pl.DataFrame, origin: date, ids: list[str] | None = None) -> pl.DataFrame:
    """Observed history of each series up to the origin (included), missing days dropped.

    Fails if a series has a gap: Chronos-2 assumes a regular daily grid.
    """
    table = sales.filter((pl.col("date") <= origin) & ~pl.col("is_missing"))
    if ids is not None:
        table = table.filter(pl.col("id").is_in(list(ids)))
    table = table.sort("id", "date")

    span = table.group_by("id").agg(
        pl.len().alias("n_days"),
        ((pl.col("date").max() - pl.col("date").min()).dt.total_days() + 1).alias("span_days"),
    )
    assert (span["n_days"] == span["span_days"]).all(), "a series has a gap in its history"
    return table


def past_blank_flags(history_table: pl.DataFrame, min_days: int = MIN_BLANK_DAYS) -> pl.DataFrame:
    """Blank-period flag 'is_blank' recomputed from the history only.

    The flags saved by notebook 02 use the whole data: a silent run that starts before
    the origin and continues after it is measured with days the model must not see.
    Recomputing on the truncated history removes that leak.
    """
    table = history_table.drop("is_blank", "blank_kind", "silent_run_length", strict=False)
    table = table.with_columns((~pl.col("is_missing")).alias("is_on_sale"))
    table = add_silent_runs(table)
    table = add_blank_flags(table, min_days=min_days, long_days=HORIZON)
    return table.drop("is_on_sale", "is_silent", "silent_run_length", "blank_kind")


def _to_pandas(table: pl.DataFrame):
    """The one polars -> pandas hop, at the model boundary. Dates become timestamps."""
    return table.with_columns(pl.col("date").cast(pl.Datetime("ns"))).to_pandas()


def forecast(
    context: pl.DataFrame,
    origin: date,
    target: str | list[str] = "sales",
    past_covariates: list[str] | tuple = (),
    known_covariates: list[str] | tuple = (),
    future: pl.DataFrame | None = None,
    context_length: int | None = None,
    cross_learning: bool = False,
    batch_size: int = 100,
    model_path: str | Path = MODEL_PATH,
    device: str | None = None,
) -> pl.DataFrame:
    """Forecast the HORIZON days after `origin` for every series of `context`.

    Only the columns named here reach the model: in `predict_df`, any extra column
    would silently become a past covariate.
    - past_covariates: columns known up to the origin only;
    - known_covariates: columns also known over the forecast window, given in `future`
      ('id', 'date' and those columns, HORIZON rows per series);
    - context_length: the model reads the last `context_length` days and ignores the
      rest. None = the model's maximum;
    - cross_learning: False forecasts every series on its own (batching is then only
      a matter of speed); True lets the series of a batch attend to each other;
    - device: None resolves it (see resolve_device). On the GPU, `batch_size` is
      bounded by VRAM rather than by patience: lower it if inference runs out.

    Returns the standard forecast format. With several targets, one block per target,
    told apart by a 'target_name' column. Every quantile is clipped at 0 (sales).
    """
    targets = [target] if isinstance(target, str) else list(target)
    columns = ["id", "date"] + targets + list(past_covariates) + list(known_covariates)
    context = context.select(columns).with_columns(pl.col(targets).cast(pl.Float64))

    future_pd = None
    if known_covariates:
        assert future is not None, "known covariates need their future values"
        future_pd = _to_pandas(future.select(["id", "date"] + list(known_covariates)))

    pipeline = get_pipeline(model_path, device)
    predictions = pipeline.predict_df(
        _to_pandas(context),
        future_df=future_pd,
        id_column="id",
        timestamp_column="date",
        target=target,
        prediction_length=HORIZON,
        quantile_levels=QUANTILE_LEVELS,
        batch_size=batch_size,
        context_length=context_length,
        cross_learning=cross_learning,
        freq="D",
    )

    predictions = pl.from_pandas(predictions).rename(
        {"predictions": "y_pred"} | {str(level): quantile_column(level) for level in QUANTILE_LEVELS}
    )
    predictions = predictions.with_columns(
        pl.col("date").cast(pl.Date),
        pl.lit(origin).alias("origin"),
        pl.col(["y_pred"] + QUANTILE_COLUMNS).clip(lower_bound=0.0),
    )
    if len(targets) > 1:
        return predictions.select(["target_name"] + FORECAST_COLUMNS).sort("target_name", "id", "date")
    return predictions.select(FORECAST_COLUMNS).sort("id", "date")


def cached(path: Path, compute: Callable[[], pl.DataFrame], force: bool = False) -> pl.DataFrame:
    """Read `path` if it exists, otherwise compute, save and return.

    Every Chronos-2 run is saved under a name that describes its configuration, so a
    notebook can be re-run end to end without recomputing, and two notebooks never
    compare a result with a silently different one.
    """
    path = Path(path)
    if path.exists() and not force:
        return pl.read_parquet(path)
    start = time.time()
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(path)
    print(f"  computed {path.name} in {time.time() - start:.0f} s")
    return result


def experiment_path(name: str, roster: str) -> Path:
    return EXPERIMENTS_DIR / f"{name}_{roster}.parquet"


def stitch(early: pl.DataFrame, late: pl.DataFrame, last_early_day: int) -> pl.DataFrame:
    """Days 1..last_early_day from `early`, the following days from `late`.

    The two parts are disjoint by construction: nothing is blended. The result is one
    forecast, quantiles included, as far as the metrics are concerned.
    """
    days_ahead = (pl.col("date") - pl.col("origin")).dt.total_days()
    parts = [
        early.select(FORECAST_COLUMNS).filter(days_ahead <= last_early_day),
        late.select(FORECAST_COLUMNS).filter(days_ahead > last_early_day),
    ]
    return pl.concat(parts).sort("origin", "id", "date")


# --- Known covariates (the exogenous part of the XGBoost feature set) ------------------

KNOWN_COVARIATE_BLOCKS = {
    "calendar": ["day_of_week", "month"],
    "price": ["sell_price", "sell_price_lag_7", "sell_price_lag_14"],
    "snap_events": ["snap", "event_uplift"],
}
ALL_KNOWN_COVARIATES = [name for block in KNOWN_COVARIATE_BLOCKS.values() for name in block]


def known_covariate_table(sales: pl.DataFrame, uplift: pl.DataFrame) -> pl.DataFrame:
    """One row per (series, day), every date: the known covariates of notebook 04.

    Same definitions as the XGBoost baseline (gbdt_features.series_features):
    day of week, month, SNAP day of the item's state, price and its 7/14-day momentum.
    `uplift` is gbdt_features.event_uplift(...) for the origin ('day', 'event_uplift').
    """
    from m5_chronos2.gbdt_features import calendar_by_day, series_features

    days = calendar_by_day().select("date", "day")
    table = sales.select(ID_COLUMNS + ["date", "sell_price"]).join(days, on="date", how="left")
    table = series_features(table).join(uplift, on="day", how="left")
    return table.select(
        ["id", "date"] + [pl.col(name).cast(pl.Float64) for name in ALL_KNOWN_COVARIATES]
    ).sort("id", "date")


def future_window(table: pl.DataFrame, origin: date, ids: list[str]) -> pl.DataFrame:
    """Rows of `table` inside the forecast window of `origin`, for the given series."""
    return table.filter(
        pl.col("id").is_in(list(ids)),
        pl.col("date") > origin,
        pl.col("date") <= origin + timedelta(days=HORIZON),
    ).sort("id", "date")


# --- Fine-tuning ---------------------------------------------------------------------

def _target_arrays(table: pl.DataFrame) -> list[np.ndarray]:
    """One 1-D float array of sales per series, in id order."""
    return [
        part["sales"].to_numpy().astype(np.float32)
        for _, part in table.sort("id", "date").group_by("id", maintain_order=True)
    ]


def finetune_inputs(sales: pl.DataFrame, origin: date, ids: list[str]) -> tuple[list, list]:
    """Training and validation inputs for a fine-tune that never sees the forecast window.

    - training: history up to origin - HORIZON; Chronos-2 cuts random windows in it;
    - validation: history up to the origin; Chronos-2 scores the last HORIZON days of
      each series (the 28 days just before the origin) and keeps the best checkpoint.
    Same idea as the early stopping of the XGBoost baseline (notebook 04).
    """
    full = history(sales, origin, ids)
    train = full.filter(pl.col("date") <= origin - timedelta(days=HORIZON))
    return _target_arrays(train), _target_arrays(full)


def checkpoint_dir(roster: str, origin: date, context_length: int, num_steps: int,
                   learning_rate: float, batch_size: int) -> Path:
    """Where a LoRA run is saved; the name holds its whole configuration."""
    name = f"origin{origin}_ctx{context_length}_steps{num_steps}_lr{learning_rate:g}_bs{batch_size}"
    return FINETUNED_DIR / roster / name


# VRAM of the tower's GTX 1080, in GiB: below this, assume gradient accumulation is
# needed rather than discovering it through an out-of-memory error at step 3.
SMALL_VRAM_GB = 12.0
DEFAULT_MICRO_BATCH = 4


def suggest_micro_batch(batch_size: int, device: str | None = None) -> int | None:
    """Micro-batch to use, or None when the whole batch fits.

    On CPU memory is the machine's RAM and the full batch is fine. On a small GPU it
    is not: 8 GiB cannot hold 32 series at a context of several thousand days in
    float32. Returns the largest divisor of `batch_size` that is <= DEFAULT_MICRO_BATCH.
    """
    device = resolve_device(device)
    if not device.startswith("cuda"):
        return None
    import torch

    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram >= SMALL_VRAM_GB:
        return None
    for micro in range(min(DEFAULT_MICRO_BATCH, batch_size), 0, -1):
        if batch_size % micro == 0:
            return micro
    return 1


def finetune(
    train_inputs: list,
    validation_inputs: list | None,
    context_length: int,
    output_dir: Path,
    num_steps: int = 1000,
    learning_rate: float = 1e-5,
    batch_size: int = 32,
    micro_batch_size: int | None = None,
    device: str | None = None,
) -> Path:
    """LoRA fine-tune of Chronos-2. Returns the checkpoint path (usable as a model_path).

    `batch_size` is always the *effective* batch size, the one that defines the update
    and therefore the result. `micro_batch_size` only changes how that update is
    computed: the trainer processes that many series at a time and applies the update
    after `batch_size // micro_batch_size` of them. The gradient is the same, the peak
    memory is a fraction of it. Left to None it is chosen by `suggest_micro_batch`,
    which splits on a small GPU and leaves the CPU alone.

    Three traps handled here:
    - without the `peft` package, Chronos-2 silently falls back to FULL fine-tuning:
      we fail instead;
    - Chronos-2 asks the trainer for the fused AdamW optimizer, which older torch
      versions (2.2 on Intel Macs) only support on CUDA: on CPU we ask for the plain one;
    - no fp16/bf16 flag is passed on purpose. The trainer defaults to full precision,
      and the model was loaded in float32 by `get_pipeline` -- on Pascal, half
      precision is a slowdown, not a speedup (see that function).
    """
    import importlib.util

    if importlib.util.find_spec("peft") is None:
        raise ImportError("LoRA needs the `peft` package (uv add peft)")

    checkpoint = Path(output_dir) / "finetuned-ckpt"
    if checkpoint.exists():
        print(f"  checkpoint found, not retrained: {checkpoint.parent.name}")
        return checkpoint

    device = resolve_device(device)
    micro = micro_batch_size if micro_batch_size is not None else suggest_micro_batch(batch_size, device)
    micro = micro or batch_size
    assert batch_size % micro == 0, f"batch_size {batch_size} is not a multiple of {micro}"
    accumulation = batch_size // micro

    pipeline = get_pipeline(MODEL_PATH, device)   # fit() trains a copy; this one stays vanilla
    print(f"  fine-tune on {device}, dtype {pipeline_dtype(pipeline)}, "
          f"batch {batch_size} = {micro} x {accumulation} accumulated")
    pipeline.fit(
        inputs=train_inputs,
        validation_inputs=validation_inputs,
        prediction_length=HORIZON,
        finetune_mode="lora",
        context_length=context_length,
        learning_rate=learning_rate,
        num_steps=num_steps,
        batch_size=micro,
        gradient_accumulation_steps=accumulation,
        output_dir=output_dir,
        optim="adamw_torch_fused" if device.startswith("cuda") else "adamw_torch",
    )
    return checkpoint
