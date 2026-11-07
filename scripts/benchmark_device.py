#!/usr/bin/env python
"""Cost of the study's two heavy operations, measured on the current machine.

Why this script exists: a computation is only worth moving to the tower if you know how
much faster the tower is. "A GPU, therefore faster" does not hold -- on a GTX 1080
(Pascal), half precision left switched on makes the GPU *slower* than the laptop's CPU.
This script produces the number.

It measures two things, which do not behave alike:

  inference  -- what the context sweep (notebook 07) does, dozens of times over;
  fine-tune  -- what notebook 08 does, and what dominates the total cost.

Run exactly the same command on both machines; only the device changes. Both write
their result to artifacts/benchmark/<device>.json, so the comparison can be read
afterwards without having to remember the numbers.

    # on the Mac
    uv run python scripts/benchmark_device.py
    # on the tower
    ssh gpu 'cd m5-chronos2 && uv run python scripts/benchmark_device.py'

On a GPU out-of-memory error, lower --micro-batch (8 GB of VRAM does not hold 32 series
at a context of 1829 days in float32).
"""

import argparse
import json
import shutil
import time
from datetime import date
from pathlib import Path

import polars as pl

from m5_chronos2.chronos2 import (
    device_report,
    finetune,
    finetune_inputs,
    forecast,
    get_pipeline,
    history,
    pipeline_limits,
    resolve_device,
    suggest_micro_batch,
)
from m5_chronos2.data import ARTIFACTS_DIR

# The protocol's validation origin (notebook 03). Never the test window: a benchmark has
# no business touching the data the final number comes from.
VALIDATION_ORIGIN = date(2016, 3, 27)

BENCHMARK_DIR = ARTIFACTS_DIR / "benchmark"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default="n500_seed42")
    parser.add_argument("--series", type=int, default=100,
                        help="number of series; 100 is enough for a timing ratio")
    parser.add_argument("--steps", type=int, default=50,
                        help="fine-tune steps measured, extrapolated to 1000")
    parser.add_argument("--context", type=int, default=1829,
                        help="the context notebook 07 selected for days 8-28")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="the fine-tune's EFFECTIVE batch, the one that defines the update")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="series processed at once; None = chosen from the VRAM")
    parser.add_argument("--infer-batch", type=int, default=100,
                        help="inference batch (speed only, no effect on the result)")
    parser.add_argument("--device", default=None, help="cpu, cuda; None = auto")
    parser.add_argument("--skip-finetune", action="store_true")
    return parser.parse_args()


def load_series_ids(roster: str, limit: int) -> list[str]:
    """The series scored at the validation origin, truncated to `limit`, in order.

    The order is fixed (sorted by id) so that both machines measure exactly the same
    work: a different subset would give a false timing ratio.
    """
    table = pl.read_parquet(ARTIFACTS_DIR / f"eval_roster_{roster}.parquet")
    scored = table.filter(pl.col("origin") == VALIDATION_ORIGIN)
    if scored.height == 0:
        raise SystemExit(
            f"no series at origin {VALIDATION_ORIGIN} in eval_roster_{roster}; "
            f"origins present: {sorted(table['origin'].unique().to_list())}"
        )
    return scored.select("id").unique().sort("id")["id"].to_list()[:limit]


def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = device_report(device)

    print("=== machine ===")
    print(json.dumps(report, indent=2, default=str))

    sales = pl.read_parquet(ARTIFACTS_DIR / f"sales_flags_{args.roster}.parquet")
    ids = load_series_ids(args.roster, args.series)

    pipeline = get_pipeline(device=device)
    limits = pipeline_limits(pipeline)
    print("\n=== model ===")
    print(json.dumps(limits, indent=2, default=str))
    if "float32" not in limits["dtype"]:
        print(f"\n  !! dtype is {limits['dtype']}, not float32: on Pascal that is a")
        print("     slowdown, not a speed-up. The number below does not measure")
        print("     what it is meant to measure.")

    context = history(sales, VALIDATION_ORIGIN, ids)
    days = context.height // max(len(ids), 1)
    print(f"\n{len(ids)} series, origin {VALIDATION_ORIGIN}, "
          f"~{days} days of history, context {args.context}")

    results = {
        "device": device,
        "report": report,
        "dtype": limits["dtype"],
        "series": len(ids),
        "context_length": args.context,
    }

    # --- 1. Inference: the context sweep's operation -------------------------------
    print("\n=== inference ===")
    start = time.time()
    forecast(
        context,
        VALIDATION_ORIGIN,
        context_length=args.context,
        batch_size=args.infer_batch,
        device=device,
    )
    elapsed = time.time() - start
    per_series = elapsed / max(len(ids), 1)
    results["inference_seconds"] = round(elapsed, 2)
    results["inference_seconds_per_series"] = round(per_series, 4)
    print(f"{len(ids)} series in {elapsed:.1f} s  ({per_series * 1000:.0f} ms/series)")
    # Notebook 07's sweep makes one pass per configuration over the ~479 scored series.
    print(f"  -> 1 configuration over 479 series : {per_series * 479 / 60:.1f} min")
    print(f"  -> a sweep of 50 configurations    : {per_series * 479 * 50 / 3600:.1f} h")

    # --- 2. Fine-tune: the operation that dominates the cost ------------------------
    if not args.skip_finetune:
        micro = args.micro_batch or suggest_micro_batch(args.batch_size, device) or args.batch_size
        print(f"\n=== LoRA fine-tune, {args.steps} steps, "
              f"batch {args.batch_size} = {micro} x {args.batch_size // micro} ===")

        train_inputs, validation_inputs = finetune_inputs(sales, VALIDATION_ORIGIN, ids)
        output_dir = BENCHMARK_DIR / f"{device.replace(':', '')}_steps{args.steps}"
        shutil.rmtree(output_dir, ignore_errors=True)   # never resume: this is a measurement

        start = time.time()
        try:
            finetune(
                train_inputs,
                validation_inputs,
                context_length=args.context,
                output_dir=output_dir,
                num_steps=args.steps,
                batch_size=args.batch_size,
                micro_batch_size=micro,
                device=device,
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            raise SystemExit(
                f"\nnot enough GPU memory with --micro-batch {micro}.\n"
                f"Run again with half of it: --micro-batch {max(micro // 2, 1)}\n"
                f"(the effective batch stays {args.batch_size}, the result is unchanged)"
            ) from error
        elapsed = time.time() - start

        per_step = elapsed / args.steps
        results |= {
            "finetune_steps": args.steps,
            "finetune_seconds": round(elapsed, 2),
            "finetune_seconds_per_step": round(per_step, 3),
            "micro_batch": micro,
        }
        print(f"{args.steps} steps in {elapsed:.1f} s  ({per_step:.2f} s/step)")
        print(f"  -> 1000 steps (notebook 08's configuration) : {per_step * 1000 / 60:.0f} min")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / f"{device.replace(':', '')}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwritten to {path.relative_to(ARTIFACTS_DIR.parent)}")

    # --- 3. Comparison, if the other machine has already run ------------------------
    others = sorted(p for p in BENCHMARK_DIR.glob("*.json") if p != path)
    for other_path in others:
        other = json.loads(other_path.read_text())
        if other.get("series") != results["series"] or \
           other.get("context_length") != results["context_length"]:
            print(f"\n{other_path.name}: different configuration, not comparable")
            continue
        print(f"\n=== {results['device']} against {other['device']} ===")
        for key, label in [("inference_seconds", "inference"),
                           ("finetune_seconds_per_step", "fine-tune (s/step)")]:
            mine, theirs = results.get(key), other.get(key)
            if mine and theirs:
                print(f"{label:22s} {mine:8.2f}  vs {theirs:8.2f}   "
                      f"factor {theirs / mine:.1f}x")


if __name__ == "__main__":
    main()
