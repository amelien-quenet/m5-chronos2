"""Shared setup for every post: one import path, one artifacts root, one figure style.

The posts compute nothing. They read the parquets and JSONs that the stages wrote
and plot them with the study's own figure layer, exactly as notebooks 01-10 do.

Posts execute from the blog root (`execute-dir: project` in _quarto.yml), so the
relative path below resolves against BLOG/blog.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

STUDY = (Path.cwd() / "..").resolve()
ARTIFACTS = STUDY / "artifacts"
ROSTER = "n500_seed42"

if not ARTIFACTS.is_dir():
    raise RuntimeError(
        f"{ARTIFACTS} not found. The posts read the study's artifacts directly, so "
        "they must be present before rendering: pull them from the tower with "
        "`scripts/remote.sh pull`. Failing here is deliberate - a render against "
        "missing artifacts would otherwise publish empty or stale figures."
    )

_src = str(STUDY / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from m5_chronos2 import metrics, plots  # noqa: E402
from m5_chronos2.evaluation import actuals, make_windows  # noqa: E402

plots.use_blog_style()


def artifact(name: str, ext: str = "parquet") -> Path:
    """Roster-stamped artifact path.

    artifact("predictions/chronos2_best")
        -> .../artifacts/predictions/chronos2_best_n500_seed42.parquet
    artifact("chronos2_best_config", ext="json")
        -> .../artifacts/chronos2_best_config_n500_seed42.json
    """
    return ARTIFACTS / f"{name}_{ROSTER}.{ext}"


def read_json(name: str) -> dict:
    """A roster-stamped JSON artifact."""
    return json.loads(artifact(name, ext="json").read_text())


# Loaded once per render. Everything a post needs to score a model lives here, so
# a post never re-derives the protocol - it inherits it.
SALES = pl.read_parquet(artifact("sales_flags"))
TRUTH = actuals(SALES)
EVAL_ROSTER = pl.read_parquet(artifact("eval_roster"))
WINDOWS = make_windows()
VALIDATION_ORIGIN, TEST_ORIGIN = WINDOWS["origin"].to_list()


def scored(model: str, role: str | None = None) -> pl.DataFrame:
    """A saved forecast, joined to the truth on the frozen evaluation roster.

    `role` is "validation" or "test"; omitted, both windows come back with a
    `role` column. This is the only way the posts read a model, so no post can
    score on a different roster than the study did.
    """
    table = metrics.join_truth(
        pl.read_parquet(artifact(f"predictions/{model}")), TRUTH, EVAL_ROSTER
    ).join(WINDOWS.select("origin", "role"), on="origin")
    return table.filter(pl.col("role") == role) if role else table


def table_of(models: list[str], role: str, by: str | None = None) -> pl.DataFrame:
    """The metric table for several models on one window."""
    return metrics.compare({model: scored(model, role) for model in models}, by=by)


NICE_NAMES = {
    "zero": "always zero",
    "seasonal_naive": "seasonal naive",
    "moving_average_28d": "28-day mean",
    "xgboost_btrotta": "XGBoost",
    "chronos2_vanilla": "Chronos-2, zero-shot",
    "chronos2_best": "Chronos-2, final",
}


def rename_models(table: pl.DataFrame) -> pl.DataFrame:
    """Readable model names in a published table."""
    return table.with_columns(pl.col("model").replace(NICE_NAMES))


def md(table: pl.DataFrame, digits: int = 2):
    """Render a polars table as Markdown, so a published table is a real table.

    Written by hand rather than through pandas: the blog environment has neither
    pandas nor tabulate, and does not need them to print a dozen rows.
    """
    from IPython.display import Markdown

    def cell(value) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:,.{digits}f}"
        return str(value)

    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.iter_rows()]
    return Markdown("\n".join([header, rule, *rows]))


__all__ = [
    "STUDY", "ARTIFACTS", "ROSTER", "artifact", "read_json",
    "SALES", "TRUTH", "EVAL_ROSTER", "WINDOWS", "VALIDATION_ORIGIN", "TEST_ORIGIN",
    "scored", "table_of", "rename_models", "NICE_NAMES", "md",
    "metrics", "plots", "pl",
]
