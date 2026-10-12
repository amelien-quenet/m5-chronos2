#!/usr/bin/env python
"""Runner par etapes : le calcul lourd de l'etude, hors des notebooks.

Principe
--------
Une *etape* est une fonction pure `configuration -> artifacts/<nom>.parquet`. Elle
est autonome (tout est dans la ligne de commande, rien dans l'etat d'un kernel),
elle ecrit son resultat sur disque, et elle est reprenable : relancee, elle saute
ce qui existe deja. C'est le contrat que `chronos2.cached()` implemente deja -- ce
script ne fait que l'exposer en ligne de commande pour qu'un calcul de six heures
puisse etre lance puis oublie.

Les etapes appellent exactement les memes fonctions que les notebooks, vers
exactement les memes chemins. Donc apres un run sur la tour et un rsync des
artifacts, les notebooks se re-executent de bout en bout en quelques secondes :
chaque `cached()` trouve son parquet. Les notebooks restent la couche narrative
-- scores, figures, redaction -- et ne calculent plus rien.

Usage
-----
    python scripts/run_stage.py --list
    python scripts/run_stage.py --stages vanilla,covariates
    python scripts/run_stage.py --all
    python scripts/run_stage.py --stages context_coarse --force
    python scripts/run_stage.py --stages finetune_test --finetune-context 1829

Routage
-------
Une etape marquee `needs_gpu` refuse de tourner sur CPU : le facteur mesure est
de ~24x, donc la lancer sur le portable n'est pas un choix mais un oubli.
`--allow-slow` leve le refus quand c'est volontaire.

Tracabilite
-----------
Chaque artifact produit recoit un `<nom>.parquet.meta.json` : machine, device,
versions des bibliotheques, seed, roster, commit git, duree. Un CPU et un GPU ne
donnent pas des nombres bit-a-bit identiques ; sans cette trace, un tableau de
comparaison n'est pas defendable.
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

# --- Configuration de l'etude, identique aux notebooks -----------------------------

SAMPLE_SIZE = 500
SEED = 42

CONTEXT_LENGTH = None          # maximum du modele, comme en 05 et 06
CROSS_LEARNING_BATCH = 100     # series par groupe d'attention (reglage du fournisseur)
SHUFFLE_SEED = 0               # ordre des series pour le cross-learning

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


# --- Tracabilite --------------------------------------------------------------------

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
    """Le sidecar qui dit d'ou vient ce fichier. Jamais un artifact sans son meta."""
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


# --- Les donnees, chargees une fois -------------------------------------------------

class Study:
    """Tout ce que les etapes partagent. Charge paresseusement : `--list` ne lit rien."""

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
                f"\n{self.choice_path.name} absent.\n"
                "Le choix des longueurs de contexte n'est pas une etape de calcul : c'est\n"
                "une decision prise en regardant le sweep, dans le notebook 07, sur le Mac.\n"
                "Lancer d'abord context_coarse et context_fine, rapatrier les artifacts,\n"
                "executer le notebook 07, puis revenir."
            )
        return json.loads(self.choice_path.read_text())

    def coarse_grid(self) -> list[int]:
        """Grille grossiere bornee par l'historique le plus long.

        Toute longueur >= max_history lit l'historique entier, donc equivaut au
        modele vanilla : on garde max_history comme borne superieure unique.
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
        """Meilleure longueur par segment (jours 1-7 / 8-28), au metrique de selection.

        C'est le seul endroit ou le runner note des resultats : la grille fine depend
        des gagnants de la grille grossiere.
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


# --- Les etapes ---------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    description: str
    run: Callable[[Study], list[Path]]
    needs_gpu: bool = False
    cost: str = "light"            # "light" < 10 min, "heavy" au-dela
    requires: list[str] = field(default_factory=list)
    note: str = ""


def stage_vanilla(study: Study) -> list[Path]:
    """Chronos-2 sans rien : pas de covariable, pas de cross-learning, contexte maximum."""
    path = ARTIFACTS_DIR / "predictions" / f"chronos2_vanilla_{study.name}.parquet"

    def compute():
        parts = []
        for origin in study.windows["origin"].to_list():
            start = time.time()
            parts.append(forecast(history(study.sales, origin), origin, device=study.device))
            print(f"  origine {origin} : {len(study.series_ids)} series en {time.time() - start:.0f} s")
        return pl.concat(parts)

    predictions = cached(path, compute)
    expected = len(study.series_ids) * HORIZON * study.windows.height
    assert predictions.height == expected, f"{predictions.height} lignes, {expected} attendues"
    return [path]


def stage_uplift(study: Study) -> list[Path]:
    """Table d'event uplift de l'origine de validation (reprise du jeu XGBoost)."""
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
    assert missing == 0, "une covariable connue manque dans la fenetre"
    return context_known, future_known


def stage_covariates(study: Study) -> list[Path]:
    """Les quatre blocs de covariables connues : calendrier, prix, SNAP+evenements, tout."""
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
    """Cross-learning actif, series melangees pour que les groupes ne soient pas homogenes."""
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
    """Trois facons de signaler les periodes blanches : covariable passee, second canal, masque."""
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
    """Grille grossiere de longueurs de contexte, espacees sur une echelle log."""
    grid = study.coarse_grid()
    print(f"  grille grossiere ({len(grid)} longueurs) : {grid}")
    paths = []
    for length in grid:
        study.context_run(length)
        paths.append(experiment_path(f"context_{length}", study.name))
    return paths


def stage_context_fine(study: Study) -> list[Path]:
    """Grille fine, par pas d'une semaine a +/- 8 semaines des gagnants de la grille grossiere."""
    grid = study.coarse_grid()
    winners = study.segment_winners(grid)
    print(f"  gagnants grossiers : {winners}")
    fine = sorted({
        length
        for winner in winners
        for length in range(max(FINE_STEP, winner - FINE_HALF_WIDTH),
                            min(study.max_history, winner + FINE_HALF_WIDTH) + 1, FINE_STEP)
    } - set(grid))
    print(f"  grille fine ({len(fine)} nouvelles longueurs) : {fine}")
    paths = []
    for length in fine:
        study.context_run(length)
        paths.append(experiment_path(f"context_{length}", study.name))
    return paths


def _finetune_at(study: Study, origin: date, context_length: int, inputs: tuple) -> Path:
    output_dir = checkpoint_dir(study.name, origin, context_length,
                               NUM_STEPS, LEARNING_RATE, BATCH_SIZE)
    print(f"  origine {origin}, fine-tune au contexte {context_length}")
    finetune(*inputs, context_length=context_length, output_dir=output_dir,
             num_steps=NUM_STEPS, learning_rate=LEARNING_RATE,
             batch_size=BATCH_SIZE, device=study.device)
    return output_dir / "finetuned-ckpt"


def stage_finetune(study: Study) -> list[Path]:
    """LoRA aux deux longueurs retenues par le notebook 07, origine de validation."""
    choice = study.choice
    contexts = sorted({choice["late_context"], choice["early_context"]},
                      key=lambda length: length != choice["late_context"])   # la longue d'abord
    inputs = finetune_inputs(study.sales, study.validation_origin, study.series_ids)
    return [_finetune_at(study, study.validation_origin, length, inputs) for length in contexts]


def stage_finetune_predict(study: Study) -> list[Path]:
    """Previsions des checkpoints LoRA, tous appariements fine-tune x contexte de prediction."""
    choice = study.choice
    finetune_contexts = sorted({choice["late_context"], choice["early_context"]})
    predict_contexts = sorted({choice["early_context"], choice["late_context"]})
    origin = study.validation_origin
    paths = []
    for finetune_context in finetune_contexts:
        checkpoint = checkpoint_dir(study.name, origin, finetune_context, NUM_STEPS,
                                    LEARNING_RATE, BATCH_SIZE) / "finetuned-ckpt"
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint absent : {checkpoint}\nlancer d'abord l'etape finetune")
        for length in predict_contexts:
            name = (f"lora_ctx{finetune_context}_steps{NUM_STEPS}_lr{LEARNING_RATE:g}"
                    f"_bs{BATCH_SIZE}_context_{length}_origin{origin}")
            path = experiment_path(name, study.name)
            print(f"  LoRA@{finetune_context} -> prediction au contexte {length}")
            cached(path, lambda length=length, checkpoint=checkpoint: forecast(
                history(study.sales, origin), origin, context_length=int(length),
                model_path=checkpoint, device=study.device))
            paths.append(path)
    return paths


def make_stage_finetune_test(contexts: list[int] | None):
    def stage_finetune_test(study: Study) -> list[Path]:
        """Tout ce dont la section 4 du notebook 08 a besoin a l'origine de TEST.

        Produit, pour chaque longueur passee en --finetune-context :
          - le checkpoint LoRA re-entraine a l'origine de test ;
          - ses previsions aux longueurs de prediction retenues ;
        et, dans tous les cas, les previsions vanilla a l'origine de test aux memes
        longueurs -- le notebook en a besoin si le gagnant d'un segment est vanilla.

        Les longueurs doivent etre passees explicitement. C'est volontaire : les
        deduire ici reviendrait a selectionner sur la fenetre de test, ce que le
        protocole du notebook 03 interdit. Passer les deux longueurs est legitime
        -- calculer n'est pas choisir, et le notebook 08 ne note que le gagnant
        designe sur la validation -- mais le geste reste explicite et trace.
        """
        if not contexts:
            raise SystemExit(
                "\nfinetune_test exige --finetune-context.\n"
                "  --finetune-context 1829        la seule longueur retenue sur validation\n"
                "  --finetune-context 1099,1829   les deux, pour que le notebook 08 tourne\n"
                "                                 d'un bout a l'autre sans rien calculer en local\n"
                "La deduire ici reviendrait a choisir sur la fenetre de test."
            )
        origin = study.test_origin
        choice = study.choice
        predict_contexts = sorted({choice["early_context"], choice["late_context"]})
        paths = []

        # Vanilla a l'origine de test. Nom identique a celui du notebook 08.
        for length in predict_contexts:
            path = experiment_path(f"context_{length}_origin{origin}", study.name)
            print(f"  vanilla test, contexte {length}")
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
                print(f"  LoRA@{finetune_context} -> prediction test au contexte {length}")
                cached(path, lambda length=length, checkpoint=checkpoint: forecast(
                    history(study.sales, origin), origin, context_length=int(length),
                    model_path=checkpoint, device=study.device))
                paths.append(path)
        return paths
    return stage_finetune_test


def build_registry(finetune_contexts: list[int] | None) -> dict[str, Stage]:
    return {s.name: s for s in [
        Stage("vanilla", "Chronos-2 vanilla, deux origines", stage_vanilla,
              cost="heavy", note="~2 min GPU / ~45 min CPU"),
        Stage("uplift", "table d'event uplift (lecture des 30 490 series)", stage_uplift,
              cost="light", note="CPU, pas de modele"),
        Stage("covariates", "4 blocs de covariables connues", stage_covariates,
              needs_gpu=True, cost="heavy", requires=["uplift"], note="~4 x 1 min GPU"),
        Stage("cross_learning", "cross-learning actif, series melangees", stage_cross_learning,
              needs_gpu=True, cost="heavy"),
        Stage("blank", "3 variantes de periodes blanches", stage_blank,
              needs_gpu=True, cost="heavy"),
        Stage("context_coarse", "grille grossiere de contextes", stage_context_coarse,
              needs_gpu=True, cost="heavy", note="~21 longueurs, ~8 min GPU"),
        Stage("context_fine", "grille fine autour des gagnants", stage_context_fine,
              needs_gpu=True, cost="heavy", requires=["context_coarse"],
              note="~30 longueurs, ~12 min GPU"),
        Stage("finetune", "LoRA aux 2 contextes, origine validation", stage_finetune,
              needs_gpu=True, cost="heavy",
              note="~32 min / contexte GPU ; exige context_choice.json"),
        Stage("finetune_predict", "previsions des checkpoints LoRA", stage_finetune_predict,
              needs_gpu=True, cost="heavy", requires=["finetune"]),
        Stage("finetune_test", "origine de test : vanilla + re-fine-tune", make_stage_finetune_test(finetune_contexts),
              needs_gpu=True, cost="heavy", note="exige --finetune-context"),
    ]}


# Ordre d'execution de --all : il s'arrete avant finetune, qui attend ta decision.
DEFAULT_ORDER = ["vanilla", "uplift", "covariates", "cross_learning", "blank",
                 "context_coarse", "context_fine"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", help="noms separes par des virgules")
    parser.add_argument("--all", action="store_true",
                        help=f"tout jusqu'au point de decision : {', '.join(DEFAULT_ORDER)}")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="recalcule meme si les artifacts existent")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-slow", action="store_true",
                        help="autorise une etape needs_gpu sur CPU")
    parser.add_argument("--finetune-context", default=None,
                        help="longueur(s) retenue(s) sur validation, pour finetune_test ; "
                             "une ou plusieurs, separees par des virgules")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    finetune_contexts = ([int(v) for v in args.finetune_context.split(",") if v.strip()]
                         if args.finetune_context else None)
    registry = build_registry(finetune_contexts)

    if args.list:
        print(f"{'etape':20s} {'gpu':4s} {'cout':7s} {'depend de':18s} note")
        for stage in registry.values():
            print(f"{stage.name:20s} {'oui' if stage.needs_gpu else '-':4s} "
                  f"{stage.cost:7s} {','.join(stage.requires) or '-':18s} {stage.note}")
        print(f"\n--all = {' -> '.join(DEFAULT_ORDER)}")
        print("puis : notebook 07 sur le Mac -> context_choice.json -> finetune")
        return

    if args.all:
        names = DEFAULT_ORDER
    elif args.stages:
        names = [n.strip() for n in args.stages.split(",") if n.strip()]
    else:
        parser.error("donner --stages, --all ou --list")

    unknown = [n for n in names if n not in registry]
    if unknown:
        parser.error(f"etape(s) inconnue(s) : {unknown}. Voir --list")

    device = resolve_device(args.device)
    report = device_report(device)
    print("=" * 70)
    print(json.dumps(report, indent=2, default=str))

    blocked = [n for n in names if registry[n].needs_gpu and not device.startswith("cuda")]
    if blocked and not args.allow_slow:
        raise SystemExit(
            f"\netape(s) marquee(s) needs_gpu lancee(s) sur {device} : {blocked}\n"
            "Le facteur mesure est d'environ 24x : ce n'est probablement pas un choix.\n"
            "Lancer sur la tour, ou ajouter --allow-slow si c'est volontaire."
        )

    study = Study(device)
    print(f"roster {study.name}, device {device}")

    if args.dry_run:
        for name in names:
            print(f"  [dry-run] {name} : {registry[name].description}")
        return

    # Le dtype est verifie une fois, avant d'engager des heures de calcul.
    dtype = pipeline_dtype(get_pipeline(device=device))
    print(f"dtype du modele : {dtype}")
    if "float32" not in dtype:
        raise SystemExit(
            f"dtype {dtype} et non float32. Sur Pascal (sm_61) la demi-precision est\n"
            "un ralentissement d'un facteur ~64, pas une acceleration. On s'arrete."
        )

    total = time.time()
    for name in names:
        stage = registry[name]
        print(f"\n{'-' * 70}\n{name} : {stage.description}")
        start = time.time()
        try:
            produced = stage.run(study)
        except KeyboardInterrupt:
            print(f"\ninterrompu pendant {name} ; les artifacts deja ecrits sont conserves")
            sys.exit(130)
        seconds = time.time() - start
        for path in produced:
            if path.exists() and path.is_file():
                write_meta(path, name, device, seconds, {"dtype": dtype})
        print(f"{name} termine en {seconds / 60:.1f} min, {len(produced)} artifact(s)")

    print(f"\n{'=' * 70}\ntotal : {(time.time() - total) / 60:.1f} min")
    if set(names) & set(DEFAULT_ORDER) and not study.choice_path.exists():
        print("\nProchaine etape : rapatrier les artifacts, executer le notebook 07 sur le\n"
              "Mac, choisir les longueurs de contexte, puis lancer l'etape finetune.")


if __name__ == "__main__":
    main()
