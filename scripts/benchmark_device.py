#!/usr/bin/env python
"""Cout des deux operations lourdes de l'etude, mesure sur la machine courante.

Pourquoi ce script existe : on ne deporte un calcul sur la tour que si on sait de
combien elle est plus rapide. "Un GPU, donc plus vite" ne suffit pas -- sur une
GTX 1080 (Pascal), une demi-precision mal desactivee rend le GPU plus LENT que le
CPU du portable. Ce script produit le chiffre.

Il mesure deux choses, qui ne se comportent pas pareil :

  inference  -- ce que fait le sweep de contexte (notebook 07), des dizaines de fois ;
  fine-tune  -- ce que fait le notebook 08, et qui domine le cout total.

Lancer exactement la meme commande sur les deux machines ; seul le device change.
Les deux ecrivent leur resultat dans artifacts/benchmark/<device>.json, donc la
comparaison se lit apres coup sans avoir a retenir les chiffres.

    # sur le Mac
    uv run python scripts/benchmark_device.py
    # sur la tour
    ssh gpu 'cd m5-chronos2 && uv run python scripts/benchmark_device.py'

En cas de depassement de memoire GPU, baisser --micro-batch (8 Go de VRAM ne
tiennent pas 32 series a un contexte de 1829 jours en float32).
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

# Origine de validation du protocole (notebook 03). Jamais la fenetre de test :
# un benchmark n'a pas a toucher aux donnees qui servent au chiffre final.
VALIDATION_ORIGIN = date(2016, 3, 27)

BENCHMARK_DIR = ARTIFACTS_DIR / "benchmark"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default="n500_seed42")
    parser.add_argument("--series", type=int, default=100,
                        help="nombre de series ; 100 suffit pour un rapport de temps")
    parser.add_argument("--steps", type=int, default=50,
                        help="pas de fine-tune mesures, extrapoles a 1000")
    parser.add_argument("--context", type=int, default=1829,
                        help="contexte retenu par le notebook 07 pour les jours 8-28")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch EFFECTIF du fine-tune, celui qui definit la mise a jour")
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="series traitees a la fois ; None = choisi selon la VRAM")
    parser.add_argument("--infer-batch", type=int, default=100,
                        help="batch d'inference (vitesse seulement, pas de resultat)")
    parser.add_argument("--device", default=None, help="cpu, cuda ; None = auto")
    parser.add_argument("--skip-finetune", action="store_true")
    return parser.parse_args()


def load_series_ids(roster: str, limit: int) -> list[str]:
    """Les series notees a l'origine de validation, tronquees a `limit`, dans l'ordre.

    L'ordre est fixe (tri par id) pour que les deux machines mesurent exactement le
    meme travail : un sous-ensemble different donnerait un rapport de temps faux.
    """
    table = pl.read_parquet(ARTIFACTS_DIR / f"eval_roster_{roster}.parquet")
    scored = table.filter(pl.col("origin") == VALIDATION_ORIGIN)
    if scored.height == 0:
        raise SystemExit(
            f"aucune serie a l'origine {VALIDATION_ORIGIN} dans eval_roster_{roster}; "
            f"origines presentes : {sorted(table['origin'].unique().to_list())}"
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
    print("\n=== modele ===")
    print(json.dumps(limits, indent=2, default=str))
    if "float32" not in limits["dtype"]:
        print(f"\n  !! dtype {limits['dtype']} et non float32 : sur Pascal, c'est un")
        print("     ralentissement, pas une acceleration. Le chiffre ci-dessous ne")
        print("     mesure pas ce qu'on veut mesurer.")

    context = history(sales, VALIDATION_ORIGIN, ids)
    days = context.height // max(len(ids), 1)
    print(f"\n{len(ids)} series, origine {VALIDATION_ORIGIN}, "
          f"~{days} jours d'historique, contexte {args.context}")

    results = {
        "device": device,
        "report": report,
        "dtype": limits["dtype"],
        "series": len(ids),
        "context_length": args.context,
    }

    # --- 1. Inference : l'operation du sweep de contexte ---------------------------
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
    print(f"{len(ids)} series en {elapsed:.1f} s  ({per_series * 1000:.0f} ms/serie)")
    # Le sweep du notebook 07 fait une passe par configuration sur les ~479 series notees.
    print(f"  -> 1 configuration sur 479 series : {per_series * 479 / 60:.1f} min")
    print(f"  -> sweep de 50 configurations     : {per_series * 479 * 50 / 3600:.1f} h")

    # --- 2. Fine-tune : l'operation qui domine le cout -----------------------------
    if not args.skip_finetune:
        micro = args.micro_batch or suggest_micro_batch(args.batch_size, device) or args.batch_size
        print(f"\n=== fine-tune LoRA, {args.steps} pas, "
              f"batch {args.batch_size} = {micro} x {args.batch_size // micro} ===")

        train_inputs, validation_inputs = finetune_inputs(sales, VALIDATION_ORIGIN, ids)
        output_dir = BENCHMARK_DIR / f"{device.replace(':', '')}_steps{args.steps}"
        shutil.rmtree(output_dir, ignore_errors=True)   # jamais de reprise : on mesure

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
                f"\nmemoire GPU insuffisante avec --micro-batch {micro}.\n"
                f"Relancer avec la moitie : --micro-batch {max(micro // 2, 1)}\n"
                f"(le batch effectif reste {args.batch_size}, le resultat est inchange)"
            ) from error
        elapsed = time.time() - start

        per_step = elapsed / args.steps
        results |= {
            "finetune_steps": args.steps,
            "finetune_seconds": round(elapsed, 2),
            "finetune_seconds_per_step": round(per_step, 3),
            "micro_batch": micro,
        }
        print(f"{args.steps} pas en {elapsed:.1f} s  ({per_step:.2f} s/pas)")
        print(f"  -> 1000 pas (config du notebook 08) : {per_step * 1000 / 60:.0f} min")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / f"{device.replace(':', '')}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\necrit dans {path.relative_to(ARTIFACTS_DIR.parent)}")

    # --- 3. Comparaison, si l'autre machine a deja tourne ---------------------------
    others = sorted(p for p in BENCHMARK_DIR.glob("*.json") if p != path)
    for other_path in others:
        other = json.loads(other_path.read_text())
        if other.get("series") != results["series"] or \
           other.get("context_length") != results["context_length"]:
            print(f"\n{other_path.name} : configuration differente, non comparable")
            continue
        print(f"\n=== {results['device']} contre {other['device']} ===")
        for key, label in [("inference_seconds", "inference"),
                           ("finetune_seconds_per_step", "fine-tune (s/pas)")]:
            mine, theirs = results.get(key), other.get(key)
            if mine and theirs:
                print(f"{label:22s} {mine:8.2f}  vs {theirs:8.2f}   "
                      f"facteur {theirs / mine:.1f}x")


if __name__ == "__main__":
    main()
