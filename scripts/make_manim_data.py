#!/usr/bin/env python
"""Donnees du film sur la forme hebdomadaire.

Ce que le film illustre
-----------------------
La construction du plongement : une serie quotidienne devient un vecteur de 7 nombres
(la part de la semaine portee par chaque jour), puis un point dans un plan 2D ou les
familles de formes se separent.

**Les familles sont les archetypes plantes du notebook 10, etape 1** -- le controle
positif -- et non des formes inventees pour la video. C'est ce qui rend le film honnete :
il montre a quoi *ressemblerait* une structure discrete, a une separation mesuree et
au-dessus du plancher de detection, pendant que le billet dit ce que les vraies donnees
se trouvent etre. Un profil de base tire de M5 porte les archetypes, les totaux
hebdomadaires sont les vrais, et le bruit de comptage est redessine comme dans l'audit.

Trois fichiers, le contrat que la scene lit
-------------------------------------------
    weekly_series.parquet       series_id, date, value      la serie expliquee a l'ecran
    archetype_embeddings.parquet series_id, archetype, v0..v6
    archetype_umap.parquet      series_id, archetype, umap_x, umap_y

Invariant verifie ici : le vecteur stocke de la serie expliquee est **exactement** celui
que la scene obtient en empilant les semaines affichees. Sinon le film montrerait une
chose et afficherait les chiffres d'une autre.

    uv run --extra viz python scripts/make_manim_data.py
"""

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from m5_chronos2.data import ARTIFACTS_DIR, roster_name
from m5_chronos2.evaluation import make_windows
from m5_chronos2.shapes import (WEEKDAYS, positive_control, separation, shape_matrix,
                                umap_layout, weekly_share_array, weekly_shapes)

SAMPLE_SIZE = 500
SEED = 42

# Nombre de semaines que la scene fait defiler. Le vecteur affiche est l'agregat de
# ces semaines-la, donc ce nombre appartient autant aux donnees qu'a l'animation.
N_WEEKS = 20

N_ARCHETYPES = 3          # une couleur, une tournee par archetype dans le film
SEPARATION = 0.8          # bien au-dessus du plancher de detection mesure au notebook 10
MIN_WEEKS = 26

ARCHETYPE_NAMES = ["weekend", "weekday", "midweek"]


def base_profile_and_totals(roster: str, last_day):
    """Profil moyen reel et totaux hebdomadaires reels : ce sur quoi la simulation repose."""
    sales = pl.read_parquet(ARTIFACTS_DIR / f"sales_flags_{roster}.parquet")
    shapes_table = weekly_shapes(sales, last_day=last_day, min_weeks=MIN_WEEKS)
    ids, blocks, totals = weekly_share_array(sales, last_day=last_day)

    position = {series_id: index for index, series_id in enumerate(ids)}
    rows = [position[series_id] for series_id in shapes_table["id"].to_list()]
    totals = [totals[row] for row in rows]

    base = shape_matrix(shapes_table).mean(axis=0)
    return base / base.sum(), totals


def example_series(profile: np.ndarray, weekly_totals: np.ndarray,
                   generator: np.random.Generator):
    """Tire N_WEEKS semaines de ventes quotidiennes, et le vecteur qu'elles donnent.

    Le vecteur est recalcule a partir des semaines tirees, pas copie du profil latent :
    c'est la double normalisation du rapport, appliquee aux chiffres que la scene va
    reellement empiler.
    """
    totals = np.maximum(np.rint(weekly_totals[:N_WEEKS]), 1).astype(np.int64)
    if len(totals) < N_WEEKS:                      # serie courte : on boucle ses totaux
        totals = np.resize(totals, N_WEEKS)
    counts = generator.multinomial(totals, profile)          # N_WEEKS x 7

    shares = counts / counts.sum(axis=1, keepdims=True)      # L1 par semaine
    vector = shares.mean(axis=0)                             # puis moyenne, qui somme a 1
    return counts, vector / vector.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="../code/manim/data",
                        help="dossier ou la scene lit ses parquets")
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()

    out = Path(arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(arguments.seed)

    roster = roster_name(SAMPLE_SIZE, SEED)
    last_day = make_windows().filter(pl.col("role") == "validation")["origin"][0]
    base, totals = base_profile_and_totals(roster, last_day)
    print(f"profil de base reel : {np.round(base, 3)}")
    print(f"{len(totals)} series, mediane {np.median([t.mean() for t in totals]):.1f} unites/semaine")

    simulation = positive_control(base, totals, n_clusters=N_ARCHETYPES,
                                  sep=SEPARATION, seed=arguments.seed)
    shapes = simulation.shapes.copy()
    print(f"{N_ARCHETYPES} archetypes, separation mesuree delta = {simulation.separation:.2f}")

    # La serie expliquee : ses semaines sont tirees, donc son vecteur doit etre refait a
    # partir d'elles et remis dans la matrice AVANT la projection, sans quoi son point
    # ne serait pas celui de la serie montree.
    example = int(np.flatnonzero(simulation.labels == 0)[0])
    counts, vector = example_series(simulation.profiles[example], totals[example], generator)
    shapes[example] = vector

    coordinates, method = umap_layout(shapes, seed=arguments.seed)
    print(f"projection : {method}")

    ids = [f"S{index:04d}" for index in range(len(shapes))]
    names = [ARCHETYPE_NAMES[label] for label in simulation.labels]
    example_id = ids[example]

    # --- l'invariant ---------------------------------------------------------------
    redone = counts / counts.sum(axis=1, keepdims=True)
    redone = redone.mean(axis=0)
    redone = redone / redone.sum()
    error = float(np.abs(redone - shapes[example]).max())
    assert error < 1e-12, f"le vecteur stocke ne vient pas des semaines ecrites ({error:.1e})"
    assert abs(shapes[example].sum() - 1) < 1e-12
    print(f"invariant : vecteur stocke == agregat des {N_WEEKS} semaines ecrites "
          f"(ecart {error:.1e})")

    # --- ecriture ------------------------------------------------------------------
    start = last_day - timedelta(days=N_WEEKS * WEEKDAYS - 1)
    dates = pl.date_range(start, last_day, interval="1d", eager=True)
    assert len(dates) == N_WEEKS * WEEKDAYS
    series = pl.DataFrame({
        "series_id": [example_id] * (N_WEEKS * WEEKDAYS),
        "date": dates,
        "value": counts.reshape(-1).astype(np.int64),
    })
    series.write_parquet(out / "weekly_series.parquet")

    embeddings = pl.DataFrame(
        {"series_id": ids, "archetype": names}
        | {f"v{d}": shapes[:, d] for d in range(WEEKDAYS)}
    )
    embeddings.write_parquet(out / "archetype_embeddings.parquet")

    pl.DataFrame({
        "series_id": ids, "archetype": names,
        "umap_x": coordinates[:, 0].astype(float),
        "umap_y": coordinates[:, 1].astype(float),
    }).write_parquet(out / "archetype_umap.parquet")

    print(f"\necrit dans {out} :")
    print(f"  weekly_series.parquet         {series.height} jours de {example_id} "
          f"({names[example]})")
    print(f"  archetype_embeddings.parquet  {embeddings.height} series x {WEEKDAYS}")
    print(f"  archetype_umap.parquet        {method}, "
          f"{', '.join(f'{n}={names.count(n)}' for n in ARCHETYPE_NAMES[:N_ARCHETYPES])}")
    print(f"\nla scene lit SERIES_ID = {example_id}")


if __name__ == "__main__":
    main()
