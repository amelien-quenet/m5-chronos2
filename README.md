# M5 × Chronos-2

How much does a pretrained time-series foundation model buy you on hard intermittent retail
demand, and where does the error that remains actually live?

Measured on [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy):
30,490 item×store series of daily Walmart sales, 1,941 days, forecast 28 days ahead. 68% of
days are zero and the median series sells under half a unit a day.

## The result

Test window (2016-04-25 → 2016-05-22), 485 scored items. Lower is better.

| model | WMAPE | MAE | P90 | WQL |
|---|---|---|---|---|
| always zero | 100.00 | — | — | 1.0000 |
| seasonal naive (repeat last week) | 94.69 | — | — | — |
| mean of the last 28 days | 77.58 | — | — | — |
| XGBoost, 52 engineered features | 76.36 | 0.9486 | **2.060** | 0.7636 |
| **Chronos-2, zero-shot** | **70.01** | 0.8697 | 2.102 | 0.5500 |
| Chronos-2, context-tuned + LoRA | 70.03 | 0.8699 | 2.118 | **0.5487** |

Three things that are easy to get backwards:

- **Out of the box, with no features and no training, Chronos-2 beats a reproduced top-4%
  gradient-boosting solution by 8.3% of WMAPE.** The lags, rolling windows and per-weekday
  moments the baseline needs, it extracts from the raw history by reading it.
- **Everything after zero-shot was a wash.** Four families of covariates, cross-learning, a
  46-length context sweep, a horizon stitch and a LoRA fine-tune: the final model is
  marginally *worse* on the point forecast and 0.23% better on WQL.
- **XGBoost wins on P90.** On the worst days — the error you would staff for — the
  feature-engineered model is 2–3% ahead. And part of Chronos-2's WMAPE lead is structural:
  it returns a median, XGBoost estimates a conditional mean, and the median is the better
  point forecast under absolute-error metrics regardless of which model forecasts better.

M5 was chosen for two specific reasons rather than its fame: Chronos-2 has no training
leakage on it, so a zero-shot number means something; and its dynamics are mostly
endogenous, which makes it a test of the model rather than of the covariates one can invent
around it.

## Where the remaining error lives

A flat result invites "which knob did we miss?". The more useful question is whether the
headroom is reachable at all, and that is answerable.

Twenty-four weekly origins, scored on non-overlapping 7-day windows, decomposed into an item
effect, a week effect, and what is left:

- **42% of the variance in log WQL sits on the identity of the item; 2% on the week.**
  There are essentially no weeks that are hard for everyone. The observed rank agreement
  between consecutive origins looks weak (+0.52) only because a single 7-day window of an
  item selling a handful of units is 56% reliable — the decomposition predicts 0.477 from
  the variance shares alone, and the underlying ordering is stable at 0.86.
- **Two thirds of that item effect is volume** (r² = 0.63). "Group the items by difficulty"
  is, to first order, "bucket them by volume" — a sort on a number already in the history.
- **The third that survives is 15.6% of the whole budget**, larger than everything transient
  combined, and it is **not in the weekly shape**: 0.7% explained against a 1.5% null
  ceiling, p = 0.15 over 1,000 volume-stratified permutations, and at most 1.5% even if the
  shape were measured perfectly.

So the axis this study built the apparatus to test is ruled out, with a bound rather than a
shrug. Knowing which axis not to spend a month on is the deliverable.

## How it is organised

```
src/m5_chronos2/     the library: everything is a pure function of its inputs
  data.py            load, roster, launch flags
  evaluation.py      the frozen protocol: windows, scored pairs, actuals
  metrics.py         MAE, WMAPE, P90, WQL, RQL — checked against known answers
  baselines.py       zero, seasonal naive, 28-day mean
  gbdt_features.py   the 52 reproduced features and the direct 28-model strategy
  chronos2.py        pipeline cache, forecast, context stitch, LoRA fine-tune, devices
  blank_periods.py   long zero runs while on sale
  persistence.py     the variance decomposition, its controls, and the covariate tests
  shapes.py          the weekly-shape embedding and the clustering audit
  plots.py           one shared look for every figure
notebooks/           01–10, with outputs, so GitHub renders the figures
scripts/             run_stage.py (one stage, one artifact) and remote.sh (drive the GPU box)
viz/                 the manim animation of the weekly-shape embedding (its own uv project)
REMOTE.md            the GPU setup: architecture, routing rule, and eight verified traps
```

The notebooks read in order: **01** the data and what a zero means · **02** long zero runs ·
**03** the evaluation protocol · **04** the gradient-boosted baseline · **05** Chronos-2
zero-shot with its sanity checks · **06** the attempts that lost · **07** context length ·
**08** LoRA and the single test-window measurement · **09** where the error lives ·
**10** the clustering audit of the shape space.

**The notebooks compute nothing.** Every expensive result is produced by a stage
(`scripts/run_stage.py`) that writes a parquet plus a `.meta.json` recording the device, GPU,
library versions, seed, roster, git commit and dtype. The notebooks read those files and
plot them, so they re-run in seconds and a figure can never disagree with the artifact
behind it.

## Reproducing it

```bash
uv sync                                         # torch pinned by platform; never delete uv.lock
kaggle competitions download -c m5-forecasting-accuracy -p data/raw && unzip data/raw/*.zip -d data/raw
uv run python scripts/run_stage.py --help       # stages, one artifact each
uv run jupyter lab                              # then notebooks 01 → 10
```

`data/` and `artifacts/` are deliberately not versioned: code travels by git, data and
results by rsync. The repo is the method, not the output.

## Two commitments the numbers rest on

**Every choice is made on a validation origin and measured once on a test origin.** The
evaluation roster is frozen to disk before any model runs, and the count removed by each
filter is reported separately — a roster that silently changes between models turns a
comparison of models into a comparison of populations.

**Any two numbers whose difference is interpreted share their hardware.** CPU and GPU
disagree in the third decimal of WQL, which is enough to invert a ranking this tight, so the
whole Chronos-2 corpus is GPU and every artifact carries the device it was produced on. Two
checks close the loop: no experiment parquet without its meta, and no meta saying `cpu`.

## Write-up

The three-part write-up, with the figures and the reasoning in full:
**<https://amelien-quenet.github.io>**

## Licence

MIT — see [LICENSE](LICENSE).
