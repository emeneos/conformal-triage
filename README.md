# Conformal Triage of Skin Lesions in Two Latin American Cohorts

Code, notebooks and recorded-seed artefacts for the paper
*Conformal Triage of Skin Lesions in Two Latin American Cohorts*
(Valdez Kao & Pieciak, 2026 — arXiv link to be added).

The paper asks a question that is usually answered only after a model has been
trained: **which per-phototype coverage guarantees can a dermatology dataset
actually support?** We show the answer is computable from released metadata
alone — before any model runs — via a *certifiability audit* whose decisive
unit is the patient, not the lesion. We then execute a pre-registered protocol
(seed 2026) on two Latin American cohorts (HIBA, Argentina; PAD-UFES-20,
Brazil): a frozen PanDerm encoder, a linear head, Mondrian conformal
prediction sets per malignant-class × Fitzpatrick-phototype cell, and a
singleton-benign triage rule. Every certified stratum attains its nominal
coverage, no malignant lesion is released at α = 0.05 in any of 100 draws,
and the standard alternatives — a single marginal threshold, a tuned softmax
threshold, temperature scaling — exceed their per-group error budgets in up
to 70–76 % of draws on the same scores.

## Repository layout

```
notebooks/
  01_data_and_embeddings.ipynb   Downloads ISIC 2019, HIBA and PAD-UFES-20,
                                 extracts frozen PanDerm ViT-L embeddings
                                 (Google Colab, T4 GPU, ~40 min, run once).
  02_main_pipeline.ipynb         Harmonisation + Table 5 verification,
                                 pre-registered splits, linear head, APS
                                 scores, Mondrian calibration, triage,
                                 100 evaluation draws, final audit (Table 6).
                                 CPU only, ~15 min.
  03_ablations.ipynb             Pre-registered core ablation tier:
                                 classification metrics, softmax-threshold and
                                 temperature-scaling reference rules, APS vs
                                 RAPS, ISIC-only head (domain ablation),
                                 AK-benign label convention with recomputed
                                 audit. CPU only, ~20 min.
scripts/
  01_download.sh                 Local (non-Colab) alternative for data + model.
  02_extract_embeddings.py       Local embedding extraction (CUDA or CPU).
figures/
  fig_alpha_scope.py             Generates Figure 1 from metadata alone.
results/                         Recorded-seed artefacts (seed 2026) backing
                                 every table and figure in the paper:
                                 head_weights.npz (the trained linear head),
                                 per-draw results, summary tables, audit
                                 tables, ablation outputs and head reports.
```

## Reproducing the paper

1. Open `notebooks/01_data_and_embeddings.ipynb` in Google Colab with a T4
   GPU runtime and run all cells. Artefacts (~61 MB of embeddings + the two
   ISIC-Archive metadata files) land in `conformal-triage/emb/` of your
   Google Drive. This is the only GPU step.
2. Run `notebooks/02_main_pipeline.ipynb` (CPU runtime). It verifies the
   cohort composition against Table 5 of the paper cell by cell, executes the
   pre-registered protocol at seed 2026, and writes the Section 5 summaries
   and the final Table 6 audit to `conformal-triage/results/`.
3. Run `notebooks/03_ablations.ipynb` (CPU runtime) for the core ablation
   tier of Section 4.

The conformal guarantee is model-agnostic: step 2 reproduces nominal coverage
even if the head is replaced, which is part of the point.

## Data and licences

No image data are redistributed here. The notebooks download everything from
the original sources at run time:

| Source | Access | Licence |
|---|---|---|
| ISIC 2019 (training set) | official challenge S3 links | CC-BY-NC (BCN20000, HAM10000 terms) |
| HIBA skin lesions 2019–2022 | ISIC Archive, collection 251 | CC-BY |
| PAD-UFES-20 | ISIC Archive, collection 406 (official mirror; also on Mendeley Data) | CC-BY 4.0 |
| PanDerm ViT-L checkpoint | official PanDerm repository release | per PanDerm licence |

## Requirements

Colab's stock environment plus `timm==0.9.16`, `open_clip_torch`, `gdown`,
`isic-cli` (installed by notebook 01). Notebooks 02–03 need only numpy,
pandas, scipy and scikit-learn.

## Citation

See `CITATION.cff`. Please cite the paper when using the code or the
recorded-seed artefacts.

## Licence

MIT for the code in this repository. Datasets and the PanDerm checkpoint
keep their own licences (see table above).
