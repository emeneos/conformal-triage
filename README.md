# Conformal Triage of Skin Lesions in Two Latin American Cohorts

Code, notebooks and recorded-seed artefacts for the paper
*Conformal Triage of Skin Lesions in Two Latin American Cohorts:
A Fitzpatrick-Stratified Foundation-Model Approach*
(Valdez Kao & Pieciak, 2026 — arXiv link to be added).

The paper asks a question that is usually answered only after a model has been
trained: **which per-phototype coverage guarantees can a dermatology dataset
actually support?** We show the answer is computable from released metadata
alone — before any model runs — via a *certifiability audit* whose decisive
unit is the patient, not the lesion. We then execute a pre-specified protocol
(seed 2026) on two Latin American cohorts (HIBA, Argentina; PAD-UFES-20,
Brazil): a frozen PanDerm encoder, a linear head, Mondrian conformal
prediction sets per (phototype group, pooled-malignant) stratum with a single
pooled benign threshold, and a singleton-benign triage rule. Every certified
stratum attains its nominal coverage, a single malignant lesion in 100 draws is
released at α = 0.05, and the standard alternatives — a single marginal
threshold, a tuned softmax threshold, temperature scaling — undercover within
phototype groups or exceed their per-group error budgets in up to 70–76 % of
draws on the same scores. Post-hoc analyses attribute the price of the
certificate to imaging modality and to the head rather than to the guarantee.

## Repository layout

```
notebooks/
  01_data_and_embeddings.ipynb   Downloads ISIC 2019, HIBA and PAD-UFES-20 and
                                 extracts frozen PanDerm ViT-L embeddings
                                 (Google Colab, T4 GPU, ~40 min, run once).
  02_main_pipeline.ipynb         Harmonisation + Table 5 verification,
                                 pre-specified splits, linear head, APS scores,
                                 Mondrian calibration, triage, 100 evaluation
                                 draws, final audit (Table 6). CPU, ~15 min.
  03_ablations.ipynb             Core ablation tier: classification metrics,
                                 softmax-threshold and temperature-scaling
                                 reference rules, APS vs RAPS, ISIC-only head
                                 (domain ablation), AK-benign label convention
                                 with recomputed audit. CPU, ~20 min.
  04_posthoc_analyses.ipynb      Post-hoc analyses of Section 5.7: the
                                 sensitivity-referral trade-off on a grid of α,
                                 benign coverage by phototype under the pooled
                                 threshold, HIBA results by imaging modality and
                                 the mixed-lesion ablation, exact binomial and
                                 patient-bootstrap intervals, a PAD-only
                                 target-domain head, per-class F1/AUC. Verifies
                                 that it reproduces the Phase-3 draws (cell 3).
                                 CPU, ~15 min.
  05_fig_alpha_scope.ipynb       Generates Figure 1 from metadata alone. CPU, ~2 min.
scripts/
  01_download.sh                 Local (non-Colab) alternative for data + model.
  02_extract_embeddings.py       Local embedding extraction (CUDA or CPU).
figures/
  fig_alpha_scope.pdf            Figure 1 (output of notebook 05).
  fig_alpha_curve.pdf            Figure 2 (output of notebook 04).
results/                         Recorded-seed artefacts (seed 2026) backing
                                 every table and figure in the paper:
                                 head_weights.npz (the trained linear head),
                                 per-draw results, summary tables, audit
                                 tables, ablation outputs (notebooks 02-03)
                                 and the f5_* post-hoc outputs (notebook 04).
```

## Reproducing the paper

1. Open `notebooks/01_data_and_embeddings.ipynb` in Google Colab with a T4
   GPU runtime and run all cells. Artefacts (~61 MB of embeddings + the two
   ISIC-Archive metadata files) land in `conformal-triage/emb/` of your
   Google Drive. This is the only GPU step.
2. Run `notebooks/02_main_pipeline.ipynb` (CPU runtime). It verifies the
   cohort composition against Table 5 of the paper cell by cell, executes the
   pre-specified protocol at seed 2026, and writes the Section 5 summaries and
   the final Table 6 audit to `conformal-triage/results/`.
3. Run `notebooks/03_ablations.ipynb` (CPU runtime) for the core ablation
   tier of Section 4.4.
4. Run `notebooks/04_posthoc_analyses.ipynb` (CPU runtime) for Section 5.7.
   Its cell 3 checks that the partitions and per-stratum coverage reproduce
   `results/results_draws.csv` for every draw and error level.
5. Run `notebooks/05_fig_alpha_scope.ipynb` for Figure 1.

Every table in the paper is generated from the files in `results/`.

### Reproducibility notes

- All patient-level partitions are seeded (`SEED*1000 + d` for draw `d`) and
  the calibration strata are iterated in a fixed, sorted order, so the
  partitions and the de-clustering are identical across sessions and
  machines. Earlier versions iterated the strata in Python set order, which
  depends on per-process string hashing; the partitions were unaffected, but
  the per-patient lesion picked by the de-clustering could change between
  sessions. The tag `v1.1-preprint` is the first with the fixed order, and
  all artefacts in `results/` come from that code.
- One quantity does not reproduce bit-for-bit across machines: the HIBA
  benign threshold at α = 0.05. The deterministic APS score equals 1 whenever
  the true class ranks last, and the order statistic that defines that
  threshold falls inside the resulting tie; floating-point rounding breaks the
  tie differently on different hardware and moves the benign coverage of that
  stratum by up to 0.015 on about half of the draws. No malignant stratum and
  nothing in PAD-UFES-20 is affected. The paper states this in Sections 3.4
  and 5.7. A future re-run can make the threshold machine-independent by
  rounding APS scores (e.g. to 1e-9) before taking the quantile; we did not do
  so for the reported run so that the released artefacts and the released
  code match exactly.
- The reference rules of notebook 03 (Table 9) use their own seed stream
  (`(SEED+7)*1000 + d`); the same-draw comparison with the certified system is
  in notebook 04 (Figure 2).

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
`isic-cli` (installed by notebook 01). Notebooks 02–05 need only numpy,
pandas, scipy, scikit-learn and matplotlib. The recorded run used
scikit-learn 1.6.1.

## Citation

See `CITATION.cff`. Please cite the paper when using the code or the
recorded-seed artefacts.

## Licence

MIT for the code in this repository. Datasets and the PanDerm checkpoint
keep their own licences (see table above).
