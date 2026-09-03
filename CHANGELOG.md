# Changelog

## v1.1-preprint (2026-09-03)

- Notebooks 02 and 03: calibration strata are now iterated in a fixed sorted
  order (previously Python set order, which depends on per-process string
  hashing). Partitions are unchanged; the per-patient lesion chosen by the
  de-clustering is now reproducible across sessions. All artefacts in
  `results/` regenerated from this code at seed 2026.
- New notebook 04 (`04_posthoc_analyses.ipynb`): post-hoc analyses of
  Section 5.7 of the paper. Reproduces the Phase-3 draws exactly (same seeds,
  same random-stream consumption, same scikit-learn code path for the
  probabilities) and verifies it against `results/results_draws.csv`.
  Outputs `results/f5_*.csv` and Figure 2.
- New notebook 05 (`05_fig_alpha_scope.ipynb`): generates Figure 1 from the
  ISIC-Archive metadata; replaces `figures/fig_alpha_scope.py`, which read
  the original Mendeley/HIBA CSV schemas from local paths.
- All notebooks, scripts and comments in English.
- README: reproducibility notes (fixed stratum order; the APS tie at score 1
  that makes the HIBA benign threshold at α = 0.05 machine-dependent at the
  last bit; separate seed stream of the reference rules).

## v1.0-preprint (2026-08-31)

- Initial release: notebooks 01-03, local scripts, recorded-seed artefacts.
