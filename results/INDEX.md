# Results Index

Quick reference for all framework versions and where results live.

## Framework Version History

| Version | Key design | Status | Spec prefix |
|---------|-----------|--------|-------------|
| v14 | DVRL two-phase, gamma=0.99, curriculum, lambda=[0.3,0.7] | Superseded | `v14_*` |
| v15a | lambda=[0.3,0.5], util_guard=0.0 | Superseded | `v15a_*` |
| v16 | Biased data (scarcity framing), gamma=0.99 — ~50% deadzone | Superseded | `v16_*` |
| v17a | gamma=1.0, curriculum disabled, delta_scale=0.10 — deadzone fixed | Superseded by v18 | `v17a_*` |
| v17b | v17a + beta warm-start from alpha — worse than v17a, abandoned | Abandoned | `v17b_*` |
| v17c | delta_scale sweep (0.20, 0.30) — monotonically worse, abandoned | Abandoned | `v17c_*` |
| **v18** | **Global-only reward, lambda=[1,1], no DVRL. Current best REINFORCE.** | **Primary** | `v18_*` |
| v19 | Normalized reward (sigmoid_k=0), REINFORCE base | Diagnostic (2 seeds) | `v19_*` |
| v20 | CMA-ES evolutionary search, normalized reward | Active (5 seeds on DRAC) | `v20_*` |
| v21 | CMA-ES + OT local reward (lambda=[0.7,0.7], w_ot=1.0) | Active (5 seeds on DRAC) | `v21_*` |
| v22 | REINFORCE + OT local reward (lambda=[0.7,0.7], w_ot=1.0) | Active (smoke passed) | `v22_*` |

## Baseline Methods

| Method | Type | Notes |
|--------|------|-------|
| GroupDRO | Reweighting | Fails under severe scarcity (motivation claim) |
| OT Repair | Reprojection | Good EO but consistent utility cost |
| FLB | Reweighting | Best fairness on census; fails on COMPAS sex |
| CTGAN | Generative | GANbased tabular synthesis |
| FairTabDDPM | Generative | Diffusion-based; fails on COMPAS/capture24 |
| SMOTE | Oversampling | Two-phase PCA-10, matched budget — sanity check |

## Summary

- [cross_dataset_summary.md](cross_dataset_summary.md) — EO/F1w/AUC tables for all methods across all three datasets at a glance

## Dataset Results Files

- [census_income.md](census_income.md) — UCI Adult, sex, bias_pct=0.10, DA+=43
- [capture24.md](capture24.md) — Oxford wearable, sex, bias_pct=0.02, DA+=45
- [compas_sex.md](compas_sex.md) — COMPAS, sex, bias_pct=0.14, DA+=43
- [compas_race.md](compas_race.md) — COMPAS, race (Caucasian), bias_pct=0.05, DA+=40 — narrative fails here
- [acs_income.md](acs_income.md) — ACS PUMS 2018 CA, sex, bias_pct=0.04 — pending results

## Analysis

- [motivation_scarcity_curve.md](motivation_scarcity_curve.md) — GroupDRO degradation across DA+ levels; paper decision pending

## Output Directories

| Directory | Contents |
|-----------|---------|
| `paper_results_v4/` | v19, v20, v21 runs (current) |
| `paper_results_v3/` | COMPAS race investigation runs |
| `paper_results_v2/` | Main paper ablation + baseline runs (v18 config) |
| `training_runs/` | Earlier runs (v14–v18) |
| `paper_figures_v3/` | Generated figures + LaTeX tables |
| `paper_specs_v2/` | Final paper specs (all datasets, all methods) |
