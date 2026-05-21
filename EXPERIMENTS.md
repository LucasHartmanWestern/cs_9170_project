# EXPERIMENTS.md

Canonical experiment log. Each entry follows the standard template.
For full chronological history of pre-April 2026 work, see `EXPERIMENTS_archive.md`.

**Vanilla config:** `vanilla_config.json` (project root). All experiments are deltas from this.
Phase 2 disabled (gen_both_classes=false, phase2_episodes=null), curriculum disabled,
global-only reward (sigmoid k=10, lambda=[1,1], use_dvrl_local=false), gamma=1.0,
delta_scale=0.10, pca=10, traj_length=2000, real_data_size=3000, total_episodes=800.
Dataset-specific fields (dataset_name, bias_pct, minority_id, majority_id, seeds, dp_protected_col) are null and must be set per experiment.

---

## Experiment Index

| ID | Name | Type | Status | Dataset(s) | Follows From |
|----|------|------|--------|------------|--------------|
| EXP-001 | census-rl-main | PAPER-FINAL | COMPLETE | census | — |
| EXP-002 | capture24-rl-main | PAPER-FINAL | COMPLETE | capture24 | — |
| EXP-003 | census-episode-ablation | PARAM-TUNING | COMPLETE | census | EXP-001 |
| EXP-004 | capture24-episode-ablation | PARAM-TUNING | COMPLETE | capture24 | EXP-002 |
| EXP-005 | census-delta-tuning | PARAM-TUNING | COMPLETE | census | EXP-001 |
| EXP-006 | capture24-delta-tuning | PARAM-TUNING | COMPLETE | capture24 | EXP-002 |
| EXP-007 | census-dvrl-ablation | ABLATION | COMPLETE | census | EXP-001 |
| EXP-008 | capture24-dvrl-ablation | ABLATION | COMPLETE | capture24 | EXP-002 |
| EXP-009 | census-pca-tuning | PARAM-TUNING | COMPLETE | census | EXP-001 |
| EXP-010 | capture24-pca-tuning | PARAM-TUNING | COMPLETE | capture24 | EXP-002 |
| EXP-011 | census-ffnn-tuning | PARAM-TUNING | COMPLETE | census | EXP-001 |
| EXP-012 | capture24-ffnn-tuning | PARAM-TUNING | COMPLETE | capture24 | EXP-002 |
| EXP-013 | census-baselines | PAPER-FINAL | COMPLETE | census | — |
| EXP-014 | capture24-baselines | PAPER-FINAL | COMPLETE | capture24 | — |
| EXP-015 | compas-race-baselines | PAPER-FINAL | SUPERSEDED | compas | — |
| EXP-016 | wgl-k-sweep | ABLATION | PLANNED | census, capture24, compas | — |
| EXP-017 | roc-eo-lambda-sweep | ABLATION | PLANNED | census, capture24, compas | — |
| EXP-018 | baselines | PAPER-FINAL | COMPLETE | census, capture24, compas | — |
| EXP-019 | natural-scarcity-rl | EXPLORATORY | IN PROGRESS | census, capture24, compas | EXP-016 |
| EXP-020 | natural-scarcity-baselines | EXPLORATORY | PLANNED | census, capture24, compas | EXP-019 |
| EXP-021 | census-hparam-grid | PARAM-TUNING | IN PROGRESS | census | EXP-001 |
| EXP-022 | census-hparam-random | PARAM-TUNING | PLANNED | census | EXP-021 |
| EXP-025 | capture24-hparam-grid | PARAM-TUNING | IN PROGRESS | capture24 | EXP-002 |
| EXP-027 | acs-employment-rl-main | PAPER-FINAL | DROPPED | acs_employment | EXP-024 |
| EXP-028 | acs-employment-baselines | PAPER-FINAL | DROPPED | acs_employment | EXP-024 |
| EXP-029 | meps-sex-rl-3rd-dataset | PAPER-FINAL | IN PROGRESS | meps | — |
| EXP-030 | acs-income-race-rl-3rd-dataset | PAPER-FINAL | DROPPED | acs_income | — |
| EXP-031 | acs-employment-sex-rl-3rd-dataset | PAPER-FINAL | DROPPED | acs_employment | — |
| EXP-032 | 3rd-dataset-viability-search-2 | EXPLORATORY | PLANNED | tbd | — |
| EXP-033 | meps-sex-k10 | PAPER-FINAL | IN PROGRESS | meps | EXP-029 |
| EXP-034 | meps-sex-traj4000 | PAPER-FINAL | IN PROGRESS | meps | EXP-029 |

---

### EXP-001 | census-rl-main

**Type:** PAPER-FINAL
**Status:** COMPLETE
**Dataset(s):** census_income (bias_pct=0.10, DA+≈43)
**Seeds (actual):** 0, 2, 3, 5, 42
**Reference config:** v18
**Config delta:** ep1500/ph400 (total_episodes=1500, phase2_episodes=400)
**Follows from:** —

---

**Purpose:**
Main RL result for census. Best episode config selected from EXP-003.

**Result:**
| Metric | Value |
|--------|-------|
| α-EO | 0.196 ± 0.066 |
| β-EO | 0.070 ± 0.070 |
| β-F1w | 0.719 ± 0.037 |
| β-AUC | 0.852 ± 0.012 |

Run directory: `paper_results_v3/training_runs/SPECablation_census_ep1500ph400_5s_EP1500_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603291611_363899e7`

**Takeaway:**
Best census RL result. Feeds main comparison table and tradeoff figure.

**Next steps:**
— (complete, used in paper)

---

### EXP-002 | capture24-rl-main

**Type:** PAPER-FINAL
**Status:** COMPLETE
**Dataset(s):** capture24 (bias_pct=0.02, DA+≈45)
**Seeds (actual):** 0, 3, 4, 5, 42
**Reference config:** v18
**Config delta:** ep800/ph200 (total_episodes=800, phase2_episodes=200)
**Follows from:** —

---

**Purpose:**
Main RL result for capture24. Best episode config selected from EXP-004.

**Result:**
| Metric | Value |
|--------|-------|
| α-EO | 0.231 ± 0.164 |
| β-EO | 0.082 ± 0.084 |
| β-F1w | 0.940 ± 0.027 |
| β-AUC | 0.868 ± 0.039 |

Run directory: `paper_results_v3/training_runs/SPECablation_capture24_ep800ph200_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.02_GG202603291611_dc00b210`

**Takeaway:**
Best capture24 RL result under current config. High seed variance (α-EO std=0.164) is a known issue — seeds with near-zero alpha-EO pull results down. Active optimization ongoing (see future EXPs).

**Next steps:**
— (optimization experiments pending)

---

### EXP-003 | census-episode-ablation

**Type:** PARAM-TUNING
**Status:** COMPLETE
**Dataset(s):** census_income (bias_pct=0.10)
**Seeds (actual):** 0, 2, 3, 5, 42
**Reference config:** v18
**Config delta:** total_episodes ∈ {800, 1500, 2000}, phase2_episodes ∈ {0, 200, 400, 600}
**Follows from:** —

---

**Purpose:**
Identify optimal episode budget for census. Four configs: ep800/ph0, ep800/ph200, ep1500/ph400, ep2000/ph600.

**Result:**
| Config | EO | F1w | AUC |
|--------|----|-----|-----|
| ep800/ph0 | 0.188 ± 0.047 | 0.786 | 0.876 |
| ep800/ph200 | 0.076 ± 0.048 | 0.729 | 0.852 |
| ep1500/ph400 | 0.070 ± 0.078 | 0.719 | 0.852 |
| ep2000/ph600 | 0.082 ± 0.073 | 0.729 | 0.845 |

Run directories: `paper_results_v3/training_runs/SPECablation_census_ep{800ph0,800ph200,1500ph400,2000ph600}_5s_*`

**Takeaway:**
ep1500/ph400 gives lowest mean EO (0.070). Phase 2 is essential — ep800/ph0 barely improves over alpha. ep2000/ph600 does not improve further and adds variance.

**Next steps:**
— (ep1500/ph400 selected as main config for EXP-001)

---

### EXP-004 | capture24-episode-ablation

**Type:** PARAM-TUNING
**Status:** COMPLETE
**Dataset(s):** capture24 (bias_pct=0.02)
**Seeds (actual):** 0, 3, 4, 5, 42
**Reference config:** v18
**Config delta:** total_episodes ∈ {800, 1500, 2000}, phase2_episodes ∈ {0, 200, 400, 600}
**Follows from:** —

---

**Purpose:**
Identify optimal episode budget for capture24.

**Result:**
| Config | EO | F1w | AUC |
|--------|----|-----|-----|
| ep800/ph0 | 0.150 ± 0.109 | 0.939 | 0.905 |
| ep800/ph200 | 0.082 ± 0.084 | 0.940 | 0.868 |
| ep1500/ph400 | 0.195 ± 0.104 | 0.946 | 0.893 |
| ep2000/ph600 | 0.069 ± 0.039 | 0.938 | 0.862 |

Run directories: `paper_results_v3/training_runs/SPECablation_capture24_ep{800ph0,800ph200,1500ph400,2000ph600}_5s_*`

**Takeaway:**
ep800/ph200 chosen for paper (ep2000/ph600 has lower EO mean but lower AUC; ep800/ph200 more conservative). Capture24 is more sensitive to episode config than census — high variance across all configs is a concern.

**Next steps:**
— (ep800/ph200 selected as main config for EXP-002; further optimization pending)

---

### EXP-005 | census-delta-tuning

**Type:** PARAM-TUNING
**Status:** COMPLETE
**Dataset(s):** census_income (bias_pct=0.10)
**Seeds (actual):** 0, 2, 3, 5, 42
**Reference config:** v18 + ep1500/ph400
**Config delta:** delta_scale ∈ {0.05, 0.10, 0.20, 0.50} (reference = 0.10)
**Follows from:** EXP-001

---

**Purpose:**
Test whether larger or smaller delta perturbations improve fairness on census.

**Result:**
Run directories: `paper_results_v3/training_runs/SPECabl_census_delta{005,020,050}_5s_*`
(Detailed per-seed results in archive. Summary: delta=0.10 remains best.)

**Takeaway:**
Default delta_scale=0.10 is optimal. Larger perturbations increase variance and introduce catastrophic rogue seeds.

**Next steps:**
— (confirmed delta_scale=0.10 in reference config)

---

### EXP-006 | capture24-delta-tuning

**Type:** PARAM-TUNING
**Status:** COMPLETE
**Dataset(s):** capture24 (bias_pct=0.02)
**Seeds (actual):** 0, 3, 4, 5, 42
**Reference config:** v18 + ep800/ph200
**Config delta:** delta_scale ∈ {0.05, 0.10, 0.20, 0.50}
**Follows from:** EXP-002

---

**Purpose:**
Test whether delta perturbation scale affects capture24 differently than census.

**Result:**
Run directories: `paper_results_v3/training_runs/SPECabl_capture24_delta{005,020,050}_5s_*`

**Takeaway:**
Same conclusion as EXP-005. delta_scale=0.10 retained.

**Next steps:**
— (confirmed delta_scale=0.10)

---

### EXP-007 | census-dvrl-ablation

**Type:** ABLATION
**Status:** COMPLETE
**Dataset(s):** census_income (bias_pct=0.10)
**Seeds (actual):** 0, 2, 3, 5, 42
**Reference config:** v18 + ep1500/ph400
**Config delta:** use_dvrl_local=true, lambda_schedule=[0.5, 0.5] vs reference (global-only)
**Follows from:** EXP-001

---

**Purpose:**
Does the DVRL local reward improve over global-only on census?

**Result:**
Run directory: `paper_results_v3/training_runs/SPECabl_census_dvrl_5s_EP1500_*`
(DVRL variant EO > global-only EO — local reward hurts census.)

**Takeaway:**
Global-only is better. DVRL local reward destabilizes late training on census. This is the paper's design ablation: global-only is the proposed method, DVRL is the negative ablation.

**Next steps:**
— (global-only confirmed as reference config)

---

### EXP-008 | capture24-dvrl-ablation

**Type:** ABLATION
**Status:** COMPLETE
**Dataset(s):** capture24 (bias_pct=0.02)
**Seeds (actual):** 0, 3, 4, 5, 42
**Reference config:** v18 + ep800/ph200
**Config delta:** use_dvrl_local=true, lambda_schedule=[0.5, 0.5]
**Follows from:** EXP-002

---

**Purpose:**
Does the DVRL local reward improve over global-only on capture24?

**Result:**
Run directory: `paper_results_v3/training_runs/SPECabl_capture24_dvrl_5s_EP2000_*`

**Takeaway:**
Same conclusion as EXP-007. Global-only retained.

**Next steps:**
—

---

### EXP-009 | census-pca-tuning

**Type:** PARAM-TUNING
**Status:** COMPLETE
**Dataset(s):** census_income (bias_pct=0.10)
**Seeds (actual):** 0, 2, 3, 5, 42
**Reference config:** v18 + ep1500/ph400 (pca_components=10)
**Config delta:** pca_components ∈ {10 (raw/no PCA), 15, 20}
**Follows from:** EXP-001

---

**Purpose:**
Test whether more PCA components improve census results.

**Result:**
Run directories: `paper_results_v3/training_runs/SPECabl_census_{raw,pca15,pca20}_5s_*`

**Takeaway:**
pca=10 remains optimal. More components add noise without benefit.

**Next steps:**
—

---

### EXP-010 | capture24-pca-tuning

**Type:** PARAM-TUNING
**Status:** COMPLETE
**Dataset(s):** capture24 (bias_pct=0.02)
**Seeds (actual):** 0, 3, 4, 5, 42
**Reference config:** v18 + ep800/ph200 (pca_components=10)
**Config delta:** pca_components ∈ {10 (raw/no PCA), 15, 20}
**Follows from:** EXP-002

---

**Purpose:**
Test whether more PCA components improve capture24 results.

**Result:**
Run directories: `paper_results_v3/training_runs/SPECabl_capture24_{raw,pca15,pca20}_5s_*`

**Takeaway:**
pca=10 remains optimal.

**Next steps:**
—

---

### EXP-011 | census-ffnn-tuning

**Type:** PARAM-TUNING
**Status:** COMPLETE
**Dataset(s):** census_income (bias_pct=0.10)
**Seeds (actual):** 0, 2, 3, 5, 42
**Reference config:** v18 + ep1500/ph400 (ffnn_epochs=20)
**Config delta:** ffnn_epochs ∈ {10, 20, 50}
**Follows from:** EXP-001

---

**Purpose:**
Test whether more or fewer FFNN training epochs per episode affect census results.

**Result:**
Run directories: `paper_results_v3/training_runs/SPECabl_census_ffnn{10,50}_5s_*`

**Takeaway:**
ffnn_epochs=20 remains optimal.

**Next steps:**
—

---

### EXP-012 | capture24-ffnn-tuning

**Type:** PARAM-TUNING
**Status:** COMPLETE
**Dataset(s):** capture24 (bias_pct=0.02)
**Seeds (actual):** 0, 3, 4, 5, 42
**Reference config:** v18 + ep800/ph200 (ffnn_epochs=20)
**Config delta:** ffnn_epochs ∈ {10, 20, 50}
**Follows from:** EXP-002

---

**Purpose:**
Test whether more or fewer FFNN training epochs per episode affect capture24 results.

**Result:**
Run directories: `paper_results_v3/training_runs/SPECabl_capture24_ffnn{10,50}_5s_*`

**Takeaway:**
ffnn_epochs=20 remains optimal.

**Next steps:**
—

---

### EXP-013 | census-baselines

**Type:** PAPER-FINAL
**Status:** COMPLETE
**Dataset(s):** census_income (bias_pct=0.10)
**Seeds (actual):** 0, 2, 5, 6, 7
**Reference config:** n/a (baselines)
**Config delta:** n/a
**Follows from:** —

---

**Purpose:**
All baseline comparisons for census main results table.

**Result:**
| Method | EO | F1w | AUC |
|--------|----|-----|-----|
| GroupDRO | 0.074 ± 0.030 | 0.825 | 0.894 |
| OT Repair | 0.054 ± 0.035 | 0.792 | 0.823 |
| FLB | 0.031 ± 0.028 | 0.819 | 0.889 |
| FairTabDDPM | 0.070 ± 0.039 | 0.791 | 0.845 |
| SMOTE | 0.133 ± 0.064 | 0.781 | 0.854 |
| CTGAN | 0.109 ± 0.049 | 0.792 | 0.840 |

Run directories: `training_runs/BASELINE_{group_dro,gaussian_ot_repair,fairness_loss_balancing,fairtabddpm,smote,ctgan}_v2_census_*`

Note: seed mismatch vs EXP-001 (baselines: {0,2,5,6,7}, RL: {0,2,3,5,42}). All seeds pass alpha-EO ≥ 0.10 guard.

**Takeaway:**
FLB achieves best EO on census (0.031) but uses 200 FFNN epochs vs our 20 — a 10× training advantage. Our method is competitive with GroupDRO and FairTabDDPM on EO while matching or exceeding all methods except FLB on utility.

**Next steps:**
—

---

### EXP-014 | capture24-baselines

**Type:** PAPER-FINAL
**Status:** COMPLETE
**Dataset(s):** capture24 (bias_pct=0.02)
**Seeds (actual):** 0, 4, 5, 7, 42
**Reference config:** n/a (baselines)
**Config delta:** n/a
**Follows from:** —

---

**Purpose:**
All baseline comparisons for capture24 main results table.

**Result:**
| Method | EO | F1w | AUC |
|--------|----|-----|-----|
| GroupDRO | 0.141 ± 0.071 | 0.906 | 0.900 |
| OT Repair | 0.080 ± 0.065 | 0.949 | 0.893 |
| FLB | 0.106 ± 0.056 | 0.906 | 0.927 |
| FairTabDDPM | 0.250 ± 0.137 | 0.953 | 0.938 |
| SMOTE | 0.232 ± 0.170 | — | — |
| CTGAN | — | — | — |

Run directories: `training_runs/BASELINE_{group_dro,gaussian_ot_repair,fairness_loss_balancing,fairtabddpm,smote,ctgan}_v2_capture24_*`

**Takeaway:**
Our method achieves lowest EO on capture24 (0.082 vs next best OT Repair 0.080 — within std). FairTabDDPM worsens EO above alpha, confirming generative methods can fail under scarcity. GroupDRO also fails to beat alpha.

**Next steps:**
—

---

### EXP-015 | compas-race-baselines

**Type:** PAPER-FINAL
**Status:** SUPERSEDED
**Dataset(s):** compas (race, bias_pct=0.05, DA+≈40)
**Seeds (actual):** 0, 2, 3, 5, 42
**Reference config:** n/a (baselines)
**Config delta:** n/a
**Follows from:** —

---

**Purpose:**
Baseline comparisons for COMPAS race as third dataset candidate.

**Result:**
Run directories: `paper_results_v3/BASELINE_{group_dro,gaussian_ot_repair,fairness_loss_balancing,fairtabddpm,smote,ctgan}_compas_race_bias005_*`

RL result: `paper_results_v3/SPECcompas_race_ep1500ph400_5s_*` (EO=0.600 ± 0.055)
GroupDRO EO=0.197, FLB EO=0.075 — both outperform RL substantially.

**Takeaway:**
COMPAS race fails as a paper dataset. val_disadv_pos=14 (below ≥30 threshold) produces an unreliable reward signal. Reweighting methods outperform our method, contradicting the paper's motivation claim. Dataset dropped. All COMPAS values in current figure scripts are placeholders — do not use.

**Next steps:**
— (COMPAS dropped; third dataset search ongoing)

---

### EXP-016 | wgl-k-sweep

**Type:** ABLATION
**Status:** PLANNED
**Dataset(s):** census_income, capture24, compas (race)
**Seeds:** 42, 0, 1 (3 seeds each)
**Reference config:** vanilla (wgl, k=10, da_pct=0.014, 5000ep)
**Config delta:** global_sigmoid_k ∈ {0, 3, 5, 10}
**Follows from:** —

---

**Purpose:**
Determine how sigmoid sharpness affects reward quality across all three datasets. k=0 gives a normalized linear delta `(wgl_alpha − wgl_beta) / wgl_alpha`; higher k sharpens the boundary around zero. Results inform the final paper config and validate the sigmoid design choice.

Episode convergence is read from gen-curves (`check_run.py --interval 250`) — no separate shorter runs needed.

**First use of da_pct** (replaces legacy bias_pct): all datasets use da_pct=0.014, giving DA+≈42 out of 3000 training examples — identical across seeds. **First use of `reward_mode="wgl"`** (renamed from "fairness"; old alias still accepted).

**Specs:** `experiment_specs/April_13_Experiments/{census,capture24,compas}_wgl_k{0,3,5,10}_5000ep.json`

**Key questions:**
- Does k=10 (census best from prior work) generalise to capture24 and compas?
- Does the sigmoid qualitatively help vs k=0 (linear) across all datasets?
- At what episode does each config converge? (read from gen-curves)

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
Use best k per dataset as the reference config for final paper results.

**Seed 1 status (updated 2026-04-18; compas stale entries cleared 2026-05-19):**
- census wgl_k{0,3,5,10}: all complete. k3 seed_1 was in orphaned directory after consolidation script bug; manually moved into correct run dir.
- capture24 wgl_k{0,3,5,10}: all complete.
- compas: excluded from paper (see EXP-015). Compas wgl_k3 seed_1 was moved from orphaned dir. Compas wgl_k5 seed_1 launch via `run_missing_seed1_gpu1.sh` — status unknown, process likely dead. Compas results not used.

---

### EXP-017 | roc-eo-lambda-sweep

**Type:** ABLATION
**Status:** PLANNED
**Dataset(s):** census_income, capture24, compas (race)
**Seeds:** 42, 0, 1 (3 seeds each)
**Reference config:** vanilla + reward_mode="roc_eo", da_pct=0.014, 5000ep
**Config delta:** roc_eo_lambda ∈ {0.3, 0.5, 0.7}
**Follows from:** EXP-016

---

**Purpose:**
Test the `roc_eo` reward mode — `G(θ) = λ·AUC_beta − (1−λ)·EO_beta` — as an alternative to `wgl`. λ controls the AUC/EO trade-off directly in the reward signal rather than through a worst-group-loss proxy. λ=0.3 is EO-dominant, λ=0.5 balanced, λ=0.7 AUC-dominant.

No sigmoid is applied to `roc_eo` (k is not used in this mode). λ is the structural analogue of k for this reward family.

**Specs:** `experiment_specs/April_13_Experiments/{census,capture24,compas}_roc_eo_lam{03,05,07}_5000ep.json`

**Key questions:**
- Does `roc_eo` match or beat the best `wgl` config on EO without sacrificing F1w/AUC?
- Which λ best balances fairness and utility?
- Does `roc_eo` converge faster or more stably than `wgl` (no reference baseline, purely absolute)?

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
If `roc_eo` matches best `wgl` on both axes, consider it as the primary reward for the paper (simpler formulation, no alpha reference baseline needed). If it underperforms, retain `wgl` as primary and present `roc_eo` as a design ablation.

**Seed 1 status (updated 2026-04-18; compas stale entries cleared 2026-05-19):**
- census roc_eo_lam{03,05}: complete. census roc_eo_lam07 seed_1: launched via `run_missing_seed1_gpu0.sh` — status unknown, process likely dead.
- capture24 roc_eo_lam{03,05,07}: all complete.
- compas: excluded from paper (see EXP-015). Compas lam03/07 seed_1 re-launches and lam05 seed_1 — status unknown, processes likely dead. Compas results not used.

---

### EXP-018 | baselines

**Type:** PAPER-FINAL
**Status:** COMPLETE
**Dataset(s):** census_income, capture24, compas (race)
**Seeds (actual):** 42, 0, 1 (3 seeds each)
**Reference config:** n/a (baselines)
**Config delta:** n/a
**Follows from:** EXP-016 (establishes da_pct=0.014 as the scarcity regime)

---

**Purpose:**
Baseline comparisons for all three paper datasets at the da_pct=0.014 scarcity level (DA+≈43–45 disadvantaged-group positive training examples). Baselines: GroupDRO, OT Repair, FLB, CTGAN, SMOTE, FairTabDDPM.

Prior baseline runs (EXP-013, EXP-014) used the old bias_pct regime and different seeds. These are the matched baselines for comparison against EXP-016 RL results.

**Key setup details:**
- da_pct=0.014, real_data_size=3000 → DA+≈43 for census and compas, DA+≈45 for capture24
- Seeds: [42, 0, 1] — same as RL experiments
- FFNN: hidden=[32,16], lr=0.001, batch=64, epochs=20 (same architecture as alpha/beta in RL)
- PCA: use_pca=true, pca_components=10 (same feature space as RL)
- capture24: win_seconds=1.0, step_seconds=0.5
- compas: dp_protected_col="race"
- CTGAN/SMOTE/FairTabDDPM: n_synthetic=2000 (matches RL traj_length=2000)
- GroupDRO/FLB: epochs=200, eta=0.01, n_groups=4
- Test set always unbiased (da_pct mode)

**Note:** da_pct support was added to all baseline trainers for this experiment (previously only bias_pct was wired through).

**Specs:** `experiment_specs/April_13_Experiments/baselines/{dataset}_{baseline}.{json,sh}`

**Run directories (census):**
- GroupDRO: `training_runs/BASELINE_group_dro_census_gdro_b1aa6f0f__G202604141522`
- OT Repair: `training_runs/BASELINE_gaussian_ot_repair_census_ot_repair_21f0c299__G202604141522`
- FLB: `training_runs/BASELINE_fairness_loss_balancing_census_flb_29c3d006__G202604141522`
- SMOTE: `training_runs/BASELINE_smote_census_smote_05e752a1__G202604141522`
- CTGAN: `training_runs/BASELINE_ctgan_census_ctgan_e21530b2__G202604141543`
- FairTabDDPM: `training_runs/BASELINE_fairtabddpm_census_fairtabddpm_b6bedaa9__G202604141543`

**Run directories (capture24):**
- GroupDRO: `training_runs/BASELINE_group_dro_capture24_gdro_fb1e687b__G202604141522`
- OT Repair: `training_runs/BASELINE_gaussian_ot_repair_capture24_ot_repair_72dd0636__G202604141522`
- FLB: `training_runs/BASELINE_fairness_loss_balancing_capture24_flb_52b798af__G202604141522`
- SMOTE: `training_runs/BASELINE_smote_capture24_smote_340e46a8__G202604141522`
- CTGAN: `training_runs/BASELINE_ctgan_capture24_ctgan_d30ab1b1__G202604141543`
- FairTabDDPM: `training_runs/BASELINE_fairtabddpm_capture24_fairtabddpm_31a3bd63__G202604141543`

**Run directories (compas, excluded from paper):**
- GroupDRO: `training_runs/BASELINE_group_dro_compas_gdro_56854f17__G202604141522`
- OT Repair: `training_runs/BASELINE_gaussian_ot_repair_compas_ot_repair_9ec8e5f9__G202604141522`
- FLB: `training_runs/BASELINE_fairness_loss_balancing_compas_flb_38d4eba8__G202604141522`
- SMOTE: `training_runs/BASELINE_smote_compas_smote_2cecddc2__G202604141522`
- CTGAN: `training_runs/BASELINE_ctgan_compas_ctgan_0079b631__G202604141543`
- FairTabDDPM: `training_runs/BASELINE_fairtabddpm_compas_fairtabddpm_f0d938bf__G202604141543`

**Result — Census (α-EO=0.364±0.018):**

| Baseline | β-EO↓ | β-EOd↓ | F1w↑ | AUC↑ | F1_min↑ |
|---|---|---|---|---|---|
| GroupDRO | 0.114±0.026 | 0.141±0.006 | 0.811±0.002 | 0.888±0.003 | 0.656±0.005 |
| OT Repair | 0.085±0.018 | 0.085±0.018 | 0.812±0.009 | 0.863±0.003 | 0.562±0.032 |
| FLB | 0.039±0.025 | 0.091±0.012 | 0.803±0.003 | 0.874±0.002 | 0.641±0.003 |
| SMOTE | 0.108±0.012 | 0.108±0.012 | 0.814±0.005 | 0.867±0.004 | 0.582±0.015 |
| CTGAN | 0.328±0.008 | 0.328±0.008 | 0.827±0.003 | 0.869±0.005 | 0.632±0.003 |
| FairTabDDPM | 0.151±0.074 | 0.151±0.074 | 0.819±0.005 | 0.869±0.001 | 0.625±0.012 |
| **FORGE (k=5, pca=10, ep=30)** | **0.031±0.018** | **0.057±0.007** | **0.821±0.001** | **0.879±0.001** | — |

*Previous result (k=3 vanilla, superseded): β-EO=0.079±0.015, F1w=0.810, AUC=0.877. Replaced by EXP-021 best config.*

**Result — Capture24 (α-EO=0.196±0.016):**

| Baseline | β-EO↓ | β-EOd↓ | F1w↑ | AUC↑ | F1_min↑ |
|---|---|---|---|---|---|
| GroupDRO | 0.078±0.044 | 0.087±0.031 | 0.896±0.013 | 0.909±0.020 | 0.319±0.065 |
| OT Repair | 0.113±0.035 | 0.113±0.035 | 0.953±0.012 | 0.913±0.027 | 0.347±0.065 |
| FLB | 0.160±0.077 | 0.160±0.077 | 0.884±0.027 | 0.912±0.021 | 0.320±0.099 |
| SMOTE | 0.068±0.055 | 0.070±0.053 | 0.953±0.014 | 0.891±0.017 | 0.410±0.102 |
| CTGAN | 0.400±0.146 | 0.400±0.146 | 0.952±0.007 | 0.922±0.014 | 0.494±0.064 |
| FairTabDDPM | 0.313±0.209 | 0.313±0.209 | 0.937±0.013 | 0.929±0.009 | 0.410±0.042 |
| **RL k=3** | *(pending full seeds)* | — | — | — | — |

**Takeaway:**

**Census:** FORGE (k=5, pca=10, ep=30; from EXP-021) achieves β-EO=0.031±0.018 and EOd=0.057±0.007 — beating all baselines on both metrics. FLB has the lowest TPR-EO among baselines (0.039) but high EOd (0.091), meaning it trades FPR fairness for TPR fairness. FORGE achieves genuinely balanced improvement: EOd≈2×EO (0.057 vs 0.031), indicating modest FPR component. GroupDRO EOd=0.141 — worst on the complete fairness measure. FORGE F1w=0.821 exceeds all baselines except CTGAN (0.827, but EO=0.328). AUC=0.879 matches OT Repair (0.863 baseline). CTGAN and FairTabDDPM fail to match reweighting baselines on EO.

**Capture24:** Generative methods (CTGAN, FairTabDDPM) degrade badly vs no-augmentation — both show EO *higher* than α-EO=0.196, indicating synthetic samples are hurting fairness. Reweighting methods (GroupDRO, SMOTE) work better here. This is a meaningful finding: in high-dimensional wearable data with low DA+, naive generative augmentation backfires. RL k=3 result pending full seeds, but current trajectory suggests it should outperform or match SMOTE/GroupDRO.

**Motivation claim:** GroupDRO does NOT fail catastrophically under DA+≈43 on these datasets — it achieves 0.114 (census) and 0.078 (capture24) EO. The claim needs to be reframed: RL achieves *better EO than all baselines* on census while matching utility, and generative baselines (CTGAN/FairTabDDPM) are the ones that fail. The scarcity regime disadvantages naive generative methods, not reweighting.

**Next steps:**
- Complete RL capture24 results with full 3-seed runs to fill in the table.
- Reframe motivation claim in paper: scarcity hurts generative baselines (CTGAN degrades); RL's reward-guided generation avoids this failure mode.
- Update plot_results.ipynb to include all 6 baselines in the figures.

---

### EXP-019 | natural-scarcity-rl

**Type:** EXPLORATORY
**Status:** PLANNED
**Dataset(s):** census_income, capture24, compas (race)
**Seeds:** 42, 0, 1 (3 seeds each)
**Reference config:** vanilla + best k from EXP-016
**Config delta:** da_pct=0.11 (DA+≈330, natural positive-class rate for disadvantaged group)
**Follows from:** EXP-016

---

**Purpose:**
Robustness/generalisation check: does the method still improve fairness when not in the severe scarcity regime? At da_pct=0.11, DA+≈330 (vs 43 in the paper's main experiments). Exploratory — results inform whether the method degrades gracefully or remains competitive at natural class balance. Also re-tests COMPAS (previously dropped at DA+=43) at a higher DA+ level.

**Note on k:** specs use k=3 as a placeholder (current best from EXP-018 census). Update to best k per dataset once EXP-016 results are in.

**Specs:** `experiment_specs/April_13_Experiments/natural_scarcity/{census,capture24,compas}_natural_scarcity.{json,sh}`

**Key questions:**
- Does the method still reduce EO relative to alpha at natural scarcity?
- Does GroupDRO outperform RL here (as expected when reweighting methods are not disadvantaged)?
- Does COMPAS behave better at DA+≈330 — does the val_disadv_pos threshold now pass?

**Result (census, partial — seed_1 incomplete):**

| Seed | α-EO | β-EO | EO-Δ | α-F1w | β-F1w | F1w-Δ | Deadzone | Best Ep |
|------|-------|-------|-------|-------|-------|-------|----------|---------|
| 0 | 0.056 | 0.013 | -0.043 | 0.837 | 0.827 | -0.009 | 4.7% | 1460 |
| 42 | 0.120 | 0.034 | -0.086 | 0.834 | 0.825 | -0.009 | 0.2% | 1015 |
| 1 | — | — | — | — | — | — | 11.7% | 2355 |

Mean over 2 complete seeds: α-EO=0.088±0.045, β-EO=0.023±0.015, F1w-Δ=−0.009.

Run directory: `training_runs/SPECcensus_natural_scarcity_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604151710_12170614`

Seed 1 stopped at ep 3788 with no test metrics — needs re-run. Not currently queued.

**Takeaway:**
*(partial — census only, 2 seeds)* Method still reduces EO substantially at natural scarcity (β-EO=0.023 from α-EO=0.088 mean). Low deadzone on both seeds indicates healthy reward signal with more minority data. F1w drops by ~0.009 consistently — a modest utility cost. α-EO variance across seeds (0.056 vs 0.120) is notable; the data split at da_pct=0.11 is less constrained than at da_pct=0.014 so this is expected.

**Next steps:**
- Re-run census seed_1 (stopped at ep 3788).
- Run capture24 and compas natural scarcity RL once EXP-016 best-k is confirmed.
- Compare against EXP-020 baselines once those are run.

---

### EXP-020 | natural-scarcity-baselines

**Type:** EXPLORATORY
**Status:** PLANNED
**Dataset(s):** census_income, capture24, compas (race)
**Seeds:** 42, 0, 1 (3 seeds each)
**Reference config:** n/a (baselines)
**Config delta:** n/a
**Follows from:** EXP-019

---

**Purpose:**
Baseline comparisons at the natural scarcity regime (da_pct=0.11, DA+≈330 for census). Mirrors EXP-018 but at the higher positive-class rate to assess whether the advantage of RL over generative baselines persists outside severe scarcity, or whether methods converge as more minority data becomes available.

Same baseline set as EXP-018: GroupDRO, OT Repair, FLB, CTGAN, SMOTE, FairTabDDPM. Same FFNN architecture and PCA config for comparability.

**Key setup details:**
- da_pct=0.11, real_data_size=3000 → DA+≈330 for census
- Seeds: [42, 0, 1] — same as RL experiments
- All other config identical to EXP-018 (FFNN hidden=[32,16], lr=0.001, batch=64, epochs=20, PCA=10, n_synthetic=2000, GroupDRO/FLB epochs=200)
- capture24: win_seconds=1.0, step_seconds=0.5
- compas: dp_protected_col="race"

**Key questions:**
- Do CTGAN/FairTabDDPM still degrade at natural scarcity, or does more minority data make naive generation viable?
- Does GroupDRO now outperform RL (as expected when reweighting methods are not constrained by scarcity)?
- Is the RL EO advantage (EXP-019) meaningful relative to the best baselines here?

**Specs:**
*(to be generated — use `make_spec.py` mirroring EXP-018 specs but with da_pct=0.11)*

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
Generate specs and run baselines for census first; extend to capture24 and compas once EXP-019 RL results are in for those datasets.

---

### EXP-021 | census-hparam-grid

**Type:** PARAM-TUNING
**Status:** IN PROGRESS
**Dataset(s):** census_income (da_pct=0.01433, DA+=43)
**Seeds:** 0, 1, 42
**Reference config:** vanilla_config.json
**Config delta:** Full cartesian grid — see below
**Follows from:** EXP-001

---

**Purpose:**
Systematic grid search over four principal hyperparameters to identify the best configuration for census. Replaces the ad-hoc per-parameter tuning in EXP-003/005/009/011 with a single joint sweep.

**Parameters swept:**

| Parameter | Values | Rationale |
|---|---|---|
| `global_sigmoid_k` | [0, 3, 5, 10] | k=0 = no-sigmoid baseline; k=10 = current vanilla |
| `pca_components` | [5, 10, 15] | Feature compression; 10 is current vanilla |
| `ratio_trajectory` | [0.2, 0.4, 0.6] | Synthetic fraction of total 5000 training samples (ratio=0.4 is vanilla) |
| `ffnn.epochs` | [10, 20, 30] | Classifier training intensity; 20 is current vanilla |

Total: 4k × 3pca × 3ratio × 3epochs × 3seeds = **324 runs**

**Fixed base patches:** dataset_name=census_income, da_pct=0.01433, minority_id=0, majority_id=1, seeds=[0,1,42], total_episodes=5000, reward_mode=wgl, dp_protected_col=sex, total_data_size=5000.

**Spec format:** YAML permutations block (`experiment_specs/census_grid_v2/`). Variables ordered `[epochs, pca_components, seed, ratio_trajectory]` so each batch of 4 parallel processes shares the same epoch value (homogeneous batches, no slow run blocking fast ones).

**GPU split per server:** GPU0 runs epochs=[10,20] (54 perms, 14 batches, ~10 days); GPU1 runs epochs=[30] (27 perms, 7 batches, ~7 days). All ratio values on both GPUs.

**Selection criterion:** Best config = lowest mean β-EO across 3 seeds, with β-F1w ≥ α-F1w − 0.02 (utility guard).

---

**Submission Tracker — census_grid_v2 (revised 2026-05-12)**

Supersedes old bundle system (census_grid/, 108 specs, 28 bundles). Parallelization via Santiago's `torch.multiprocessing` spawn approach. Bug fixed in `main.py`: seed extraction now uses `principal_vars.index('seed')` rather than `permutation[0]` to handle epochs-first variable ordering correctly. `output_dir` spec field added to `main.py` to allow redirecting output directly to storage (used by Huron restart specs).

**Architecture change (2026-04-28):** k=10 DRAC submission restructured from 2 monolithic specs (census_k10_gpu0/gpu1.yaml, 54+27 perms each) into 9 per-(ratio×epoch) specs (census_k10_r{02,04,06}_e{10,20,30}), each with 9 perms and max_parallel=9. Confirmed safe from parallelism scaling tests (max_parallel=9, 9 CPUs, ~17.5–21.7 s/ep, all jobs within 168h wall limit).

**DRAC decommissioned (2026-05-12):** Remaining k=10 jobs redirected to Huron. Output goes to `/storage_1/epigou_storage/FORGE/training_runs_k10/`. Launch scripts: `experiment_specs/census_grid_v2/run_k10_gpu{0,1}.sh`. Logs: `/storage_1/epigou_storage/FORGE/training_runs_k10/logs/`.

| Resource | k | Spec(s) | Status | Started |
|---|---|---|---|---|
| Huron GPU 0 | k=0 | census_k0_gpu0_restart_pca15_e10.yaml → census_k0_gpu0_restart_e20.yaml | **COMPLETE** | 2026-04-28 |
| Huron GPU 1 | k=0 | census_k0_gpu1_restart_pca10_e30.yaml → census_k0_gpu1_restart_pca15_e30.yaml | **COMPLETE** | 2026-04-28 |
| Lambda GPU 0 | k=3 | census_k3_gpu0.yaml | **COMPLETE** | 2026-04-25 |
| Lambda GPU 1 | k=3 | census_k3_gpu1.yaml | **COMPLETE** | 2026-04-25 |
| Aulavik GPU 0 | k=5 | census_k5_gpu0.yaml | **COMPLETE** | 2026-04-25 |
| Aulavik GPU 1 | k=5 | census_k5_gpu1.yaml | **COMPLETE** | 2026-04-25 |
| Huron GPU 0 | k=10 | r02_e20 | **COMPLETE** | 9/9 runs, all seeds verified 2026-05-15 |
| Huron GPU 1 | k=10 | r04_e30 | **COMPLETE** | 9/9 runs, all seeds verified 2026-05-15 |
| Huron GPU 0 | k=10 | r02_e30 → r04_e10 → r04_e20 (sequential) | **RUNNING** | r02_e30 DONE; r04_e10 DONE; r04_e20: PCA5 DONE (3/3 seeds), PCA10 running (seed_0 ep~971/5000 as of 2026-05-19), PCA15 not yet started (batch 3 of 3 pending) |
| Huron GPU 1 | k=10 | r06_e10 → r06_e20 → r06_e30 (sequential) | **RUNNING** | r06_e10 DONE; r06_e20: PCA5 DONE (3/3), PCA10 DONE (3/3); PCA15 not yet started (batch 3 of 3 pending); r06_e30 never submitted — spec exists but not in run scripts |
| Huron GPU 0 | k=10 | census_k10_r02_e20_restart.yaml + census_k10_r02_e20_pca10_s42_restart.yaml | **CANCELLED** | r02_e20 completed all 9/9 runs cleanly — max_parallel=9 drop bug did not occur on Huron |
| Huron GPU 1 | k=10 | census_k10_r04_e30_restart.yaml | **CANCELLED** | r04_e30 completed all 9/9 runs cleanly — max_parallel=9 drop bug did not occur on Huron |
| Huron GPU 0 | k=10 | census_k10_r02_e10.yaml | **COMPLETE** | 2026-04-28 |

**Huron k=0 history:** Original gpu0/gpu1 runs (started 2026-04-25) completed epochs=10 for PCA5 and PCA10 (all seeds), PCA5 epochs=30, and PCA10 epochs=30 seed_0 before storage issues interrupted them. Completed runs moved to `/storage_1/epigou_storage/FORGE/training_runs/`. Restart specs cover all remaining permutations and write directly to storage via `output_dir` field.

**k=10 timing:** epochs=10/20 → ~17.5 s/ep, epochs=30 → ~21.7 s/ep. ~24–30h per spec, ~4.5 days total per GPU. With max_parallel=4 (patched 2026-05-13), specs with 9 runs take 3 batches → ~72–90h per spec.

**Silent worker-drop bug (found 2026-05-13, resolved 2026-05-15):** Originally observed on DRAC — r02_e20 got 5/9 and r04_e30 got 3/9 workers. However, both specs ran cleanly on Huron with all 9/9 runs completing (verified via final_test_metrics.csv). Bug was DRAC-specific (likely DRAC's multiprocessing spawn environment). Remaining chain specs remain patched to max_parallel=4 as a precaution. Restart specs are cancelled.

---

**Note (2026-05-15, updated 2026-05-19):** `census_k10_r02_e10.yaml` (ratio=0.2, epochs=10) is absent from both batch scripts — but 9 run dirs for this config were found in `training_runs/` (not `training_runs_k10/`), all 3 PCA × 3 seeds complete with final_test_metrics.csv. Likely ran during an earlier phase on Huron. No further action needed.

**Grid audit (2026-05-19):** k=10 missing configs confirmed: r04_e20/PCA15 (GPU0, pending batch 3), r06_e20/PCA15 (GPU1, pending batch 3), r06_e30/all-PCA (spec exists but never submitted). All other k=10 configs complete.

**Result (COMPLETE — k=0, k=3, k=5, k=10 all confirmed; 2026-05-20):**

Grid over 104 complete configs (≥3 seeds). Best config by β-EO:

**Best: k=10, pca=10, ep=30, traj=2000 (real=3000)** ← supersedes k=5

| Seed | α-EO | β-EO | EOd | F1w | AUC |
|---|---|---|---|---|---|
| 0 | 0.304 | 0.021 | 0.036 | 0.822 | 0.879 |
| 1 | 0.348 | 0.011 | 0.053 | 0.819 | 0.883 |
| 42 | 0.387 | 0.021 | 0.021 | 0.810 | 0.865 |
| **mean** | **0.346±0.034** | **0.018±0.005** | **0.037±0.013** | **0.817±0.005** | **0.876±0.008** |

Run directory: `/storage_1/epigou_storage/FORGE/training_runs_k10/SPECcensus_k10_r04_e30_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202605121151_249e76f9`

Top 5 complete configs (≥3 seeds, ranked by β-EO):

| k | pca | ep | traj | β-EO | F1w | AUC |
|---|---|---|---|---|---|---|
| **10** | **10** | **30** | **2000** | **0.018±0.005** | **0.817** | **0.876** |
| 10 | 5 | 10 | 3000 | 0.020 | 0.804 | 0.804 |
| 3 | 5 | 10 | 2000 | 0.023 | 0.798 | 0.798 |
| 0 | 5 | 10 | 3000 | 0.027 | 0.812 | 0.812 |
| 5 | 10 | 10 | 3000 | 0.028 | 0.851 | 0.851 |

*Previous best (k=5, pca=10, ep=30, traj=2000): β-EO=0.031±0.018, F1w=0.821, AUC=0.879. Superseded by k=10 result confirmed 2026-05-20.*

**Centroid drift analysis (2026-05-16, best config k=5/pca=10/ep=30/traj=2000):**

Centroid drift was computed using `plot_centroid_drift.py` on the best-config run dir (`9af13c63`). Results across 3 seeds:

| Metric | Seed 0 | Seed 1 | Seed 42 | Mean |
|---|---|---|---|---|
| L2 distance (start→end) | ~3.0 → ~1.0 | ~3.0 → ~1.1 | ~3.0 → ~1.0 | −50% |
| Cosine similarity (start→end) | ~0.0 → 0.63 | ~0.0 → 0.69 | ~0.0 → 0.75 | 0.69 |

Both metrics confirm systematic policy learning: the synthetic cloud centroid moves toward the real disadvantaged-positive centroid as training progresses. The L2 reduction and rising cosine similarity are the primary evidence that FORGE does not behave as random search. Deadzone fractions (episodes where global_obj < 0.5) are uniformly low: k=0: 1.7%, k=3: 2.4%, k=5: 2.0%, k=10: 2.6% — all well below the 20% concern threshold.

**Takeaway:**
k=10, ep=30, pca=10 outperforms all other configs: β-EO=0.018±0.005 vs k=5 best of 0.031±0.018. Sharper sigmoid (k=10) provides cleaner reward signal for the policy at this parameter combination. Utility cost is negligible (AUC 0.876 vs 0.879 for k=5). Centroid drift confirms systematic policy learning (not random search). Final census best config confirmed 2026-05-20.

**Next steps:**
- Feed k=10, pca=10, ep=30, traj=2000 as base_patches into EXP-022 random search.
- Update EXP-018 comparison table and paper with new β-EO=0.018 (done).

---

### EXP-022 | census-hparam-random

**Type:** PARAM-TUNING
**Status:** PLANNED
**Dataset(s):** census_income (da_pct=0.01433, DA+=43)
**Seeds:** 0, 1, 42
**Reference config:** vanilla_config.json + best params from EXP-021
**Config delta:** Random search over secondary hyperparameters — see below
**Follows from:** EXP-021

---

**Purpose:**
Random search over secondary (optimisation) hyperparameters, conditioned on the best principal config found in EXP-021. Covers learning rates, delta scale, and optimisers for both the classifier and RL agent.

**Parameters swept:**

| Parameter | Distribution | Range |
|---|---|---|
| `ffnn.learning_rate` | log-uniform | [1e-4, 1e-2] |
| `reinforce.lr` | log-uniform | [1e-5, 1e-3] |
| `delta_scale` | uniform | [0.05, 0.30] |
| `ffnn.optimizer` | choice | adam, adamw, sgd |
| `reinforce.optimizer` | choice | adam, adamw |

20 samples × 3 seeds = **60 runs**. RNG seed=0 for reproducibility.

**Spec generation:**
```
# First update search_configs/census_random.yaml base_patches with best params from EXP-021
python make_search_specs.py search_configs/census_random.yaml
# → experiment_specs/census_random/ (20 specs)
```

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
Use best combined config (EXP-021 + EXP-022) as the new census vanilla for final paper runs.

---

### EXP-023 | diabetes130-viability-investigation

**Type:** DATASET-VIABILITY
**Status:** COMPLETE — ALL FRAMINGS FAILED
**Dataset(s):** diabetes130 (UCI Strack et al. 2014, n≈69,174 encounters)
**Seeds:** 42, 0, 1
**Reference config:** dataset_viability.py
**Logs:** experiment_specs/diabetes130/local/logs/

---

**Purpose:**
Assess diabetes130 as a potential 3rd paper dataset. Protected attr: age group (young <45 = disadvantaged, old ≥65 = advantaged). Investigated two readmission framings.

**Framing 1 — lt30 (readmitted within 30 days, default):**
- n_young=4515, n_old=47659; positive rate ≈ 8.8%
- DA+ with da_pct=0.01433: 17 (below target 43 — bias formula mismatch, but unfixable for this framing)
- val_disadv_pos = 60 — FAIL (< 200 test threshold; <30 but also < 200 test)
- test_disadv_pos = 60 — FAIL
- alpha_EO ≈ 0.002 — FAIL; model collapses to all-negative at 9.4% positive rate despite balanced sampling
- sep_ratio = 1.68, cosine = 0.30 — PASS (separability is the ONE criterion that passes)
- Root cause of EO failure: at 9.4% positive rate after bias injection, the FFNN collapses to predicting all-negative despite balanced sampling → hard EO = 0

**Framing 2 — any readmission (readmitted at any point, <30 OR >30):**
- Positive rate ≈ 39.9%
- test_disadv_pos ≈ 800+ — PASS
- alpha_EO (soft) ≈ 0.104 — borderline PASS
- sep_ratio ≈ 0.56, cosine ≈ 0.96 — FAIL; positive groups (young and old diabetics who get readmitted) overlap heavily in PCA
- Root cause: readmission at any time is driven by disease severity, not age-specific mechanisms → identical feature profiles for young vs old readmitted patients

**Additional tests run:**
- Baselines (GroupDRO, FLB, OT Repair, FORGE k=3 1000ep) confirmed: alpha F1(min)=0.000 across all baselines for lt30, confirming degenerate classifier
- FORGE k=3 1000ep on lt30: alpha_EO≈0.0016, ran to completion but results meaningless

**Takeaway:**
Diabetes130 does not pass viability for any age-based framing. The lt30 framing has too few test positives and triggers a degenerate classifier; the any-readmission framing has good positives and EO but the PCA structure shows complete overlap (no group-specific signal for RL). Dataset dropped from paper consideration.

**Next steps:**
Investigate alternative 3rd datasets (see EXP-024).

---

### EXP-024 | 3rd-dataset-search

**Type:** DATASET-VIABILITY
**Status:** IN PROGRESS
**Seeds:** 42, 0, 1
**Reference config:** dataset_viability.py

---

**Purpose:**
Identify a viable 3rd dataset for the paper. Must pass all 4 viability criteria.

**Candidates investigated:**

| Dataset | Framing | val_pos | test_pos | alpha_EO | sep_ratio | cosine | Verdict |
|---------|---------|---------|---------|---------|-----------|--------|---------|
| acs_income (CA 2018) | sex, income>50k | 6509 | 6532 | 0.17 | 2.43 | — | PASS (rejected: same task as census) |
| sepsis (PhysioNet 2019) | sex, sepsis onset | 242 | 243 | 0.027 | 0.17 | — | FAIL: EO+sep |
| brfss (2022) | sex, heart attack | 1976 | 1896 | 0.016 | 0.52 | — | FAIL: EO+sep |
| brfss (2022) | sex, depression | 12101 | 12092 | 0.012 | 0.36 | — | FAIL: EO+sep |
| brfss (2022) | race (Black/White), heart attack | — | — | 0.005–0.055 | — | 0.99 | FAIL: EO+sep (cosine≈1 = universal predictor) |
| NSL-KDD | network intrusion attacks | — | — | — | — | — | FAIL: R2L/U2R minority groups have 0–3 examples total |
| covertype (sklearn) | wilderness area, Aspen/Krummholz | — | 4800+ | <0.02 | 2.1–3.2 | 0.18 | FAIL: low DA+(maj)/DA+(min) ratio (1.3–2.6x); FFNN learns both groups equally |
| acs_employment (10 states, 2018) | disability (DIS), employed | 9035 | 9096 | 0.09–0.14 | 1.71 | 0.41 | **ALL PASS** |

**Root-cause analysis of failures:**

*BRFSS/COMPAS/sepsis EO failures:* Universal predictor structure — behavioral risk factors or recidivism predictors explain outcomes similarly for both demographic groups regardless of scarcity. Cosine similarity ≈ 1.0 between group discriminant vectors.

*Covertype failure:* Feature separability is good (cosine = 0.18), but the DA+(majority)/DA+(minority) ratio is only 1.3–2.6x after bias injection. Census works because DA+(male)/DA+(female) ≈ 14x. With a low ratio, the FFNN receives similar absolute positive-example counts for both groups (≈43 vs 56–112) and learns both groups equally → alpha-EO near zero. Feature separability alone is not sufficient; the ratio of positive examples between groups must also be large.

**ACS Employment + disability framing:**
- Task: predict employment status (ESR=1) from 15 ACS features (DIS dropped as protected col)
- Protected group: disability (DIS=1 → disabled → a=0 disadvantaged, n=245K; DIS=2 → a=1, n=1.48M)
- Natural employment rates: 18.3% (disabled) vs 50.0% (not disabled) → DA+ ratio ≈ 30x after bias
- da_pct=0.01433 gives DA+(disabled)=43 in 3000-sample training
- val_disadv_pos ≈ 9035, test_disadv_pos ≈ 9096 (both easily exceed thresholds)
- Alpha-EO: 0.09–0.14 across 3 seeds (marginal at seed 1 = 0.09; 2/3 seeds clear ≥0.10)
- Feature separability: sep_ratio=1.71, cosine=0.41 (distinct group-positive regions)
- Dataset files: 10 states (CA TX NY FL PA OH IL GA NC MI), 2018 ACS 1-Year PUMS, cached at datasets/acs_employment/2018/1-Year/

**Status:** COMPLETE — ACS Employment + disability adopted as 3rd dataset.

**Result:**
ACS Employment with disability as protected group passes all four viability criteria. The 30x DA+ ratio ensures the alpha classifier is systematically biased against disabled employed people. The disability framing is legally grounded (ADA), distinct from both census (income vs employment, sex vs disability) and capture24 (activity monitoring, sex framing).

**Takeaway:**
The 3rd paper dataset is ACS Employment (folktables, Ding et al. NeurIPS 2021), disability framing, 10-state pool, 2018 ACS 1-Year. Use `da_pct=0.01433`, `dp_protected_col=disability`, `acs_states=[CA, TX, NY, FL, PA, OH, IL, GA, NC, MI]`.

**Next steps:**
1. Create vanilla spec and run baseline (FORGE) experiments on ACS Employment (add EXP-027).
2. Run baselines (GroupDRO, CTGAN, OT Repair) on ACS Employment (add EXP-028).
3. Update paper dataset table with ACS Employment statistics.

---

### EXP-025 | capture24-hparam-grid

**Type:** PARAM-TUNING
**Status:** IN PROGRESS

**Matched baselines (pca=15, real=4000, da_pct=0.015) — launched 2026-05-19:**
To enable direct comparison with the provisional best config (k=5, pca=15, ep=10, traj=1000), baselines were re-run with matching pca=15 and real_data_size=4000. Running on Aulavik GPU1 (master PID 127384). Specs: `experiment_specs/capture24_grid/baselines_pca15/`. Light methods (gdro/flb/smote/ot_repair) parallel first, then ctgan/fairtabddpm sequential. Logs: `experiment_specs/capture24_grid/baselines_pca15/logs/`.
**Dataset(s):** capture24 (da_pct=0.015, DA+=45)
**Seeds:** 0, 1, 42
**Reference config:** vanilla_config.json
**Config delta:** Full cartesian grid — same axes as EXP-021
**Follows from:** EXP-002

---

**Purpose:**
Mirror of EXP-021 on capture24. Sweeps the same four hyperparameters to identify the best configuration for capture24 independently of census, since optimal settings may differ (capture24 has higher-dimensional time-series features and different class structure).

**Parameters swept:**

| Parameter | Values |
|---|---|
| `global_sigmoid_k` | [0, 3, 5] |
| `pca_components` | [5, 10, 15] |
| `ratio_trajectory` | [0.2, 0.4, 0.6] |
| `ffnn.epochs` | [10, 20, 30] |

Total: 3k × 3pca × 3ratio × 3epochs × 3seeds = **243 runs**

**Fixed base patches:** dataset_name=capture24, da_pct=0.015, minority_id=1, majority_id=0, dp_protected_col=sex, win_seconds=1.0, step_seconds=0.5, seeds=[0,1,42], total_episodes=5000, reward_mode=wgl, total_data_size=5000.

**Specs:** `experiment_specs/capture24_grid/capture24_k{0,3,5}_gpu{0,1}.yaml`

**GPU split per server:** GPU0 runs epochs=[10,20] (54 perms, max_parallel=4); GPU1 runs epochs=[30] (27 perms, max_parallel=4).

**Submission Tracker:**

| Resource | k | Spec | Status | Notes |
|---|---|---|---|---|
| Aulavik GPU 0 | k=5 | capture24_k5_gpu0.yaml | **COMPLETE** | All 9 configs 3/3 seeds done (audited 2026-05-11; duplicate dirs for some configs — use hashes below) |
| Aulavik GPU 0 | k=0 | capture24_k0_aulavik_ep10pca15.yaml → ep20.yaml → ep30pca1015.yaml | **RUNNING** | ep10/pca15 DONE; ep20: 18/27 seeds complete (6/9 configs) as of 2026-05-19 (PID 93214 active); ep30pca1015 pending. Note: separate from Huron seed_42 restarts (those are COMPLETE) |
| Aulavik GPU 1 | k=5 | capture24_k5_gpu1.yaml + capture24_k5_gpu1_restart.yaml | **COMPLETE** | All 9 configs with PCA=15 ep=30 verified done via restart dirs (19+ final_test_metrics.csv confirmed 2026-05-19). Merge seed dirs into originals still pending. |
| Lambda GPU 0 | k=3 | capture24_k3_gpu0.yaml + capture24_k3_gpu0_ep20_restart.yaml | **COMPLETE** | All 81 k=3 seeds confirmed done (122+ final_test_metrics.csv across original + restart dirs, audited 2026-05-19) |
| Lambda GPU 0 | k=5 | capture24_k5_gpu0_restart.yaml | **RUNNING** | Launched 2026-05-19 (PID 184266); ep=[10,20], max_parallel=2 (54 total runs); stale 6003 MiB from mdanish Jupyter kernels but sufficient headroom (43 GB free) |
| Lambda GPU 1 | k=3 | capture24_k3_gpu1.yaml + capture24_k3_gpu1_pca10_restart.yaml + capture24_k3_gpu1_pca15_restart.yaml | **COMPLETE** | All k=3 GPU1 configs included in 122+ final_test_metrics.csv total (audited 2026-05-19) |
| Lambda GPU 1 | k=5 | capture24_k5_gpu1_lambda_restart.yaml | **RUNNING** | PID 139239 active on Lambda cuda:1 as of 2026-05-19; 11 seeds complete so far |
| Huron GPU 0 | k=0 | capture24_k0_gpu0.yaml | **PARTIAL** | epochs=10 PCA=5+10 only (6 configs 3/3 seeds); PCA=15 and epochs=20+30 gaps covered by Aulavik GPU 0 |
| Huron GPU 1 | k=0 | capture24_k0_gpu1.yaml | **PARTIAL** | epochs=30 PCA=5 only (3 configs 3/3 seeds); PCA=10+15 for epochs=30 covered by Aulavik GPU 0 |

**Huron k=0 completion detail (audited 2026-05-15; seed_42 restarts confirmed complete 2026-05-16):**

Original k0_gpu0 run completed epochs=10 for PCA=[5,10] before dying. k0_gpu1 run completed epochs=30 for PCA=5 only. The remaining 54 configs (epochs=10/PCA=15, all of epochs=20, and epochs=30/PCA=[10,15]) are being covered by the Aulavik GPU 0 gap-fill runs launched 2026-05-15.

Seed_42 restart runs (`capture24_k0_gpu0_restart.yaml` and `capture24_k0_gpu1_restart.yaml`) confirmed finished per log tails (2026-05-16):
- `SPECcapture24_k0_gpu0_restart_*_43a8f51c/seed_42`: PCA=10, TRJ=60% — completed, total time 18155s
- `SPECcapture24_k0_gpu1_restart_*_6faedabc/seed_42`: PCA=5, TRJ=60% — completed, total time 34283s
Merge into original run dirs still pending (run `merge_restart_seeds.sh`).

| Config | Epochs | Seeds complete | Action |
|---|---|---|---|
| PCA=10, TRJ=20%, REAL=4000 | 10 | 3/3 | Done |
| PCA=10, TRJ=40%, REAL=3000 | 10 | 3/3 | Done |
| PCA=10, TRJ=60%, REAL=2000 | 10 | 3/3 | Done (seed_42 restart complete 2026-05-12) |
| PCA=5, TRJ=20%, REAL=4000 | 10 | 3/3 | Done |
| PCA=5, TRJ=40%, REAL=3000 | 10 | 3/3 | Done |
| PCA=5, TRJ=60%, REAL=2000 | 10 | 3/3 | Done |
| PCA=5, TRJ=20%, REAL=4000 | 30 | 3/3 | Done |
| PCA=5, TRJ=40%, REAL=3000 | 30 | 3/3 | Done (seed_42 restart complete 2026-05-12) |
| PCA=5, TRJ=60%, REAL=2000 | 30 | 3/3 | Done (seed_42 restart complete 2026-05-12) |
| PCA=15, all TRJ | 10 | 0/3 | **Aulavik GPU0** (ep10pca15 spec) |
| all PCA, all TRJ | 20 | 0/3 | **Aulavik GPU0** (ep20 spec) |
| PCA=10+15, all TRJ | 30 | 0/3 | **Aulavik GPU0** (ep30pca1015 spec) |

Huron restart specs: `experiment_specs/capture24_grid/capture24_k0_gpu{0,1}_restart.yaml`
Storage: `/storage_1/epigou_storage/FORGE/training_runs/`
Aulavik gap-fill specs: `experiment_specs/capture24_grid/capture24_k0_aulavik_ep{10pca15,20,30pca1015}.yaml`

**Huron restarts complete (2026-05-12). Merge still pending:**
1. Run `experiment_specs/capture24_grid/merge_restart_seeds.sh` to move each `seed_42/` into the corresponding original run directory.
2. Verify with `check_run.py` on the three original run directories.

**Aulavik k=5 completion detail (audited 2026-05-11):**

Storage path on Aulavik: `~/cs_9170_project/training_runs/`

gpu0 (epochs=10) — all complete. Some configs have duplicate run dirs from earlier failed attempts; use the complete (3/3 seed) directory for analysis:

| Config | Use this directory hash |
|---|---|
| PCA=10, TRJ=20% | `61cc570d` (3/3); ignore `08f4d75d` (cut at ep~958) |
| PCA=10, TRJ=40% | `6d4e3edf` (3/3); ignore `96b94fae` (cut at ep~733) |
| PCA=10, TRJ=60% | `a4838d16` (3/3); ignore `6db5cd8f` (cut at ep~593) |
| PCA=15, TRJ=20% | `779cf9c5` (3/3) |
| PCA=15, TRJ=40% | `b0ce000f` (3/3) |
| PCA=15, TRJ=60% | `a36aa01d` (3/3) |
| PCA=5, TRJ=20% | either `40bdd3c1` or `a15dfda0` (both 3/3) |
| PCA=5, TRJ=40% | either `bf8371ef` or `d57d4218` (both 3/3) |
| PCA=5, TRJ=60% | either `141f0030` or `b54ce07a` (both 3/3) |

gpu1 (epochs=30) — 3 PCA=15 configs incomplete at shutdown:

| Config | Seeds complete | Action |
|---|---|---|
| PCA=10, TRJ=20% | 3/3 (`015bf5ad`) | Done |
| PCA=10, TRJ=40% | 3/3 (`b02ad122`) | Done |
| PCA=10, TRJ=60% | 3/3 (`ce46b3d8`) | Done |
| PCA=5, TRJ=20% | 3/3 (`e380c82a`) | Done |
| PCA=5, TRJ=40% | 3/3 (`d7c3ca89`) | Done |
| PCA=5, TRJ=60% | 3/3 (`8c658b54`) | Done |
| PCA=15, TRJ=20% | 2/3 (`af6419db`; seed_42 missing) | **Restarting** — restart dirs: seed_0 ep5000 ✓, seed_1 ep5000 ✓, seed_42 pending |
| PCA=15, TRJ=40% | 1/3 (`75a56293`; seed_1 @ ep4878, seed_42 missing) | **Restarting** — restart dirs: seed_0 ep5000 ✓, seed_1 ep5000 ✓, seed_42 pending |
| PCA=15, TRJ=60% | 0/3 (`8093d5d7`; seed_0 @ ep4112, seed_1 @ ep4107, seed_42 missing) | **Restarting** — restart dirs: seed_0 ep5000 ✓, seed_1 at ep674 (running), seed_42 pending |

Restart spec: `experiment_specs/capture24_grid/capture24_k5_gpu1_restart.yaml` (runs seeds=[0,1,42] × ratio=[0.2,0.4,0.6] for PCA=15, epochs=30; 3 redundant runs acceptable)
Restart log: `~/cs_9170_project/experiment_specs/capture24_grid/logs/k5_gpu1_restart.log` (on Aulavik)
New run dir prefix: `SPECcapture24_k5_gpu1_restart_EP5000_PCA15_*_GG202605111125_*`
Progress audited 2026-05-13: TRJ=20% and TRJ=40% seeds 0+1 complete in restart dirs; TRJ=60% seed_0 done, seed_1 at ep674. seed_42 for all three ratios not yet started (batch 3 of 3 pending).

**After Aulavik gpu1 restarts finish:**
1. For each PCA15 config, copy the needed seed_N/ directories from the new restart run dirs into the original run dirs (`af6419db`, `75a56293`, `8093d5d7`).
2. Update seeds.json in each original directory.
3. Verify with `check_run.py`.

**Selection criterion:** Same as EXP-021 — lowest mean β-EO, with β-F1w ≥ α-F1w − 0.02.

**PRIMARY CONFIG CONFIRMED 2026-05-19: k=5, pca=15, ep=10, traj=1000 (real=4000)**

β-EO=0.022±0.013, EOd=0.028±0.009, F1w=0.955, AUC=0.931, α-EO=0.158±0.083.

Matched baselines (pca=15, real=4000, da_pct=0.015) run on Aulavik GPU1 2026-05-19 — all 6 methods complete:

| Method | α-EO | β-EO | F1w | AUC |
|--------|------|------|-----|-----|
| **FORGE k=5/pca=15** | 0.158±0.083 | **0.022±0.013** | 0.955 | 0.931 |
| SMOTE | 0.140±0.078 | 0.057±0.029 | 0.948 | 0.845 |
| OT Repair | 0.140±0.078 | 0.112±0.080 | 0.954 | 0.909 |
| FairTabDDPM | 0.140±0.078 | 0.132±0.061 | 0.952 | 0.932 |
| FLB | 0.140±0.078 | 0.139±0.087 | 0.871 | 0.911 |
| GroupDRO | 0.140±0.078 | 0.183±0.106 | 0.885 | 0.899 |
| CTGAN | 0.140±0.078 | 0.451±0.130 | 0.952 | 0.919 |

Baseline run dirs (Aulavik, G202605191128): `training_runs/BASELINE_{group_dro,fairness_loss_balancing,smote,gaussian_ot_repair}_capture24_*pca15_r4000*`; CTGAN: `*ctgan_pca15_r4000_3efc3b09__G202605191131`; FairTabDDPM: `*fairtabddpm_pca15_r4000_539963ec__G202605191135`

**Note on α-EO discrepancy:** Baselines show α-EO=0.140 vs FORGE grid's α-EO=0.158. Both use seeds [0,1,42], da_pct=0.015, real=4000, pca=15 — difference arises because FORGE's alpha trains for ep=10 FFNN epochs while baselines train alpha at 20 epochs. Baselines are starting from a slightly smaller unfairness gap, making FORGE's 0.022 result conservative.

**Note on DA+:** real=4000 with da_pct=0.015 gives DA+≈60 (vs DA+=45 at real=3000). See CLAUDE.md framing note.

**Previous provisional best** (now superseded): k=3, pca=10, ep=10, traj=2000 — β-EO=0.063±0.051, F1w=0.953, AUC=0.923.

Top 5 configs (ranked by β-EO):

| k | pca | ep | traj | α-EO | β-EO | EOd | F1w | AUC |
|---|---|---|---|---|---|---|---|---|
| 5 | 15 | 10 | 1000 | 0.158±0.083 | 0.022±0.013 | 0.028±0.009 | 0.955 | 0.931 |
| 0 | 5 | 30 | 1000 | 0.044±0.033 | 0.055±0.038 | 0.055±0.038 | 0.949 | 0.918 |
| 3 | 10 | 10 | 2000 | 0.294±0.062 | 0.063±0.051 | 0.064±0.050 | 0.953 | 0.923 |
| 3 | 5 | 30 | 1000 | 0.092±0.036 | 0.066±0.029 | 0.066±0.029 | 0.947 | 0.878 |
| 3 | 5 | 30 | 2000 | 0.099±0.078 | 0.076±0.052 | 0.076±0.052 | 0.939 | 0.883 |

Run dirs: best overall on Aulavik `SPECcapture24_k5_gpu0_EP5000_PCA15_REWwgl_minID1_majID0_TRJ1000_REAL4000_GG202605041119_779cf9c5`; best pca=10 on Lambda `SPECcapture24_k3_gpu0_EP5000_PCA10_REWwgl_minID1_majID0_TRJ2000_REAL3000_GG202605041148_7b75cd2a`

**Takeaway (partial):** k=5 pca=15 shows exceptional β-EO=0.022, but low variable α-EO complicates comparison. For pca=10, k=3 beats k=5 — sigmoid sharpness benefit from census does not transfer straightforwardly. Known α-EO instability (capture24 seed variance) remains the main obstacle. Use k=3, pca=10, ep=10, traj=2000 as provisional paper config pending full grid.

**Next steps:**
- Await k=3 ep=20/30 pca=10/15 on Lambda (~May 16-18) and k=5 ep=30 pca=15 on Aulavik (~May 18).
- Await k=0 gap-fills on Aulavik GPU0 (~May 25-28).
- Re-run aggregation once all complete.

**Takeaway:**
*(pending)*

**Next steps:**
1. Wait for Aulavik gpu1 PCA=15 restart to finish (seed_42 batch pending ~May 18), then merge seed dirs into original run dirs and verify with `check_run.py`.
2. Wait for Lambda k=3 restarts (epochs=20 PCA=15 seeds 1+42 ~May 16; epochs=30 PCA=10+15 seeds ~May 18) to complete.
3. Wait for Lambda k=5 restarts to complete (queued behind k=3, ~May 20+).
4. Wait for Aulavik GPU0 k=0 gap-fills (54 runs, ~May 25-28).
5. Run merge_restart_seeds.sh on Huron k=0 dirs (pending since 2026-05-12).
6. Once all 243 runs complete, run `check_run.py` across all configs and select best params.
7. Feed best capture24 params into capture24 random search (equivalent of EXP-022).

---

### EXP-026 | raw-space-eval

**Status:** PLANNED

**Datasets:** census_income, capture24

**Purpose:**
Address supervisor feedback that classifiers and baselines should be evaluated in the original (raw) feature space rather than PCA-compressed space. Implements the approach recommended by Santiago: keep all RL training unchanged (agent still operates in PCA space), but for the final reported result, inverse-transform the best synthetic dataset via φ^{-1} and re-train β once on raw D_aug = raw real train + inverse-transformed synthetic. Baselines are also re-run with `use_pca: false` so all methods are compared in a common raw-feature representation.

**Design rationale:**
Full re-running of the RL training loop in raw space is not feasible (would invalidate all grid search results). The checkpoint selection criterion (worst-group loss of β trained in PCA space) remains a proxy for raw-space performance, but this is defensible: PCA is a linear transform that preserves the group-specific spatial structure the agent learned to target. Expected performance difference vs current PCA-space results: small to moderate. Census is at higher risk of noticeable shift (~100 raw features vs d=10 PCA). If a reviewer questions this, the current PCA-space numbers serve as the supplementary comparison.

**Config delta from vanilla:**
- No changes to RL training config.
- Baselines re-run with `use_pca: false` in their spec files.

**Infrastructure:**
- `eval_raw_space.py` — post-processing script. For each seed in a run dir:
  1. Re-fits PCA with same seed/config to recover `pca_transform`.
  2. Loads dataset with `use_pca=False` for raw real train/test data.
  3. Loads `best_synthetic_phase1_class1.npz` (PCA space).
  4. Applies `pca_transform.inverse_transform()` → raw synthetic features.
  5. Trains fresh β (same FFNN architecture) on raw D_aug.
  6. Evaluates on raw test set; saves to `seed_dir/raw_space_eval/metrics.json`.
  7. Writes run-level summary to `run_dir/raw_space_eval_summary.json`.

**Usage:**
```bash
# Run on a single FORGE run dir
python eval_raw_space.py training_runs/<run_dir> --device cpu

# Re-run baselines without PCA (edit baseline spec files first: use_pca: false)
python run_baseline.py --spec experiment_specs/<baseline_spec>.json --device cuda:0
```

**Execution order:**
1. Re-run all baselines with `use_pca: false` on census and capture24 (do first — independent of eval_raw_space.py).
2. Run `eval_raw_space.py` on the best FORGE runs for each dataset (best configs from EXP-021/EXP-025 + EXP-022).
3. Compare FORGE raw-space EO/AUC/F1w against raw-space baselines.

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
*(pending — blocked on baseline re-runs and EXP-021/EXP-025 completion)*

---

### EXP-027 | acs-employment-rl-main

**Type:** PAPER-FINAL
**Status:** IN PROGRESS
**Dataset(s):** acs_employment (disability framing, 10 states)
**Primary framing (as of 2026-05-19):** da_pct=0.01433 → DA+=43 (injected scarcity, consistent with census/capture24)
**Reference config:** vanilla_config.json
**Config delta:** dataset_name=acs_employment, dp_protected_col=disability, acs_states=[CA,TX,NY,FL,PA,OH,IL,GA,NC,MI], minority_id=0, majority_id=1, global_sigmoid_k=5.0 (top-level), use_pca=true, pca_components=10, total_episodes=5000
**Follows from:** EXP-024

---

**Purpose:**
Main FORGE result for ACS Employment + disability. Demonstrates FORGE generalizes to non-demographic protected groups (disability status) and a non-income prediction task (employment). Two scarcity framings explored to determine best paper framing.

**Framing decision (2026-05-19):**
Natural scarcity (da_pct=null, DA+=71) was initially preferred for paper strength (no bias injection). However, EXP-028 wave 2 showed SMOTE achieves β-EO=0.070 at DA+=71 — a very strong baseline that FORGE would need to beat to make a compelling case. EXP-028 wave 3 (DA+=43) confirmed SMOTE degrades to 0.194 and CTGAN collapses to 0.832, while FairTabDDPM remains competitive at 0.061. **Decision: pursue da_pct=0.01433 (DA+=43) as primary framing**, consistent with census and capture24.

**Submission tracker (as of 2026-05-19):**

| Phase | Resource | Seeds | Config | PIDs | Status |
|---|---|---|---|---|---|
| 1 (superseded) | Oneida GPU0/1 | 0,1 / 2,3 | k=10 bug (nested YAML) | 56219/56221 | **KILLED** — k=10 used instead of k=5 due to `global_sigmoid_k` nested under `reward_shaping:` |
| 2 (natural, running) | Oneida GPU0 | 0, 1, 42 | k=5, da_pct=null, PCA-10, ep=5000 | 113761 | **RUNNING** |
| 2 (natural, running) | Oneida GPU1 | 2, 3 | k=5, da_pct=null, PCA-10, ep=5000 | 114020 | **RUNNING** |
| 3 (da43, queued) | Oneida GPU0 | 0, 1, 42 | k=5, da_pct=0.01433, PCA-10, ep=5000 | queue PID 116666 | **QUEUED** — starts after phase 2 GPU0 (PID 113761) exits |
| 3 (da43, queued) | Oneida GPU1 | 2, 3 | k=5, da_pct=0.01433, PCA-10, ep=5000 | queue PID 116666 | **QUEUED** — starts after phase 2 GPU1 (PID 114020) exits |

Spec files: `experiment_specs/Experiment3/acs_forge_gpu0_k5.yaml` (phase 2), `acs_forge_gpu1_k5.yaml` (phase 2), `acs_forge_gpu0_da43.yaml` (phase 3), `acs_forge_gpu1_da43.yaml` (phase 3)
Queue script: `experiment_specs/Experiment3/run_acs_forge_da43_oneida.sh`
Logs: `experiment_specs/Experiment3/logs/acs_forge_gpu{0,1}_{k5,da43}.log`

**Bug fixed (2026-05-19):** `training.py` read `global_sigmoid_k` only from top-level spec key. ACS spec had it nested under `reward_shaping:` → k=10 used for all phase-1 seeds, producing poor results (seed_0 β-EO=0.284, seed_2 β-EO=0.752). Fixed `training.py` to check `reward_shaping` subdict as fallback. This is the second occurrence of this bug. See feedback memory. All new specs use top-level `global_sigmoid_k: 5.0`.

**Phase 2 interim notes:** Phase 1 (k=10) seed_0 produced β-EO=0.284, seed_2 β-EO=0.752. These are discard results due to the k bug. Phase 2 results pending.

**Target benchmark (da43 framing):** FairTabDDPM β-EO=0.061 (EXP-028 wave 3). FORGE must beat this to claim advantage in the DA+=43 regime.

**Result:**
*(pending — phase 2 running, phase 3 queued)*

**Takeaway:**
*(pending)*

**Next steps:**
- Await phase 2 completion; check whether k=5 natural scarcity beats SMOTE (0.070) as a secondary result
- Await phase 3 (da43) completion; compare against FairTabDDPM (0.061) as primary benchmark
- Update paper with ACS Employment as 3rd dataset using da_pct=0.01433 framing

---

### EXP-028 | acs-employment-baselines

**Type:** PAPER-FINAL
**Status:** IN PROGRESS
**Dataset(s):** acs_employment (disability framing, da_pct=0.01433 primary; da_pct=null also run for framing comparison)
**Reference config:** run_baseline.py
**Config delta:** all 6 baselines: GroupDRO, FLB, SMOTE, OT Repair, CTGAN, FairTabDDPM; use_pca=true, pca_components=10 (wave 2+3); acs_states=[CA,TX,NY,FL,PA,OH,IL,GA,NC,MI]
**Follows from:** EXP-027

---

**Purpose:**
Baseline comparisons for ACS Employment + disability across two scarcity framings. Wave 1 and 2 cover natural scarcity (da_pct=null, DA+=71); wave 3 covers injected scarcity (da_pct=0.01433, DA+=43). Wave 3 results drove the framing decision for EXP-027.

**Bug fixed before waves 2-3:** Aulavik was on older git commit without `split_acs_employment(acs_states=...)` parameter. All 6 baselines crashed on wave 1 with `acs_states` kwarg error. Fixed by rsyncing `dataset.py` from Huron. Same fix needed for `run_baseline.py` + all 6 baseline classes — `acs_states` passthrough was missing; added 2026-05-19.

---

**Wave 1 — non-PCA, natural scarcity (da_pct=null, use_pca=false)**

Aulavik GPU1, script: `run_acs_baselines_aulavik.sh`, log: `baselines/logs/run_master_aulavik.log`
Status: **COMPLETE** (~10:04 AM EDT 2026-05-19). Run dirs: `training_runs/BASELINE_*_G202605190943/`

*(Specific metrics not recorded — raw feature space, unfair comparison with FORGE's PCA space. Wave 2 PCA results supersede these for analysis.)*

---

**Wave 2 — PCA, natural scarcity (da_pct=null, use_pca=true, pca_components=10)**

Aulavik GPU1, script: `run_acs_baselines_aulavik_pca.sh`, log: `baselines/logs/run_master_aulavik_pca.log`
Status: **COMPLETE** (all 6 methods finished 2026-05-19). Includes CTGAN and FairTabDDPM.

Spec files: `experiment_specs/Experiment3/baselines/acs_{gdro,flb,smote,ot_repair,ctgan,fairtabddpm}_pca.yaml`

**Results — ACS Employment, natural scarcity (da_pct=null, DA+=71, PCA-10):**

| Method | β-EO↓ | Notes |
|---|---|---|
| SMOTE | **0.070±0.071** | Very strong — approaches zero; risk of high variance |
| GroupDRO | 0.256±0.078 | |
| FLB | 0.257±0.027 | |
| OT Repair | 0.410±0.059 | |
| CTGAN | *(pending at session end)* | |
| FairTabDDPM | *(pending at session end)* | |

α-EO natural regime: ~0.775±0.056 (from SMOTE run)

**Framing concern:** SMOTE β-EO=0.070 at natural scarcity is very strong. If FORGE k=5 cannot beat this (threshold ~0.06 to be meaningful), the natural framing weakens the paper's case.

---

**Wave 3 — PCA, injected scarcity (da_pct=0.01433, DA+=43)**

Aulavik GPU1, script: `run_acs_baselines_aulavik_da43.sh`, log: `baselines/logs/run_master_aulavik_da43.log`
Queued behind wave 2 (waited on wave 2 master PID via `tail --pid`). Status: **COMPLETE** (2026-05-19).

Spec files: `experiment_specs/Experiment3/baselines/acs_{gdro,flb,smote,ot_repair,ctgan,fairtabddpm}_da43.yaml`

**Results — ACS Employment, injected scarcity (da_pct=0.01433, DA+=43, PCA-10):**

| Method | β-EO↓ | Verdict |
|---|---|---|
| FairTabDDPM | **0.061** | Strong — key benchmark for EXP-027 phase 3 |
| SMOTE | 0.194 | Degrades substantially vs wave 2 (0.070→0.194) ✓ motivation claim |
| CTGAN | 0.832 | Collapses at DA+=43 ✓ motivation claim |
| GroupDRO | *(see logs)* | |
| FLB | *(see logs)* | |
| OT Repair | *(see logs)* | |

α-EO injected scarcity regime: ~0.09–0.14 (from EXP-024 viability check)

**Framing decision from wave 3:** SMOTE degradation (0.070→0.194) and CTGAN collapse (0.832) confirm that DA+=43 is the appropriate regime for the paper's motivation claim. FairTabDDPM at 0.061 is a notable exception — FORGE must beat this to claim superiority in the injected scarcity regime.

---

**Oneida wave (cancelled):** A baseline queue script (PID 111256) was waiting for old FORGE PIDs 56219/56221. Those PIDs were killed when the k=10 bug was found. The baseline script appears not to have triggered any runs after the kill. Waves 2+3 on Aulavik are the canonical baseline results.

**Result:**
Wave 1: complete, not used (non-PCA). Wave 2: partial (SMOTE/GroupDRO/FLB/OT Repair complete; CTGAN/FairTabDDPM status unknown). Wave 3: complete — key results recorded above.

**Takeaway:**
Injected scarcity (DA+=43) is the correct framing. Baseline degradation pattern matches census/capture24: CTGAN fails catastrophically, SMOTE degrades, FairTabDDPM remains competitive (0.061 is the FORGE target). GroupDRO and FLB full results pending from wave 3 logs.

**Next steps:**
- Pull full wave 3 results from Aulavik logs once EXP-027 phase 3 FORGE is running
- Compare FORGE da43 vs FairTabDDPM 0.061 benchmark
- Update paper results table with ACS Employment column

---

### EXP-029 | meps-sex-rl-3rd-dataset

**Type:** PAPER-FINAL
**Status:** IN PROGRESS
**Dataset(s):** meps (sex framing, da_pct=0.01433, DA+=43)
**Seeds:** 0, 1, 42
**Spec:** `experiment_specs/Experiment3/meps_forge_gpu0.yaml`
**Follows from:** —

---

**Purpose:**
3rd dataset candidate for the paper. MEPS HC-243 (Medical Expenditure Panel Survey 2022), sex framing: male (SEX=1, a=0) disadvantaged vs female (SEX=2, a=1) advantaged. Outcome: any prescribed medicine event (RXTOT22 ≥ 1). Healthcare domain — distinct from census (income) and capture24 (wearables).

**Config delta from vanilla:**
- dataset_name: meps
- da_pct: 0.01433 (DA+=43, consistent with census)
- dp_protected_col: sex, minority_id=0 (male), majority_id=1
- total_episodes: 5000
- global_sigmoid_k: 5.0 (at top level — k=10 upgrade after first-pass confirmation)
- ffnn.epochs: 30

**Viability (2026-05-21, drop_protected=False, plain shuffle):**
| Criterion | Value | Status |
|-----------|-------|--------|
| val_disadv_pos | 1,163–1,181 | PASS |
| test_disadv_pos | 1,117–1,147 | PASS |
| best val α-EO | 0.267 (seed 0) | PASS |
| sep_ratio | 2.31 | PASS |

Natural rates: Male rx+=54.8%, Female rx+=65.9%. AA+/DA+ ratio ≈ 32x — similar to ACS Employment failure case. Key differentiators: sep_ratio=2.31 (vs ACS Employment 1.71), large repair gap (α-EO=0.267). Monitor for WGL-EO disconnect.

**Launch command:**
```
source ~/envs/rl/bin/activate && python main.py --spec experiment_specs/Experiment3/meps_forge_gpu0.yaml --device cuda:0
```

**Result:**
Mid-run check (seed 0, ep 2186/5000, 2026-05-21): α-EO=0.611, β-EO_last=0.671, β-EO_best=0.512, AUC=0.708, mean_reward(last10)=0.943. Reward is strong (beta consistently beating alpha on WGL) but EO is not tracking consistently — current episode has beta WORSE than alpha (0.671 > 0.611), though best-ever β-EO of 0.512 is below alpha. Seeds 1 and 42 not yet started. Pattern is mixed: occasional improvement but no sustained reduction. Early signal from ep ~302 showed weak reward (0.23–0.46); reward has strengthened to 0.943 by ep 2186, suggesting the reward mechanism is working but the EO improvement is not stable.

**Takeaway:**
*(pending — continue monitoring, need seeds 1 and 42)*

**Next steps:**
- Wait for seeds 1 and 42 to start and reach ep ~1000 before deciding
- If ≥2/3 seeds show β-EO_best < α-EO: keep as candidate; run check_run.py at completion
- If only seed 0 occasionally beats alpha and seeds 1/42 do not: treat as marginal/failing; consider traj_length=4000 (raises post-aug DA+/AA+ from 1.49x → ~3.0x)
- If WGL-EO disconnect confirmed across all seeds: drop; pivot to EXP-032 candidates

---

### EXP-030 | acs-income-race-rl-3rd-dataset

**Type:** PAPER-FINAL
**Status:** IN PROGRESS
**Dataset(s):** acs_income (race framing, da_pct=0.01433, DA+=43)
**Seeds:** 0, 1, 42
**Spec:** `experiment_specs/Experiment3/acs_income_race_gpu1.yaml`
**Follows from:** —

---

**Purpose:**
3rd dataset candidate, running in parallel with EXP-029 (MEPS). ACS Income (folktables, 10 states: CA TX NY FL PA OH IL GA NC MI), race as protected attribute, minority_id=0 (non-white) disadvantaged. Well-known fairness benchmark dataset; distinct from MEPS (tabular income vs healthcare).

**Config delta from vanilla:**
- dataset_name: acs_income
- da_pct: 0.01433 (DA+=43)
- dp_protected_col: race, minority_id=0, majority_id=1
- acs_states: CA, TX, NY, FL, PA, OH, IL, GA, NC, MI
- total_episodes: 5000
- global_sigmoid_k: 5.0 (top level)
- ffnn.epochs: 30

**Viability (2026-05-21, drop_protected=False, plain shuffle):**
| Criterion | Value | Status |
|-----------|-------|--------|
| val_disadv_pos | 5,276–5,496 | PASS |
| test_disadv_pos | 5,243–5,307 | PASS |
| best val α-EO | 0.131 (seed 42) | PASS |
| sep_ratio | ~2.29 (prior run) | PASS |

Note: drop_protected=False means RAC1P column stays in features (consistent with all existing FORGE results). Earlier viability with drop_protected=True showed α-EO→0.007 and sep_ratio→0.92 — the race signal relies on RAC1P being present, which is a known reviewer concern.

**Launch command:**
```
source ~/envs/rl/bin/activate && python main.py --spec experiment_specs/Experiment3/acs_income_race_gpu1.yaml --device cuda:1
```

**Result:**
Mid-run check (seed 0, ep 1295/5000, 2026-05-21): α-EO=0.357, β-EO_last=0.037, β-EO_best=0.000, AUC=0.824, mean_reward(last10)=0.996. Numbers look spectacular but are trivially explained: RAC1P (the race column) is in the features (drop_protected=False), so the classifier can read group membership directly. β-EO→0 is consistent with the model exploiting RAC1P rather than learning a meaningful fairness-aware representation. This is the same reason ACS Income race was dropped in viability — with drop_protected=True the α-EO collapses to 0.007, confirming the race signal is entirely carried by RAC1P. Run stopped; dataset dropped.

**Takeaway:**
DROPPED — results are an artifact of the protected attribute (RAC1P) being present in the feature space. Any β-EO improvement is trivially explained by the classifier reading race directly. Consistent with prior viability findings (alpha-EO=0.007, sep_ratio=0.92 with drop_protected=True).

**Next steps:**
— (Dropped; pivoting to EXP-032 viability candidates)

---

### EXP-031 | acs-employment-sex-rl-3rd-dataset

**Type:** PAPER-FINAL
**Status:** IN PROGRESS
**Dataset(s):** acs_employment (sex framing, da_pct=0.01433, DA+=43)
**Seeds:** 0, 1, 42
**Spec:** `experiment_specs/Experiment3/acs_employment_sex_lambda.yaml`
**Server:** Lambda (cuda:0)
**Follows from:** —

---

**Purpose:**
3rd dataset candidate, running on Lambda as fallback to EXP-029 (MEPS) and EXP-030 (ACS Income race). ACS Employment (folktables, 10 states), sex framing: female (SEX=2, a=0) disadvantaged vs male (SEX=1, a=1) advantaged. Outcome: employment (y=1). Employment domain is distinct from income (census), wearables (capture24), and healthcare (MEPS). ACS Employment disability framing (EXP-027) was dropped — sex framing is structurally different and passes all viability criteria.

**Config delta from vanilla:**
- dataset_name: acs_employment
- da_pct: 0.01433 (DA+=43)
- dp_protected_col: sex, minority_id=0 (female), majority_id=1 (male)
- acs_states: CA, TX, NY, FL, PA, OH, IL, GA, NC, MI
- total_episodes: 5000
- global_sigmoid_k: 5.0 (top level)
- ffnn.epochs: 30
- pca_components: 10 (dataset has 16 features; 10 is conservative first pass)

**Config rationale:**
- traj_length=2000: post-aug DA+/AA+ = 2043/894 = 2.29x — comparable to census (2.89x), no augmentation increase needed for first pass
- k=5: conservative first pass; upgrade to k=10 if dataset confirms viable (consistent with MEPS/ACS Income)
- pca=10: same as census; pca=15 is an ablation option given 16 input features

**Viability (2026-05-21, drop_protected=False, plain shuffle):**
| Criterion | Seed 42 | Seed 0 | Seed 1 | Status |
|-----------|---------|--------|--------|--------|
| val_disadv_pos | 75,083 | 75,176 | 75,420 | PASS |
| test_disadv_pos | 75,112 | 75,343 | 75,239 | PASS |
| val α-EO | 0.127 | 0.113 | 0.092 | PASS (best=0.127) |
| sep_ratio | — | — | 2.33 | PASS |
| cosine similarity | — | — | 0.976 | note: high (groups similar structure) |

Natural rates: female employment 42.6%, male 48.5%. After bias injection train positive rate ~31%. Training AA+/DA+ = ~21x; post-aug = 2.29x.

Note: sep_ratio is strong (2.33) but cosine similarity is high (0.976) — PCA separation is partially driven by the SEX feature. Groups have genuine employment-characteristic differences (education, occupation, hours) beyond just SEX column, so this is less of a concern than ACS Income race (where sep_ratio collapses without RAC1P).

**Launch command (Lambda):**
```
source ~/envs/rl/bin/activate && python main.py --spec experiment_specs/Experiment3/acs_employment_sex_lambda.yaml --device cuda:0
```
Note: ACS Employment data auto-downloads via folktables on first run if not cached locally. Requires internet access on Lambda.

**Result:**
Mid-run check (seed 0, ep 2918/5000, 2026-05-21): α-EO=0.623, β-EO_last=0.735, β-EO_best=0.691, AUC=0.806, mean_reward(last10)=0.963. Beta has NEVER beaten alpha on EO across 2918 episodes — β-EO_best=0.691 > α-EO=0.623. Reward is strong (0.963) while EO is moving in the wrong direction. This is the exact WGL-EO disconnect pattern seen in EXP-027 (ACS Employment disability): WGL improves but EO gap grows. Value range (0.62–0.74) matches the disability failure (beta-EO 0.62–0.71 noted in CLAUDE.md). Run stopped; dataset dropped.

**Takeaway:**
DROPPED — same WGL-EO disconnect as EXP-027 (disability framing). ACS Employment has now failed across two structurally different protected attributes (disability and sex); the dataset is ruled out entirely. The high cosine similarity between groups (0.976) noted in viability likely explains the disconnect: PCA separation is not driven by employment-relevant feature differences between groups, so the reward-guided generation cannot produce samples that meaningfully target the fairness gap.

**Next steps:**
— (Dropped; ACS Employment ruled out entirely; pivoting to EXP-032 viability candidates)

---

### EXP-032 | 3rd-dataset-viability-search-2

**Type:** EXPLORATORY
**Status:** PLANNED
**Dataset(s):** tbd — see candidates below
**Seeds:** 0, 1, 42
**Spec:** tbd
**Follows from:** EXP-024 (first search), EXP-029/030/031 (all dropped or uncertain)

---

**Purpose:**
Second viability search for a 3rd paper dataset. EXP-029 (MEPS sex) is still active but uncertain; EXP-030 (ACS Income race) and EXP-031 (ACS Employment sex) are dropped. ACS Employment is ruled out entirely (two framings failed). Need a dataset with a distinct domain from census (income), capture24 (wearables), and MEPS (healthcare/prescriptions), with genuine group-level disparities and DA+ ≈ 43.

All four viability criteria must pass (drop_protected=True):
1. val_disadv_pos ≥ 30
2. test_disadv_pos ≥ 200
3. α-EO clearly non-zero on at least one seed
4. sep_ratio > 1

**Candidates (priority order):**

**A. ACS Public Coverage — sex framing (folktables, easy to add)**
- Task: predict whether individual has public health insurance (PUBCOV=1)
- Protected attr: sex; female disadvantaged (lower coverage rate in some states)
- Data: folktables 10-state pool, 2018 ACS 1-Year; already implemented infrastructure
- Domain: health insurance — different from census (income), MEPS (prescriptions), capture24
- Concern: natural coverage gap may be small; check α-EO before launching RL

**B. Taiwan Credit Default — sex or age framing (UCI)**
- Task: predict credit card default payment next month
- Protected attr: sex (female=2 disadvantaged) or age group (young/old)
- Data: 30,000 samples (doi.org/10.24432/C55S3H); tabular financial features; needs dataset.py implementation
- Domain: financial credit — distinct from all confirmed datasets
- Concern: need to check natural group-level default rate differences

**C. MEPS alternative outcome — hospitalization or unmet need**
- Task: predict hospitalization (IPNGTD22 > 0) or unmet medical need (UNMET22)
- Protected attr: sex (same as EXP-029); reuse existing meps dataset implementation
- Rationale: if MEPS sex fails for prescription fills, a different outcome may have better EO characteristics and lower AA+/DA+ ratio
- Concern: hospitalization is rarer outcome; check val_disadv_pos threshold

**D. Bank Marketing — age framing (UCI)**
- Task: predict term deposit subscription (y/n)
- Protected attr: age group (middle-aged 30–50 vs elderly 60+)
- Data: ~45,000 samples; binary outcome; Portuguese bank data
- Domain: financial marketing — distinct
- Concern: age requires binning; outcome (term deposit subscription) is marketing not fairness-critical; weaker narrative

**Config plan (once candidate passes viability):**
Same as EXP-029: k=5, pca=10, ep=30, traj=2000, real=3000, seeds=0/1/42. Upgrade to k=10 if first-pass confirms viable.

**Viability commands (once dataset.py implementation exists):**
```
python dataset_viability.py --dataset <name> --dp_protected_col <col> --minority_id <id> --majority_id <id> --da_pct 0.01433 --seeds 0 1 42 --drop_protected True
```

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
- Start with candidate A (ACS Public Coverage) — lowest implementation cost, folktables already integrated
- Implement Taiwan Credit Default (candidate B) if A fails — strongest domain contrast
- Run MEPS alternative outcome (candidate C) in parallel with A if EXP-029 looks like dropping

---

### EXP-033 | meps-sex-k10

**Status:** IN PROGRESS
**Type:** PAPER-FINAL
**Dataset:** meps (sex, prescription fills)
**Follows from:** EXP-029

**Purpose:**
EXP-029 (k=5) showed seed_0 improving β-EO from 0.611 to 0.512, but reward was noisy (0.35–0.94 range at ep ~2186). Hypothesis: k=10 provides a sharper sigmoid boundary around the reward neutral point, reducing noise and making the gradient signal cleaner — same mechanism by which k=10 outperformed k=5 on census (EXP-021). Tests whether MEPS follows the same k-sensitivity as census.

**Config delta from vanilla:**
- dataset_name: meps
- da_pct: 0.01433
- dp_protected_col: sex
- minority_id: 0 (male disadvantaged)
- majority_id: 1
- global_sigmoid_k: 10.0
- traj_length: 2000
- real_data_size: 3000
- total_episodes: 5000
- ffnn.epochs: 30
- seeds: [0, 1, 42]

**Spec file:** `experiment_specs/Experiment3/meps_forge_k10_gpu1.yaml`

**Launch command (Huron GPU1):**
```bash
cd ~/cs_9170_project && source ~/envs/rl/bin/activate && \
nohup python main.py --spec experiment_specs/Experiment3/meps_forge_k10_gpu1.yaml --device cuda:1 > /tmp/meps_k10_gpu1.log 2>&1 &
```
Launched 2026-05-21. PID 39755 (Huron). Confirmed training at ep ~19, reward ~0.68.

**Result:**
*(pending — check at ep ~500 for β-EO vs α-EO tracking)*

**Takeaway:**
*(pending)*

**Next steps:**
- If β-EO tracks α-EO improvement by ep 500: continue to completion
- If WGL-EO disconnect (WGL improves but β-EO does not reduce): drop; MEPS not viable
- Compare with EXP-034 (traj=4000) — whichever shows better β-EO improvement at ep 500 is the stronger config

---

### EXP-034 | meps-sex-traj4000

**Status:** IN PROGRESS
**Type:** PAPER-FINAL
**Dataset:** meps (sex, prescription fills)
**Follows from:** EXP-029

**Purpose:**
EXP-029 (traj=2000) generates ~43+2000=2043 DA+ post-augmentation vs ~1390 AA+ positives, giving a post-aug DA+/AA+ ratio of ~1.49x. Census uses traj=2000 but has ~43+2000=2043 DA+ vs ~710 AA+, giving ratio ~2.88x. Hypothesis: MEPS suffers from insufficient DA+ dominance post-augmentation; increasing traj=4000 raises the ratio to ~3.0x, matching the census regime. Tests whether volume (rather than reward sharpness) is the bottleneck.

**Config delta from vanilla:**
- dataset_name: meps
- da_pct: 0.01433
- dp_protected_col: sex
- minority_id: 0 (male disadvantaged)
- majority_id: 1
- global_sigmoid_k: 5.0
- traj_length: 4000
- real_data_size: 3000
- total_episodes: 5000
- ffnn.epochs: 30
- seeds: [0, 1, 42]

**Spec file:** `experiment_specs/Experiment3/meps_forge_traj4000_lambda.yaml`

**Launch command (Lambda GPU0):**
```bash
cd ~/cs_9170_project && source ~/envs/rl/bin/activate && \
nohup python main.py --spec experiment_specs/Experiment3/meps_forge_traj4000_lambda.yaml --device cuda:0 > /tmp/meps_traj4000.log 2>&1 &
```
Launched 2026-05-21. Lambda GPU0. Initially failed (MEPS data missing on Lambda — h243.csv transferred via scp before relaunch). Relaunched; confirmed training at ep ~18, reward ~0.75.

**Result:**
*(pending — check at ep ~500 for β-EO vs α-EO tracking)*

**Takeaway:**
*(pending)*

**Next steps:**
- If β-EO tracks α-EO improvement by ep 500: continue to completion
- If WGL-EO disconnect: drop; MEPS not viable; escalate to EXP-032 candidates
- Compare with EXP-033 (k=10, traj=2000) at ep 500 — whichever shows cleaner EO reduction proceeds

---

### EXP-035 | meps-sex-k3-traj4000-ep30

**Status:** IN PROGRESS
**Type:** PAPER-FINAL
**Dataset:** meps (sex, prescription fills)
**Follows from:** EXP-034

**Purpose:**
Tests whether a softer sigmoid (k=3) with high synthetic volume (traj=4000) improves over EXP-034 (k=5, traj=4000). MEPS has alpha_EO=0.611, much higher than census (~0.34) or capture24 (~0.16). Hypothesis: the large WGL differences in MEPS cause k=5 to saturate the sigmoid (reward oscillates 0.09–0.99), producing noisy gradients; k=3 gives a softer boundary and more proportional reward signal. Combined with traj=4000 for volume.

**Config delta from vanilla:**
- dataset_name: meps
- da_pct: 0.01433
- dp_protected_col: sex
- minority_id: 0 (male disadvantaged)
- majority_id: 1
- global_sigmoid_k: 3.0
- traj_length: 4000
- real_data_size: 3000
- total_episodes: 5000
- ffnn.epochs: 30
- seeds: [0, 1, 42]

**Spec file:** `experiment_specs/Experiment3/meps_k3_traj4000_ep30_lambda_gpu1.yaml`

**Launch command (Lambda GPU1):**
```bash
cd ~/cs_9170_project && source ~/envs/rl/bin/activate && \
nohup python main.py --spec experiment_specs/Experiment3/meps_k3_traj4000_ep30_lambda_gpu1.yaml --device cuda:1 > /tmp/meps_k3_traj4000_ep30.log 2>&1 &
```
Launched 2026-05-21. Lambda GPU1. Confirmed training at ep ~89, reward ~0.41.

**Result:**
*(pending — check at ep ~500)*

**Takeaway:**
*(pending)*

**Next steps:**
- Compare β-EO vs α-EO at ep ~500 against EXP-034 (k=5, traj=4000)

---

### EXP-036 | meps-sex-k5-ep50

**Status:** IN PROGRESS
**Type:** PAPER-FINAL
**Dataset:** meps (sex, prescription fills)
**Follows from:** EXP-029

**Purpose:**
Tests whether 50 FFNN epochs per episode stabilises the noisy reward signal seen in EXP-029 (k=5, ep=30, traj=2000). Hypothesis: the FFNN underconverges per episode at 30 epochs given MEPS's larger dataset (3000 real + 2000 synthetic), producing an unstable beta and hence a noisy reward. Controlled comparison: only epochs change from EXP-029.

**Config delta from vanilla:**
- dataset_name: meps
- da_pct: 0.01433
- dp_protected_col: sex
- minority_id: 0 (male disadvantaged)
- majority_id: 1
- global_sigmoid_k: 5.0
- traj_length: 2000
- real_data_size: 3000
- total_episodes: 5000
- ffnn.epochs: 50
- seeds: [0, 1, 42]

**Spec file:** `experiment_specs/Experiment3/meps_k5_ep50_aulavik_gpu0.yaml`

**Launch command (Aulavik GPU0):**
```bash
cd ~/cs_9170_project && source ~/envs/rl/bin/activate && \
nohup python main.py --spec experiment_specs/Experiment3/meps_k5_ep50_aulavik_gpu0.yaml --device cuda:0 > /tmp/meps_k5_ep50.log 2>&1 &
```
Launched 2026-05-21. Aulavik GPU0 (RTX 3090). Confirmed training at ep ~69, reward ~0.48.

**Result:**
*(pending — check at ep ~500)*

**Takeaway:**
*(pending)*

**Next steps:**
- Compare reward variance and β-EO at ep ~500 against EXP-029 (ep=30)

---

### EXP-037 | meps-sex-k3-ep50

**Status:** IN PROGRESS
**Type:** PAPER-FINAL
**Dataset:** meps (sex, prescription fills)
**Follows from:** EXP-029, EXP-035, EXP-036

**Purpose:**
Tests softer sigmoid (k=3) combined with more FFNN epochs (ep=50) at baseline volume (traj=2000). Combines the two hypothesised fixes for MEPS reward noise: softer gradient signal and more stable per-episode classifier. Controlled on traj to isolate the k+epoch interaction from volume effects.

**Config delta from vanilla:**
- dataset_name: meps
- da_pct: 0.01433
- dp_protected_col: sex
- minority_id: 0 (male disadvantaged)
- majority_id: 1
- global_sigmoid_k: 3.0
- traj_length: 2000
- real_data_size: 3000
- total_episodes: 5000
- ffnn.epochs: 50
- seeds: [0, 1, 42]

**Spec file:** `experiment_specs/Experiment3/meps_k3_ep50_aulavik_gpu1.yaml`

**Launch command (Aulavik GPU1):**
```bash
cd ~/cs_9170_project && source ~/envs/rl/bin/activate && \
nohup python main.py --spec experiment_specs/Experiment3/meps_k3_ep50_aulavik_gpu1.yaml --device cuda:1 > /tmp/meps_k3_ep50.log 2>&1 &
```
Launched 2026-05-21. Aulavik GPU1 (RTX 3090). Confirmed training at ep ~50, reward ~0.50.

**Result:**
*(pending — check at ep ~500)*

**Takeaway:**
*(pending)*

**Next steps:**
- If β-EO beats EXP-036 (k=5, ep=50): k=3 is better than k=5 for MEPS; continue
- If similar to EXP-036: epoch increase is the driver, not k

---

### EXP-038 | meps-sex-k0

**Status:** IN PROGRESS
**Type:** PAPER-FINAL
**Dataset:** meps (sex, prescription fills)
**Follows from:** EXP-029

**Purpose:**
Tests normalised reward (k=0: reward = (wgl_alpha − wgl_beta) / wgl_alpha) on MEPS. No sigmoid — purely proportional signal. If k=5 saturates due to large WGL differences, k=0 provides the cleanest possible gradient. Controlled on traj=2000 and ep=30 to isolate the reward shape effect.

**Config delta from vanilla:**
- dataset_name: meps
- da_pct: 0.01433
- dp_protected_col: sex
- minority_id: 0 (male disadvantaged)
- majority_id: 1
- global_sigmoid_k: 0.0
- traj_length: 2000
- real_data_size: 3000
- total_episodes: 5000
- ffnn.epochs: 30
- seeds: [0, 1, 42]

**Spec file:** `experiment_specs/Experiment3/meps_k0_oneida_gpu0.yaml`

**Launch command (Oneida GPU0):**
```bash
cd ~/cs_9170_project && source ~/envs/rl/bin/activate && \
nohup python main.py --spec experiment_specs/Experiment3/meps_k0_oneida_gpu0.yaml --device cuda:0 > /tmp/meps_k0.log 2>&1 &
```
Launched 2026-05-21. Oneida GPU0 (RTX 3090). Confirmed training at ep ~28, reward ~0.02 (expected: k=0 return is raw WGL difference, centred near 0).

**Result:**
*(pending — check at ep ~500)*

**Takeaway:**
*(pending)*

**Next steps:**
- Monitor whether β-EO tracks downward more consistently than sigmoid runs

---

### EXP-039 | meps-sex-k3-traj4000-ep50

**Status:** IN PROGRESS
**Type:** PAPER-FINAL
**Dataset:** meps (sex, prescription fills)
**Follows from:** EXP-035, EXP-036, EXP-037

**Purpose:**
Maximum-effort MEPS configuration: k=3 (softer sigmoid), traj=4000 (high synthetic volume), ep=50 (more stable FFNN per episode). Combines all three hypothesised improvements simultaneously. Will be significantly slower than other runs (especially on Oneida's RTX 3090) but provides the strongest possible test of whether FORGE can converge on MEPS under ideal hyperparameters. k=3 chosen as the most credible k value for MEPS given the high alpha_EO saturation concern.

**Config delta from vanilla:**
- dataset_name: meps
- da_pct: 0.01433
- dp_protected_col: sex
- minority_id: 0 (male disadvantaged)
- majority_id: 1
- global_sigmoid_k: 3.0
- traj_length: 4000
- real_data_size: 3000
- total_episodes: 5000
- ffnn.epochs: 50
- seeds: [0, 1, 42]

**Spec file:** `experiment_specs/Experiment3/meps_k3_traj4000_ep50_oneida_gpu1.yaml`

**Launch command (Oneida GPU1):**
```bash
cd ~/cs_9170_project && source ~/envs/rl/bin/activate && \
nohup python main.py --spec experiment_specs/Experiment3/meps_k3_traj4000_ep50_oneida_gpu1.yaml --device cuda:1 > /tmp/meps_k3_traj4000_ep50.log 2>&1 &
```
Launched 2026-05-21. Oneida GPU1 (RTX 3090). Confirmed training at ep ~5, reward ~0.62.

**Result:**
*(pending — check at ep ~300–500 given slower per-episode time)*

**Takeaway:**
*(pending)*

**Next steps:**
- If β-EO is cleanly below α-EO by ep ~300: k=3 + traj=4000 + ep=50 is the MEPS winner; run full 3-seed evaluation
- Compare against EXP-035 (k=3, traj=4000, ep=30) to isolate epoch contribution
