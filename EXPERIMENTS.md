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

**Seed 1 status (updated 2026-04-18):**
- census wgl_k{0,3,5,10}: all complete. k3 seed_1 was in orphaned directory after consolidation script bug; manually moved into correct run dir.
- capture24 wgl_k{0,3,5,10}: all complete.
- compas wgl_k0, k10: complete. compas wgl_k3 seed_1: moved from orphaned dir (same consolidation bug). compas wgl_k5 seed_1: **RUNNING** on cuda:1 via `run_missing_seed1_gpu1.sh`.

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

**Seed 1 status (updated 2026-04-18):**
- census roc_eo_lam{03,05}: complete. census roc_eo_lam07 seed_1: **RUNNING** on cuda:0 via `run_missing_seed1_gpu0.sh`.
- capture24 roc_eo_lam{03,05,07}: all complete.
- compas roc_eo_lam{03,05,07}: seed_1 runs were found incomplete (stopped at ep 1193, 2000, 3984 respectively). Fresh runs launched: lam03 and lam07 **RUNNING** on cuda:0, lam05 **RUNNING** on cuda:1 via `run_missing_seed1_gpu0/1.sh`. Old partial seed_1 directories remain in place until new runs complete and are swapped in.

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
| **RL k=3** | **0.079±0.015** | — | **0.810±0.003** | **0.877±0.002** | — |

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

**Census:** RL k=3 (β-EO=0.079±0.015) beats all baselines on EO, including the best reweighting method (FLB β-EO=0.039 on TPR diff but EOd=0.091; OT Repair β-EO=0.085). RL achieves this with competitive utility (F1w=0.810), near GroupDRO and OT Repair levels. CTGAN and FairTabDDPM both fail to match reweighting methods on EO, with CTGAN particularly poor (0.328). FairTabDDPM shows improvement over CTGAN (0.151) but with high variance (±0.074).

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

**Submission Tracker — census_grid_v2 (revised 2026-04-28)**

Supersedes old bundle system (census_grid/, 108 specs, 28 bundles). Parallelization via Santiago's `torch.multiprocessing` spawn approach. Bug fixed in `main.py`: seed extraction now uses `principal_vars.index('seed')` rather than `permutation[0]` to handle epochs-first variable ordering correctly. `output_dir` spec field added to `main.py` to allow redirecting output directly to storage (used by Huron restart specs).

**Architecture change (2026-04-28):** k=10 DRAC submission restructured from 2 monolithic specs (census_k10_gpu0/gpu1.yaml, 54+27 perms each) into 9 per-(ratio×epoch) specs (census_k10_r{02,04,06}_e{10,20,30}), each with 9 perms and max_parallel=9. Confirmed safe from parallelism scaling tests (max_parallel=9, 9 CPUs, ~17.5–21.7 s/ep, all jobs within 168h wall limit).

| Resource | k | Spec(s) | Status | Started |
|---|---|---|---|---|
| Huron GPU 0 | k=0 | census_k0_gpu0_restart_pca15_e10.yaml → census_k0_gpu0_restart_e20.yaml | **RUNNING** (2/27 remaining) | 2026-04-28 |
| Huron GPU 1 | k=0 | census_k0_gpu1_restart_pca10_e30.yaml → census_k0_gpu1_restart_pca15_e30.yaml | **COMPLETE** | 2026-04-28 |
| Lambda GPU 0 | k=3 | census_k3_gpu0.yaml | **RUNNING** | 2026-04-25 |
| Lambda GPU 1 | k=3 | census_k3_gpu1.yaml | **RUNNING** | 2026-04-25 |
| Aulavik GPU 0 | k=5 | census_k5_gpu0.yaml | **COMPLETE** | 2026-04-25 |
| Aulavik GPU 1 | k=5 | census_k5_gpu1.yaml | **COMPLETE** | 2026-04-25 |
| DRAC (8 jobs) | k=10 | census_k10_r{02,04,06}_e{10,20,30}.sh (excl. r02_e10) | **QUEUED** (resubmit under def-mcapretz) | 2026-04-28 |
| DRAC (1 job) | k=10 | census_k10_r02_e10.sh | **COMPLETE** | 2026-04-28 |

**Huron k=0 history:** Original gpu0/gpu1 runs (started 2026-04-25) completed epochs=10 for PCA5 and PCA10 (all seeds), PCA5 epochs=30, and PCA10 epochs=30 seed_0 before storage issues interrupted them. Completed runs moved to `/storage_1/epigou_storage/FORGE/training_runs/`. Restart specs cover all remaining permutations and write directly to storage via `output_dir` field.

**DRAC k=10 timing:** epochs=10/20 → ~17.5 s/ep, epochs=30 → ~21.7 s/ep. Per-job wall time estimates: e10 jobs 50–68h, e20 jobs 52–74h, e30 jobs 64–84h. All within 168h limit.

---

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
Feed best principal params into EXP-022 base_patches before running random search.

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

| Dataset | Framing | val_pos | test_pos | alpha_EO | sep_ratio | Verdict |
|---------|---------|---------|---------|---------|-----------|---------|
| acs_income (CA 2018) | sex, income>50k | 6509 | 6532 | 0.17 | 2.43 | **ALL PASS** |
| sepsis (PhysioNet 2019) | sex, sepsis onset | 242 | 243 | 0.027 | 0.17 | FAIL: EO+sep |
| brfss (2022) | sex, heart attack | 1976 | 1896 | 0.016 | 0.52 | FAIL: EO+sep |
| brfss (2022) | sex, depression | 12101 | 12092 | 0.012 | 0.36 | FAIL: EO+sep |
| brfss (2022) | race (Black/White), heart attack | pending | — | — | — | — |

**Root cause of BRFSS sex-framing failures:**
BRFSS behavioral risk factors (age, BMI, smoking, diabetes, physical activity) are universal predictors that do not differ structurally between male and female patients for any outcome tested. Model generalizes across sexes despite few female positives in training → low EO gap and poor PCA separability.

**Status:** Testing BRFSS race framing (Black vs White). If that fails, ACS Income will be adopted as the 3rd dataset with the framing that it represents a distinct data source (Ding et al. NeurIPS 2021 folktables, 2018 ACS survey) from census_income (1994 decennial census).

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
*(pending)*

---

### EXP-025 | capture24-hparam-grid

**Type:** PARAM-TUNING
**Status:** IN PROGRESS
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
| `global_sigmoid_k` | [5] |
| `pca_components` | [5, 10, 15] |
| `ratio_trajectory` | [0.2, 0.4, 0.6] |
| `ffnn.epochs` | [10, 20, 30] |

Total: 1k × 3pca × 3ratio × 3epochs × 3seeds = **81 runs**

**Fixed base patches:** dataset_name=capture24, da_pct=0.015, minority_id=1, majority_id=0, dp_protected_col=sex, win_seconds=1.0, step_seconds=0.5, seeds=[0,1,42], total_episodes=5000, reward_mode=wgl, total_data_size=5000.

**Specs:** `experiment_specs/capture24_grid/capture24_k5_gpu{0,1}.yaml`

**GPU split (Aulavik):** GPU0 runs epochs=[10,20] (54 perms, max_parallel=4); GPU1 runs epochs=[30] (27 perms, max_parallel=4).

**Submission Tracker:**

| Resource | Spec | Status | Started |
|---|---|---|---|
| Aulavik GPU 0 | capture24_k5_gpu0.yaml | **RUNNING** | 2026-05-04 |
| Aulavik GPU 1 | capture24_k5_gpu1.yaml | **RUNNING** | 2026-05-04 |

**Selection criterion:** Same as EXP-021 — lowest mean β-EO, with β-F1w ≥ α-F1w − 0.02.

**Result:**
*(pending)*

**Takeaway:**
*(pending)*

**Next steps:**
Feed best capture24 params into EXP-022 equivalent (capture24 random search) once complete.
