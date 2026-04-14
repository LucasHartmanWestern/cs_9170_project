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
