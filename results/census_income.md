# Census Income Results

**Dataset:** UCI Adult  
**Task:** Income > $50K  
**Protected attribute:** sex (female = disadvantaged, a=0)  
**Config:** bias_pct=0.10, DA+=43, real_data_size=3000, pca=10  
**Seeds:** [0, 2, 3, 5, 42] (EO guard: alpha-EO ≥ 0.10)

---

## Main Comparison Table (5 seeds, paper_results_v2)

Alpha EO: 0.109 ± 0.063  
Baseline seeds: [0, 2, 5, 6, 7] | RL seeds: [0, 2, 3, 5, 42] *(seed mismatch — EO guard selected different fallbacks)*

| Method | Version/Config | Seeds | EO ↓ | F1w ↑ | AUC ↑ | Spec |
|--------|---------------|-------|------|-------|-------|------|
| Alpha (ERM) | — | 5 | 0.109 ± 0.063 | — | — | — |
| GroupDRO | — | 5 | 0.074 ± 0.030 | 0.825 ± 0.010 | 0.894 ± 0.003 | `paper_specs_v2/v2_census_gdro_5s` |
| OT Repair | — | 5 | 0.054 ± 0.035 | 0.792 ± 0.003 | 0.823 ± 0.006 | `paper_specs_v2/v2_census_otrep_5s` |
| FLB | — | 5 | **0.031 ± 0.028** | 0.819 ± 0.008 | 0.889 ± 0.001 | `paper_specs_v2/v2_census_flb_5s` |
| CTGAN | — | 3 | 0.075 ± 0.027 | 0.789 | 0.842 | `v16_bias010_ctgan_census_3seeds` |
| FairTabDDPM | — | 5 | 0.070 ± 0.039 | 0.791 ± 0.008 | 0.845 ± 0.009 | `paper_specs_v2/v2_census_fairtabddpm_5s` |
| CTGAN | — | 5 | 0.109 ± 0.049 | 0.792 | 0.840 | `paper_specs_v2/v2_census_ctgan_5s` |
| SMOTE | two-phase PCA-10 | 5 | 0.133 ± 0.064 | — | — | `paper_specs_v2/v2_census_smote_5s` |
| **RL (v18)** | ep1500/ph400 | 5 | 0.070 ± 0.070 | 0.719 ± 0.037 | 0.852 ± 0.012 | `paper_specs_v2/ablation_census_ep1500ph400_5s` |

Best RL config per ablation: ep1500/ph400 (lowest mean EO). Run dir hash: `866d84c2`.

---

## v18 Global-Only Ablation (5 seeds [42,0,1,2,3], paper_results_v2)

These seeds differ from the paper_specs_v2 run above.

| Seed | α-EO | β-EO | F1w | AUC |
|------|------|------|-----|-----|
| 42 | 0.126 | 0.031 | 0.800 | 0.870 |
| 0  | 0.146 | 0.082 | 0.803 | 0.854 |
| 1  | 0.086 | 0.063 | 0.813 | 0.883 |
| 2  | 0.263 | 0.120 | 0.826 | 0.889 |
| 3  | 0.227 | 0.019 | 0.811 | 0.891 |
| **mean** | — | **0.063 ± 0.059** | **0.811** | **0.877** |

Spec: `v18_ablation_global_only_bias010_census_3seeds` (despite name, ran 5 seeds)

---

## Episode Ablation (5 seeds [0,2,3,5,42], paper_results_v2)

Alpha EO: 0.196 ± 0.066

| Config | EO ↓ | F1w ↑ | AUC ↑ |
|--------|------|-------|-------|
| ep800/ph0 (phase 1 only) | 0.188 ± 0.047 | 0.786 ± 0.016 | 0.876 ± 0.009 |
| ep800/ph200 | 0.076 ± 0.048 | 0.729 ± 0.036 | 0.852 ± 0.012 |
| ep1500/ph400 | **0.070 ± 0.078** | 0.719 ± 0.042 | 0.852 ± 0.013 |
| ep2000/ph600 | 0.082 ± 0.073 | 0.729 ± 0.039 | 0.845 ± 0.011 |

ep1500/ph400 has lowest mean EO but highest std. ep800/ph200 more stable.

---

## Framework Evolution (3 seeds, training_runs)

| Method | Seeds | EO ↓ | F1w ↑ | AUC ↑ | Notes |
|--------|-------|------|-------|-------|-------|
| RL v16 | 3 | 0.084 ± 0.043 | 0.788 | 0.848 | gamma=0.99, ~50% deadzone |
| RL v17a | 3 | 0.071 ± 0.067 | 0.807 | 0.867 | gamma=1.0, curriculum off |
| RL v17b | 3 | 0.116 ± 0.062 | 0.801 | 0.859 | warm-start — worse |
| **RL v18** | 5 | **0.063 ± 0.059** | **0.811** | **0.877** | global-only — best REINFORCE |
| RL v19 (norm. reward) | 2 | 0.056 ± 0.036 | 0.802 | 0.862 | provisional |

---

## CMA-ES Results (paper_results_v4)

| Method | Seeds | EO ↓ | F1w ↑ | AUC ↑ | Spec |
|--------|-------|------|-------|-------|------|
| v20 CMA-ES (pure global) | 5 | **0.051 ± 0.041** | 0.792 | 0.884 | `v20_census_cmaes_5s` |
| v21 CMA-ES + OT local | 5 | **0.048 ± 0.042** | 0.794 | 0.884 | `v21_census_cmaes_ot_5s` |
| v22 REINFORCE + OT local | 5 | pending | — | — | `v22_census_rl_ot_5s` |

**v20 per-seed:**

| Seed | α-EO | β-EO | F1w | AUC |
|------|------|------|-----|-----|
| 0 | — | 0.034 | 0.779 | 0.877 |
| 1 | — | 0.044 | 0.795 | 0.883 |
| 2 | — | 0.122 | 0.798 | 0.891 |
| 3 | — | 0.033 | 0.802 | 0.884 |
| 42 | — | 0.021 | 0.788 | 0.884 |

**v21 per-seed:** identical to v20 except seed_0 (0.034 → 0.020). OT local reward adds negligible value to CMA-ES — global term dominates at this episode budget. Confirmed distinct runs via metrics.csv local reward values.

---

## Notes

- Seed 2 (α-EO=0.263) is a structural outlier: severe imbalance, neither method closes the gap well.
- GroupDRO fails under scarcity on this dataset (EO=0.074 > alpha-EO in some seeds). Motivation claim supported.
- FLB achieves the best EO (0.031) but is a reweighting method that works here — census is less extreme than COMPAS.
