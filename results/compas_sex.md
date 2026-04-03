# COMPAS Results (Sex — Primary Protected Attribute)

**Dataset:** ProPublica COMPAS Recidivism  
**Task:** Recidivism prediction (2-year)  
**Protected attribute:** sex (female = disadvantaged, a=0)  
**Config:** bias_pct=0.14, DA+=43, pca=10  
**Seeds:** [1, 3, 6, 7, 42]  
**⚠️ EO guard failure:** All 5 seeds have alpha-EO < 0.10 (range 0.026–0.074, mean 0.052±0.018). The EO guard criterion was not satisfied — either it was disabled or no seeds passed. This dataset does **not** have a proper fairness problem by the paper's own inclusion criterion.

---

## Main Comparison Table (5 seeds, paper_results_v2)

Alpha EO: 0.070 ± 0.044

| Method | Version/Config | Seeds | EO ↓ | F1w ↑ | AUC ↑ | Spec |
|--------|---------------|-------|------|-------|-------|------|
| Alpha (ERM) | — | 5 | 0.070 ± 0.044 | — | — | — |
| GroupDRO | — | 5 | 0.156 ± 0.050 🔴 | **0.605 ± 0.043** | 0.648 ± 0.050 | `paper_specs_v2/v2_compas_gdro_5s` |
| OT Repair | — | 5 | 0.036 ± 0.020 | 0.466 ± 0.015 | **0.700 ± 0.009** | `paper_specs_v2/v2_compas_otrep_5s` |
| FLB | — | 5 | 0.118 ± 0.016 🔴 | **0.614 ± 0.050** | 0.658 ± 0.069 | `paper_specs_v2/v2_compas_flb_5s` |
| FairTabDDPM | — | 5 | 0.064 ± 0.046 | 0.483 ± 0.021 | 0.681 ± 0.013 | `paper_specs_v2/v2_compas_fairtabddpm_5s` |
| CTGAN | — | 5 | 0.158 ± 0.076 🔴 | 0.562 | 0.644 | `paper_specs_v2/v2_compas_ctgan_5s` |
| SMOTE | two-phase PCA-10 | 5 | 0.507 ± 0.138 🔴 | — | — | `paper_specs_v2/v2_compas_smote_5s` |
| **RL (v18)** | ep2000/ph600 | 5 | **0.016 ± 0.011** | 0.440 ± 0.031 | 0.659 ± 0.006 | `paper_specs_v2/ablation_compas_ep2000ph600_5s` |

🔴 = worsens EO relative to alpha. Run dir hash: `7e86910b`.

---

## Episode Ablation (5 seeds [1,3,6,7,42], paper_results_v2)

Alpha EO: 0.052 ± 0.017

| Config | EO ↓ | F1w ↑ | AUC ↑ |
|--------|------|-------|-------|
| ep800/ph0 (phase 1 only) | 0.047 ± 0.017 | **0.457 ± 0.024** | **0.703 ± 0.019** |
| ep800/ph200 | 0.038 ± 0.034 | 0.436 ± 0.038 | 0.672 ± 0.020 |
| ep1500/ph400 | 0.039 ± 0.027 | 0.444 ± 0.030 | 0.668 ± 0.015 |
| ep2000/ph600 | **0.016 ± 0.012** | 0.440 ± 0.035 | 0.659 ± 0.007 |

More episodes strongly help EO. ep2000/ph600 is the clear best (0.016 vs 0.047 at ep800/ph0).

---

## Per-Seed Alpha-EO Breakdown

| Seed | α-EO | Passes guard (≥0.10)? | β-EO | F1w |
|------|------|-----------------------|------|-----|
| 1 | 0.054 | ❌ | 0.020 | 0.492 |
| 3 | 0.026 | ❌ | 0.007 | 0.439 |
| 6 | 0.058 | ❌ | 0.035 | 0.444 |
| 7 | 0.051 | ❌ | 0.004 | 0.428 |
| 42 | 0.074 | ❌ | 0.012 | 0.395 |
| **mean** | **0.052±0.018** | **0/5** | **0.016±0.012** | **0.440** |

## Notes

- **⚠️ Weak fairness problem.** Alpha-EO mean = 0.052, all seeds below the 0.10 inclusion threshold. The RL "win" (0.016) is real but the initial gap was small. A reviewer will question whether this constitutes a meaningful fairness problem.
- GroupDRO (0.156) and FLB (0.118) worsen EO vs alpha, as do CTGAN and SMOTE — consistent with the scarcity framing. But the alpha baseline is so low that "worsening" is less dramatic than it appears.
- RL F1w (0.440) is substantially lower than reweighting methods (GroupDRO 0.605, FLB 0.614).
- COMPAS sex (bias_pct=0.14) is distinct from COMPAS race (bias_pct=0.05) — see compas_race.md.
- **Decision needed:** Whether to include COMPAS sex in the paper given the EO guard failure.
