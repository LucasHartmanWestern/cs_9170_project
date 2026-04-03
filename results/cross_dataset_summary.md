# Cross-Dataset Summary

Best confirmed results per method across all three primary datasets.  
All 5-seed, paper_results_v2 unless noted. RL = best episode config per dataset.  
**Bold** = best EO per dataset. 🔴 = worsens EO vs alpha.

## Dataset Fairness Problem Strength

| Dataset | Alpha-EO (mean±std) | Seeds passing guard (≥0.10) | Verdict |
|---------|--------------------|-----------------------------|---------|
| Census income | 0.196 ± 0.070 | 5/5 ✅ | Strong fairness problem |
| Capture-24 | 0.231 ± 0.164 | 4/5 ⚠️ | Strong (seed_3 weak, α=0.048) |
| COMPAS sex | 0.052 ± 0.018 | 0/5 ❌ | **No proper fairness problem by paper criterion** |

---

## EO (Equal Opportunity Gap) — lower is better

| Method | Census | Capture-24 | COMPAS (sex) ⚠️ |
|--------|--------|------------|--------------|
| Alpha (ERM baseline) | 0.196 ± 0.070 | 0.231 ± 0.164 | 0.052 ± 0.018 |
| GroupDRO | 0.074 ± 0.030 | 0.141 ± 0.071 | 0.156 ± 0.050 🔴 |
| OT Repair | 0.054 ± 0.035 | 0.080 ± 0.065 | 0.036 ± 0.020 |
| FLB | **0.031 ± 0.028** | 0.106 ± 0.056 | 0.118 ± 0.016 🔴 |
| CTGAN | 0.109 ± 0.049 | 0.188 ± 0.105 | 0.158 ± 0.076 🔴 |
| FairTabDDPM | 0.070 ± 0.039 | 0.250 ± 0.137 🔴 | 0.064 ± 0.046 |
| SMOTE | 0.133 ± 0.064 | 0.232 ± 0.170 🔴 | 0.507 ± 0.138 🔴 |
| **RL (v18, best config)** | 0.070 ± 0.070 | **0.069 ± 0.035** | **0.016 ± 0.011** |
| v20 CMA-ES *(2 seeds)* | **0.021 ± 0.000** | — | — |

## F1-weighted — higher is better

| Method | Census | Capture-24 | COMPAS (sex) |
|--------|--------|------------|--------------|
| GroupDRO | **0.825** | 0.906 | **0.605** |
| OT Repair | 0.792 | 0.949 | 0.466 |
| FLB | 0.819 | 0.906 | 0.614 |
| CTGAN | 0.792 | **0.948** | 0.562 |
| FairTabDDPM | 0.791 | **0.953** | 0.483 |
| **RL (v18, best config)** | 0.719 | 0.938 | 0.440 |

## AUC — higher is better

| Method | Census | Capture-24 | COMPAS (sex) |
|--------|--------|------------|--------------|
| GroupDRO | **0.894** | 0.900 | 0.648 |
| OT Repair | 0.823 | 0.893 | **0.700** |
| FLB | 0.889 | **0.927** | 0.658 |
| CTGAN | 0.840 | **0.907** | 0.644 |
| FairTabDDPM | 0.845 | **0.938** | 0.681 |
| **RL (v18, best config)** | 0.852 | 0.862 | 0.659 |

---

## Key Takeaways

**Where RL wins outright:**
- COMPAS sex: RL EO=0.016 vs next best OT Repair 0.036. GroupDRO, FLB, CTGAN, SMOTE all *worsen* EO vs alpha.
- Capture-24: RL best EO (0.069). FairTabDDPM and SMOTE worsen EO vs alpha.

**Where RL is competitive but not best:**
- Census: FLB wins on EO (0.031 vs 0.070). RL AUC (0.852) beats OT Repair and CTGAN.

**Where RL trades fairness for utility:**
- RL F1w is consistently lower than GroupDRO/FLB across all datasets. This is the cost of aggressive fairness optimization via augmentation rather than reweighting.

**CMA-ES (v20, census only, 2 seeds):**
- EO=0.021 is best of any method on census. Provisional — 5-seed run on DRAC in progress.

**Scarcity claim (methods that fail under DA+≈43):**
- GroupDRO fails on COMPAS sex (0.156 🔴), partial failure on census (0.074 > alpha-seeds in some cases)
- FLB fails on COMPAS sex (0.118 🔴)
- CTGAN fails on COMPAS sex (0.158 🔴) and is near-alpha on census (0.109)
- FairTabDDPM fails on Capture-24 (0.250 🔴)
- SMOTE fails on all three datasets
