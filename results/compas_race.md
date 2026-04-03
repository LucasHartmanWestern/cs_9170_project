# COMPAS Results (Race — Investigated, Narrative Fails)

**Dataset:** ProPublica COMPAS Recidivism  
**Task:** Recidivism prediction (2-year)  
**Protected attribute:** race (Caucasian = disadvantaged, a=0)  
**Config:** bias_pct=0.05, DA+=40, pca=10  
**Seeds:** [0, 2, 3, 5, 42]  
**Status:** Investigated and dropped from paper — reweighting baselines outperform RL. Motivation claim does not hold.

---

## Main Comparison Table (5 seeds, paper_results_v3)

Alpha EO: 0.677 ± 0.024 (Caucasian TPR collapses to ~0.024 under bias)

| Method | Seeds | EO ↓ | F1w | AUC | Spec |
|--------|-------|------|-----|-----|------|
| Alpha (ERM) | 5 | 0.677 ± 0.024 | 0.641 | 0.681 | — |
| GroupDRO | 5 | **0.197 ± 0.105** | 0.650 | 0.692 | `compas_race_bias005_gdro_5s` |
| OT Repair | 5 | 0.556 ± 0.041 | 0.637 | 0.675 | `compas_race_bias005_otrep_5s` |
| FLB | 5 | **0.075 ± 0.054** | 0.628 | 0.671 | `compas_race_bias005_flb_5s` |
| FairTabDDPM | 5 | 0.541 ± 0.043 | 0.647 | 0.686 | `compas_race_ep1500ph400_5s` |
| RL (v18, ep1500/ph400) | 5 | 0.600 ± 0.055 | 0.641 | 0.682 | `compas_race_ep1500ph400_5s` |
| v20 CMA-ES | 2 | 0.597 ± 0.039 | 0.652 | 0.691 | `v20_compas_cmaes_2s` |
| CTGAN | — | timed out on DRAC | | | |
| SMOTE | — | timed out on DRAC | | | |

---

## Structural Analysis

| Metric | Value | Census (reference) |
|--------|-------|--------------------|
| DA+ | 40 | 43 |
| Val Caucasian positives | **14** | 86 |
| Test Caucasian positives | 162 | 244 |
| Alpha-EO | 0.677 | 0.109 |
| Cosine (adv/disadv PCA) | 0.757 | 0.983 |

Val_pos=14 is the root cause of failure — the reward signal is computed on only 14 validation
examples, too noisy for stable policy gradient. GroupDRO and FLB work because they operate
directly on the training labels, bypassing the noisy val signal.

---

## Why the Narrative Fails Here

1. The race signal in COMPAS is a strong predictor — reweighting on it works even at low count.
2. Caucasian TPR collapses to ~0.024 under alpha, creating an extreme EO gap (0.677) that
   the delta-action agent cannot bridge from such a small search budget.
3. PCA cosine=0.757 (vs 0.983 for census) suggests adv/disadv groups are less separable
   in PCA space — the OT target is less informative.
4. Only 14 val Caucasian positives — worst of all candidates evaluated.

---

## Notes

- This config was replaced in the paper by COMPAS sex (bias_pct=0.14, sex protected attr).
- The race investigation is worth keeping as a supplementary discussion point: not all scarcity
  configurations are equally solvable by generative augmentation.
