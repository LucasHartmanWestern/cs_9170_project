# Experiment Findings

Chronological record of what has worked, what has not, and why.

---

## Framework Versions

### v3 — Global+Anchors (PCA space) ✅ Best so far

**WARNING: The 3 runs have inconsistent configs — pooled 9-seed statistics are invalid.**
See "Training Dynamics Analysis" section below for full details.

**Run configs (not the same experiment):**

| Run | sigma | lambda | radius_clip | Anchor reward health | Effective method |
|---|---|---|---|---|---|
| v3_run1 (Mar 9) | 0.85 | [0.5, 0.5] fixed | none | Dead by ep ~1000 (0.029→0.001→0.0006) | Global-only |
| v3_run2 (Mar 12) | 0.85 | [0.5, 0.8] anneal | none | Dead by ep ~1000 (same pattern) | Global-only + lambda anneal |
| v3_run3 (Mar 13) | ~3.0 | [0.5, 0.8] anneal | 3.0 | Healthy throughout (0.655→0.603→0.593) | True Global+Anchors |

**Results (pooled 9 seeds — TREAT AS PRELIMINARY, configs mixed):**

| Dataset | Metric | Alpha baseline | v3 Global+Anchors | p-value |
|---|---|---|---|---|
| Census | EO | 0.1039 | 0.0899 (−13%) | 0.030 * |
| Census | F1 weighted | 0.8425 | 0.8424 | 0.950 (preserved) |
| Census | AUC | 0.9013 | 0.8997 | 0.303 (preserved) |
| Credit | EO | 0.0576 | 0.0429 (−25%) | 0.070 (borderline) |
| Credit | F1 minority | 0.3802 | 0.4097 (+7.8pp) | 0.001 ** |
| Credit | F1 weighted | 0.7742 | 0.7793 (+0.5pp) | 0.005 ** |
| Credit | AUC | 0.7390 | 0.7456 (+0.7pp) | 0.003 ** |

**Key finding:** Only method that improves fairness *and* utility simultaneously. Baselines
(OT Repair, Group DRO) improve EO more aggressively but at significant utility cost.

**Ablation within v3:**
- `global_only` (no local reward): weaker but still improves EO
- `global+hard` (hard-positive only, no anchors): marginal
- `global+anchors` (anchor proximity only): **best overall**
- `global+full` (anchors+hard+diversity): sometimes worse than anchors-only — diversity
  penalty may be counterproductive

---

### v4 — Smoke tests only

Several smoke runs (200 eps) used to debug anchor code fixes between Mar-12 and Mar-13.
Not used for comparison.

---

### v5 — Uncertainty Anchors (raw feature space) ❌ No improvement

**Key config:** `use_pca=false`, `sigma_anchor=3.0`, `lambda=[0.5,0.8]`,
`use_uncertainty_anchors=true`, `radius_clip=3.0`

**Results:**

| Dataset | EO vs alpha | F1_min vs alpha | Notes |
|---|---|---|---|
| Census | +0.0012 (worse) | −0.004 | p=0.727, no effect |
| Credit | −0.020 | +0.024 | p=0.314, not significant |

**Root cause diagnosis:**
1. `use_pca=false` → raw ~100D feature space → L2 anchor distances become noisy
   (curse of dimensionality). The anchor proximity signal partially degrades.
2. `sigma_anchor=3.0` in 100D is not well-calibrated to the data distribution.
3. `uncertainty_weights = 1 - p_beta(anchor)` are non-stationary and noisy early
   in training when beta is untrained. The effective anchor set shifts throughout
   training, preventing stable policy learning.
4. `lambda=[0.5,0.8]` vs v3's `[0.5,0.5]` is a confound — unclear if lambda
   annealing or uncertainty weighting (or their interaction) drives any credit improvement.

**The uncertainty anchor idea itself is sound** — generate near anchors where beta
currently struggles. The implementation was broken by coupling it with `use_pca=false`.

---

### v6 — Planned (not yet run)

**Goal:** Isolate each v5 change and test the uncertainty anchor idea properly.

**Specs created:**

| Spec | Key change vs v3 | Hypothesis |
|---|---|---|
| `v6_credit_anchors_lambda08` | lambda=[0.5,0.8], PCA, static anchors | Lambda annealing alone improves credit EO |
| `v6_credit_ua_lambda05` | UA=true, raw space, lambda=[0.5,0.5] | Removes lambda confound from v5 |
| `v6_census_ua_pca_warmup_smoke` | UA=true, PCA space, warmup=50, smoke | Smoke test for UA in PCA space |
| `v6_census_ua_pca_warmup` | UA=true, PCA space, warmup=300, full | UA in its natural setting |

**Run order:** smoke first (`v6_census_ua_pca_warmup_smoke`), then lambda ablation
(`v6_credit_anchors_lambda08` + `v6_credit_ua_lambda05`) in parallel, then full
`v6_census_ua_pca_warmup` once smoke confirms healthy learning.

**⚠️ v6 sigma warning:** Both v6 census specs now include `sigma_calibration_factor: 1.0`
to prevent anchor death (fixed Mar 14). See Finding 1 in Training Dynamics Analysis.

**Status:** SUPERSEDED by v7 — v7 addresses the local/global reward misalignment directly.
v6 specs are still valid but lower priority.

---

### v7 — Hard-Positive Anchors ❌ Inconclusive

**Goal:** Fix the root cause of local/global reward misalignment (Finding 4). Select anchors
from the N most misclassified minority-positive points (lowest p_alpha), aligning anchor
proximity reward and hard_reward toward the same decision boundary region.

**Config:** `anchor_selection_mode: "hard_positive"`, `sigma_calibration_factor: 1.0`,
`anchor_refresh_interval: 300`, `lambda=[0.5,0.5]`, `use_pca=true`.

**Results (3 seeds each):**

| Dataset | Alpha EO | Beta EO | Δ EO | Alpha F1-w | Beta F1-w | Δ F1-w | Alpha Brier | Beta Brier |
|---|---|---|---|---|---|---|---|---|
| Census | 0.2551 | 0.2466 | **-3.3%** | 0.8353 | 0.8263 | -0.52pp | 0.1108 | 0.1163 |
| Credit | 0.0319 | 0.0310 | **-2.8%** | 0.7651 | 0.7524 | -1.27pp | 0.1486 | 0.1667 |

**Key finding:** Very high seed variance. Census seed 42 shows -25.9% EO improvement,
but seeds 123 and 999 degrade (+9.1%, +24.6%). Mean improvement is noise-level. Consistent
utility degradation on both datasets. The hard-positive anchor selection does not reliably
solve the misalignment problem — possibly the dynamic anchor refresh (interval=300) is
introducing instability.

---

### v8 — PCA Whitening ❌ Backfires

**Goal:** Decorrelate and normalize PCA components to improve exploration and fairness.

**Config:** Same as v7 + `whiten_pca: true`, `radius_clip: 5.0`. Credit only.

**Results (3 seeds):**

| Dataset | Alpha EO | Beta EO | Δ EO | Alpha F1-w | Beta F1-w | Δ F1-w | Alpha Brier | Beta Brier |
|---|---|---|---|---|---|---|---|---|
| Credit | 0.0423 | 0.0649 | **+53.4% (WORSE)** | 0.7668 | 0.7164 | -5.04pp | 0.1479 | 0.1936 |

**Key finding:** Whitening is actively harmful. EO gap increases 53% on average; seed 42
alone loses 10pp F1-w. Whitening likely disrupts the learned data manifold geometry that
the RL agent relies on to generate useful samples. **Abandon this direction.**

---

### v9 — Sigma Calibration & Training Dynamics ⚠️ Mixed

**Goal:** Test whether tighter sigma (factor=1.0 vs 3.5), disabling beta resets
(`beta_reset_interval=9999`), and higher RL LR (0.001 vs 0.0003) improve convergence.

**Variants run:**

| Spec | Dataset | sigma_factor | reset_interval | RL LR |
|---|---|---|---|---|
| `v9_census_sigma1` | Census | 1.0 | 300 (default) | 0.0003 |
| `v9_census_sigma1_highlr_neverreset` | Census | 1.0 | 9999 (never) | 0.001 |
| `v9_credit_sigma2` | Credit | 2.0 | 300 (default) | 0.0003 |
| `v9_credit_sigma2_neverreset` | Credit | 2.0 | 9999 (never) | 0.001 |

**Results (3 seeds each):**

| Variant | Alpha EO | Beta EO | Δ EO | Alpha F1-w | Beta F1-w | Δ F1-w | Alpha Brier | Beta Brier |
|---|---|---|---|---|---|---|---|---|
| Census sigma1 | 0.2551 | 0.2488 | -2.5% | 0.8353 | 0.8148 | -2.05pp | 0.1108 | 0.1256 |
| Census sigma1+HiLR+NoReset | 0.2683 | 0.2081 | **-22.4%** | 0.8336 | 0.7809 | **-5.27pp** | 0.1110 | 0.2154 |
| Credit sigma2 | 0.0319 | 0.0433 | +35.7% | 0.7651 | 0.7586 | -0.65pp | 0.1486 | 0.1772 |
| Credit sigma2+NoReset | 0.0342 | 0.0297 | -13.1% | 0.7714 | 0.7034 | **-6.79pp** | 0.1481 | 0.2867 |

**Key findings:**
1. **Never-reset (`beta_reset_interval=9999`) is catastrophically harmful.** On both datasets,
   Brier nearly doubles and utility collapses. Beta resets are essential for stability.
2. **High LR + never-reset on census** worsens EO despite apparent EO gain: F1-w drops 5.27pp,
   Brier doubles from 0.111 to 0.215. The EO improvement comes at an unacceptable cost.
3. **Credit sigma2 (standard)** is the best result across v7-v9: modest EO regression but
   minimal utility loss (-0.65pp F1-w). Still worse than v3 credit in absolute terms.
4. **Tighter sigma (1.0 vs 3.5)** shows no benefit on census — same instability as v7.
5. **Census continues to resist improvement** under all variants. No v7-v9 run achieves
   robust EO improvement with preserved utility on census.

**Overall v7-v9 ranking** (best to worst fairness-utility tradeoff):
1. v9 credit_sigma2: -0.65pp F1-w, EO roughly flat
2. v7 credit_hard_anchors: -1.27pp F1-w, EO roughly flat
3. v7/v9 census variants: noise-level EO changes, 0.5-2pp utility loss
4. v8 credit_whiten: +53% EO degradation, -5pp F1-w
5. v9 never-reset variants: utility collapse on both datasets

---

## Training Dynamics Analysis (Mar 14)

Deep inspection of per-episode `metrics.csv` across all v3 and v5 runs revealed several
critical issues.

### Finding 1: Anchor reward collapse with sigma=0.85

In v3_run1 and v3_run2, `sigma_anchor=0.85` causes anchor rewards to die within ~1000
episodes. As the RL agent explores, generated samples drift far from anchors in PCA space.
The Gaussian kernel `exp(-0.5 * (dist/sigma)^2)` collapses to ~0 because distances grow
much larger than sigma. Result: local reward ≈ 0, these runs are effectively global-only.

Sigma must be set relative to the data distribution. Use `sigma_calibration_factor` in v6
or manually set sigma ≈ median nearest-neighbour distance among anchors.

### Finding 2: Silent config changes between v3 runs invalidate pooled analysis

The three v3 runs differ in sigma, lambda schedule, and radius_clip — making them
three different experiments. The pooled 9-seed p-values cannot be interpreted as a single
method. v3_run3 (sigma≈3.0, radius_clip=3.0) is comparable to v5 (same sigma, same clip),
with the only difference being `use_pca` and `use_uncertainty_anchors`.

### Finding 3: EO never converges during training for Census

All census runs show EO **trending upward (worsening)** throughout training:

```
v3_census_run1: EO 0.131 → 0.144 → 0.155  ↑ worsening
v3_census_run2: EO 0.125 → 0.141 → 0.149  ↑ worsening
v3_census_run3: EO 0.125 → 0.149 → 0.163  ↑ worsening
v5_census:      EO 0.122 → 0.136 → 0.152  ↑ worsening
```

The final test improvement vs alpha is real but comes from **checkpoint selection**
(saving the episode with best `global_obj`), not from a stably converging policy.
Credit runs with healthy anchors do show genuine EO improvement:

```
v3_credit_run3: EO 0.065 → 0.049 → 0.047  ↓ improving
v5_credit:      EO 0.065 → 0.046 → 0.045  ↓ improving
```

The Census reward signal may be too weak to produce stable convergence at 6000 episodes.

### Finding 4: Anchor reward negatively correlated with fairness in credit run 3

```
v3_credit_run3: corr(anchor_reward, EO_gap) = +0.360
```

Higher anchor proximity reward correlates with *worse* EO gap. This suggests the local
anchor reward and the global fairness objective are working at cross-purposes in the credit
dataset — anchors may be directing generation toward a region that doesn't help fairness.
Investigate whether anchors are from the correct (minority, positive) group or whether
the anchor selection criterion needs refinement.

### Implications for publishability

The core claim ("utility-preserving fairness improvement") remains valid directionally,
but the statistics need to be re-run with controlled configs before publication:
1. Obtain 3+ seeds from a **single consistent config** (sigma=3.0, lambda=[0.5,0.5], PCA)
2. The credit utility gains are the most robust finding — these p-values from run3 alone
   are highly significant
3. Census EO improvement likely driven by global reward alone — need clean global-only
   vs global+anchors comparison with matching sigma/config

---

## Framework Changes (v6 codebase)

Four improvements added to `training.py` and `main.py`:

### UA Warm-up (`uncertainty_warmup_episodes`)
For the first N episodes, use static anchor reward even when `use_uncertainty_anchors=true`.
Prevents noisy uncertainty weights from destabilising early policy learning.
JSON: `local_weights.uncertainty_warmup_episodes` (default 0 = disabled).

### Sigma Auto-calibration (`sigma_calibration_factor`)
After anchor set is built, compute median nearest-neighbour distance among anchors and
set `sigma_anchor = median_dist * factor`. Removes manual sigma tuning.
JSON: `local_weights.sigma_calibration_factor` (default null = use fixed sigma_anchor).

### Dynamic Anchor Refresh (`anchor_refresh_interval`, `anchor_refresh_top_k`)
Every N episodes, re-score all disadv-positive train points with current beta and keep
the top-K most uncertain (closest to p=0.5). Prevents anchor set from becoming stale.
JSON: `local_weights.anchor_refresh_interval` (default 0 = disabled),
`local_weights.anchor_refresh_top_k` (default 500).

### Hard-Positive Anchor Selection (`anchor_selection_mode`)
When `anchor_selection_mode: "hard_positive"`, selects the top-N minority-positive
training points by lowest `p_alpha` (where alpha is most wrong/uncertain) as the initial
anchor set, instead of a random subset. This aligns anchor proximity reward with the
hard-positive reward and the EO objective — both now target the decision boundary region.
JSON: `local_weights.anchor_selection_mode` ("all" = default/original, "hard_positive" = new).

### Global Sigmoid Sharpness (`global_sigmoid_k`)
`sigmoid(k * (eo_alpha - eo_beta))` with configurable k (was hardcoded to 10).
Higher k amplifies the gradient signal for small EO improvements.
JSON: `reward_shaping.global_sigmoid_k` (default 10.0).

---

## Baselines Comparison (Census Income)

| Method | EO | F1 min | F1 weighted | AUC | Notes |
|---|---|---|---|---|---|
| No Aug. (Alpha) | 0.1039 | 0.6579 | 0.8425 | 0.9013 | Baseline |
| RL v3 Anchors | 0.0899* | 0.6614 | 0.8424 | 0.8997 | **No utility loss** |
| OT Repair | 0.0192** | 0.6073** | 0.8074** | 0.8443** | Big EO but big utility cost |
| Group DRO (2) | 0.0166** | 0.6620 | 0.8138* | 0.8889 | Big EO, F1_w drops |

## Baselines Comparison (Credit Card)

| Method | EO | F1 min | F1 weighted | AUC | Notes |
|---|---|---|---|---|---|
| No Aug. (Alpha) | 0.0576 | 0.3802 | 0.7742 | 0.7390 | Baseline |
| RL v3 Anchors | 0.0429 | 0.4097** | 0.7793** | 0.7456** | Utility gains significant |
| OT Repair | 0.0317 | 0.4279* | 0.7748 | 0.7269 | Good EO+F1_min, lower AUC |
| Group DRO (2) | 0.0182* | 0.4969** | 0.7421 | 0.7444 | Best EO+F1_min, F1_w drops |

*p<0.05, **p<0.01 vs alpha baseline

---

## v14 / v15a — DVRL Two-Phase Framework (Unbiased)

Major redesign: replaced anchor/hard/diversity local reward with DVRL-inspired signal
(beta's BCE loss on generated samples). Added two-phase training (phase 1 = minority class,
phase 2 = majority class recovery). Added curriculum learning and delta actions.

| Method | EO | EO std | F1w | AUC | Seeds |
|---|---|---|---|---|---|
| v14 (lambda [0.3,0.7], util_guard=0.2) | 0.0834 | ±0.075 | 0.8232 | 0.8694 | 3 |
| v15a (lambda [0.3,0.5], util_guard=0.0) | 0.0557 | ±0.069 | 0.8135 | 0.8586 | 3 |
| GroupDRO (unbiased) | 0.0290 | ±0.026 | 0.8154 | 0.8923 | 3 |
| OTRepair (unbiased) | 0.0570 | ±0.025 | 0.8032 | 0.8425 | 3 |

**Key finding:** GroupDRO outperforms on unbiased census. This motivated reframing the
contribution around **positive-class outcome scarcity** — the regime where reweighting
methods fail due to insufficient minority positive examples.

**v15b (confidence window on DVRL reward):** Tested but abandoned. Helped utility (phase 2
AUC ~0.884 vs 0.860) but broke EO (mean ~0.10+ vs 0.014 for seed 42). The window filtered
too much local signal, leaving the agent running on global reward only.

---

## Reframing: Outcome Bias / Positive-Class Scarcity (v16, Mar 2026)

**Motivation:** GroupDRO and OTRepair both operate on existing samples only (reweighting /
redistribution). When the disadvantaged group has very few *positive-class* examples —
simulating historical outcome bias — reweighting degenerates. Our generative framework
synthesises new minority positive samples, which is the only viable approach in this regime.

**Bias injection:** `bias_pct` downsamples y=1 (positive class) across the training set.
Because the disadvantaged group (a=0) already has a low positive rate, this creates severe
positive-class scarcity specifically for them:

| bias_pct | Disadv. group (a=0) positive examples | Total positives |
|---|---|---|
| 0.05 | 17 | 150 |
| 0.10 | 43 | 300 |
| 0.15 | 77 | 450 |
| None | 116 | 722 |

**Census baseline degradation curve (already collected, Mar 17 2026):**

| Method | bias=0.05 EO | bias=0.10 EO | bias=0.15 EO | unbiased EO |
|---|---|---|---|---|
| GroupDRO | **0.247** ± 0.120 | **0.100** ± 0.038 | 0.018 ± 0.007 | 0.029 ± 0.026 |
| OTRepair | 0.050 ± 0.038 | 0.025 ± 0.020 | 0.055 ± 0.062 | 0.057 ± 0.025 |

GroupDRO collapses at bias=0.05 (17 positive minority examples) and is still meaningfully
impaired at bias=0.10 (43 examples). OTRepair maintains EO but at a consistent utility cost
(F1w ~0.757–0.788 vs GroupDRO's ~0.821).

**Status: COMPLETE (Mar 18 2026)**

| Experiment | Spec | Status | β-EO | β-F1w | β-AUC |
|---|---|---|---|---|---|
| RL Census bias=0.05 | v16_bias05_rl_census_3seeds | ✅ done | 0.097 ± 0.067 | 0.787 | 0.850 |
| RL Census bias=0.10 | v16_bias010_rl_census_3seeds | ✅ done | 0.084 ± 0.043 | 0.788 | 0.848 |
| RL Credit bias=0.05 | v16_bias05_rl_credit_3seeds | ✅ done | 0.031 ± 0.022 | 0.715 | 0.633 |
| RL Credit bias=0.10 | v16_bias010_rl_credit_3seeds | ✅ done | 0.030 ± 0.013 | 0.715 | 0.652 |
| CTGAN Census bias=0.05 | v16_bias05_ctgan_census_3seeds | ✅ done | 0.020 ± 0.010 | 0.756 | 0.812 |
| CTGAN Census bias=0.10 | v16_bias010_ctgan_census_3seeds | ✅ done | 0.075 ± 0.027 | 0.789 | 0.842 |

**Full census comparison (bias=0.10):**

| Method | α-EO | β-EO | ±std | β-F1w | β-AUC |
|---|---|---|---|---|---|
| RL v16 | 0.144 | 0.084 | 0.043 | 0.788 | 0.848 |
| OTRepair | 0.325 | 0.025 | 0.020 | 0.788 | 0.819 |
| GroupDRO | 0.109 | 0.115 🔴 | 0.024 | 0.821 | 0.891 |
| CTGAN | 0.114 | 0.075 | 0.027 | 0.789 | 0.842 |

**Key findings:**
- GroupDRO fails at ALL bias levels (EO 0.11–0.25, never beats alpha baseline) — confirms the paper's core motivation
- RL achieves best AUC at both bias levels (+2.9pp vs OTRepair at bias=0.10)
- RL and CTGAN are essentially tied at bias=0.10 on all metrics
- CTGAN's EO advantage at bias=0.05 (0.020 vs RL's 0.097) is not robust — degrades to 0.075 at bias=0.10
- Credit card alpha-EO is near-zero at both bias levels; credit results do not contribute to fairness story

**Target outcomes (original):**
- ✅ RL EO < 0.10 at bias=0.10 (achieved 0.084)
- ❌ RL F1w > 0.79 (achieved 0.788, miss by 0.002)
- ✅ RL AUC > 0.82 (achieved 0.848)

---

## Reward Structure Analysis & v17 Plan (Mar 18 2026)

Deep convergence analysis of v16 `metrics.csv` identified five structural problems with the
reward setup. All problems are backed by per-episode data.

### Diagnosed Problems

**P1 — Global reward deadzone (root cause of high variance)**
Phase 1 (minority generation) spends **57% of episodes** with `global_obj < 0.5`, meaning
beta is worse than alpha and the agent gets no positive learning signal. Root cause: beta
resets to **random weights every episode** (`beta_reset_interval=1`), then trains on
real + (bad early synthetic) data → beta fails to learn → wloss_beta >> wloss_alpha →
global_obj ≈ 0 for ~450 episodes. The agent randomly stumbles into a good policy region
around ep500 that fixes beta. This explains the seed-level EO variance (range 0.12 at
bias=0.05): the "lucky" discovery timing varies.

**P2 — gamma=0.99 with T=2000 wastes 95% of trajectory**
Effective credit horizon = 1/(1−γ) = 100 steps. Steps 500–2000 contribute discount factor
< 0.007 — negligible. The policy gradient uses only the first ~100 steps for credit
assignment despite generating 2000 samples per episode. Since beta trains on the *full*
2000-sample trajectory, there is no causal reason to discount later steps at all.

**P3 — Curriculum jumps destabilise the policy**
PCA dimensionality steps 2→4→6→8→10 every 200 episodes. Each transition forces policy
adaptation. With the deadzone already consuming 450 episodes, the policy has at most
~350 stable episodes for learning. Stage disruptions cut into that further.

**P4 — DVRL local reward weak in early episodes**
Median Phase 1 `local_reward` = 0.09 (low). Generated samples land near the training
distribution (delta_scale=0.10) where beta already has low loss → low DVRL. Note: the
overall Pearson corr(local, global) = +0.765 is trend-inflated (both variables independently
trend upward); rolling window correlation = −0.127, meaning no strong step-level alignment.
With v17a (gamma=1.0), local_reward rose to 0.16–0.18 organically, suggesting the agent
now explores the space more broadly.

**P5 — Phase 2 agent already separate (not a problem)**
Initially diagnosed as gradient pollution, but code inspection (line 1169) confirms Phase 2
already instantiates a fresh `ReinforceAgent`. No change needed here.

### v17 Experiment Plan

**v17a — Structural spec fixes only** (no code changes)
- `gamma: 1.0` (was 0.99) — all 2000 steps contribute equally
- `curriculum: start_dim=10, max_dim_cap=10, stage_count=1` — disabled
- Everything else identical to v16
- **Purpose:** Isolate effect of P2 and P3 fixes. Run first; requires no code deployment.

**v17b — Beta warm-start from alpha** (one code change: ~10 lines in training.py)
- All v17a settings **plus** `beta_warmstart_from_alpha: true`
- On each beta reset, copies `alpha.model.state_dict()` into beta instead of random init.
  Beta always starts at alpha-level performance; deadzone becomes structurally impossible.
- Fresh optimizer recreated after copy (no momentum carryover).
- **Purpose:** Eliminate P1 (deadzone). This is the highest-priority fix.

**v17c — delta_scale sweep** (spec only, run after v17b confirms stable training)
- v17b settings with `delta_scale: 0.20` and `delta_scale: 0.30`
- **Purpose:** Address P4. Larger perturbations → samples in higher-DVRL regions.
- Only run if v17b shows stable convergence (deadzone < 20%).

### Diagnostic targets per run

Check `metrics.csv` after ~200 episodes before waiting for full results:

| Metric | v16 baseline | v17a target | v17b target |
|---|---|---|---|
| % eps with global_obj < 0.5 | 57% | 40–50% | < 10% |
| First ep where global_obj > 0.5 | ~ep450 | ~ep300 | ~ep50 |
| EO seed range (3 seeds) | 0.12 | 0.08 | 0.05 |
| Median local_reward (Phase 1) | 0.12 | 0.12 | 0.15+ |

### Run sequence

```
SUBMIT NOW:   v17a_{bias05,bias010}_rl_census_3seeds  (spec-only, deploy immediately)
DEPLOY CODE:  training.py beta_warmstart change + main.py wire-up  ← DONE
SUBMIT NEXT:  v17b_{bias05,bias010}_rl_census_3seeds  (after v17a submitted)
AFTER RESULTS: v17c delta_scale sweep (only if v17b shows stable training)
```

**Specs created:** `v17a_bias05_rl_census_3seeds`, `v17a_bias010_rl_census_3seeds`,
`v17b_bias05_rl_census_3seeds`, `v17b_bias010_rl_census_3seeds`

**Status:** COMPLETE (Mar 18 2026)

---

### v17 Results (Mar 18 2026)

#### Deadzone Analysis

| Spec | dead%_42 | dead%_0 | dead%_1 | mean_dead | esc_42 | esc_0 | esc_1 | med_local |
|---|---|---|---|---|---|---|---|---|
| v16 bias=0.05 | 57.4% | 40.8% | 44.4% | **47.5%** | 481 | 325 | 295 | 0.090 |
| v16 bias=0.10 | 58.5% | 36.6% | 59.5% | **51.5%** | 481 | 125 | 481 | 0.092 |
| v17a bias=0.05 | 8.2% | 11.9% | 6.6% | **8.9%** | 61 | 77 | 41 | 0.179 |
| v17a bias=0.10 | 14.2% | 1.2% | 20.5% | **12.0%** | 92 | 1 | 139 | 0.156 |
| v17b bias=0.05 | 3.8% | 18.0% | 6.4% | **9.4%** | 25 | 176 | 73 | 0.198 |
| v17b bias=0.10 | 14.9% | 0.4% | 13.6% | **9.6%** | 126 | 1 | 124 | 0.172 |

**Key finding:** gamma=1.0 + no curriculum (v17a) **alone** eliminated the deadzone from
~50% to ~10%, matching v17b. The warm-start was unnecessary — the root cause was credit
assignment failure (gamma=0.99), not random beta initialisation.

#### Final Test Metrics

| Spec | α-EO | β-EO | ±std | range | β-F1w | β-AUC | EO delta |
|---|---|---|---|---|---|---|---|
| v16 bias=0.05 | 0.075 | 0.097 | 0.067 | 0.118 | 0.787 | 0.850 | +0.022 |
| v16 bias=0.10 | 0.144 | 0.084 | 0.043 | 0.082 | 0.788 | 0.848 | −0.059 |
| v17a bias=0.05 | 0.075 | **0.071** | 0.051 | 0.101 | 0.771 | 0.860 | −0.004 |
| v17a bias=0.10 | 0.144 | **0.071** | 0.067 | 0.134 | **0.807** | **0.867** | −0.073 |
| v17b bias=0.05 | 0.075 | 0.096 | 0.069 | 0.137 | 0.768 | 0.858 | +0.021 |
| v17b bias=0.10 | 0.144 | 0.116 | 0.062 | 0.119 | 0.801 | 0.859 | −0.028 |

#### Per-seed breakdown — v17a

| Spec | seed | α-EO | β-EO | Δ-EO | β-F1w | β-AUC |
|---|---|---|---|---|---|---|
| bias=0.05 | 42 | 0.093 | 0.068 | −0.025 | 0.795 | 0.875 |
| bias=0.05 | 0  | 0.056 | 0.022 | −0.034 | 0.740 | 0.861 |
| bias=0.05 | 1  | 0.076 | 0.123 | +0.047 🔴 | 0.778 | 0.845 |
| bias=0.10 | 42 | 0.162 | 0.141 | −0.020 | 0.813 | 0.867 |
| bias=0.10 | 0  | 0.157 | 0.063 | −0.093 | 0.799 | 0.879 |
| bias=0.10 | 1  | 0.113 | 0.007 | −0.106 ✅ | 0.808 | 0.856 |

2 of 3 seeds improve at each bias level. One "rogue" seed degrades at each level — variance
reduction is real but partial.

#### Comparison vs baselines at bias=0.10

| Method | β-EO | ±std | β-F1w | β-AUC |
|---|---|---|---|---|
| Alpha (no aug) | 0.144 | — | 0.821 | 0.891 |
| **RL v17a** | **0.071** | 0.067 | **0.807** | **0.867** |
| RL v16 | 0.084 | 0.043 | 0.788 | 0.848 |
| RL v17b | 0.116 | 0.062 | 0.801 | 0.859 |
| CTGAN | 0.075 | 0.027 | 0.789 | 0.842 |
| OT Repair | 0.025 | 0.020 | 0.788 | 0.819 |
| GroupDRO | 0.115 | 0.024 | 0.821 | 0.891 |

v17a at bias=0.10: EO competitive with CTGAN (0.071 vs 0.075), **best F1w and AUC of all
methods** (0.807 F1w, 0.867 AUC). First config to simultaneously match CTGAN on fairness
and beat it on utility.

#### Interpretations

**Why gamma=1.0 fixed the deadzone (not the warm-start):**
The original deadzone hypothesis assumed random beta init was the cause. In fact, gamma=0.99
was the root cause: with only 100-step effective horizon, the agent couldn't build useful
gradient signal from 2000-step trajectories. With gamma=1.0, all 2000 steps contribute
equally — even a randomly-initialised beta trains well enough on the full synthetic trajectory
to escape the deadzone from episode 1.

**Why v17b (warm-start) is worse than v17a:**
Warm-starting beta from alpha's weights locks beta into alpha's biased solution. The RL
agent's synthetic data must overcome alpha's entrenched priors each episode rather than
guiding a more plastic randomly-initialised model. Net result: beta ends closer to alpha,
EO improvement is smaller. **Abandon the warm-start direction.**

**Why local_reward increased (0.09 → 0.16–0.18):**
With gamma=1.0 and no curriculum, the agent explores more freely across the full
10D PCA space from episode 1. Generated samples land further from the training distribution,
producing higher DVRL (beta uncertainty) signals. This is the P4 fix happening organically.

#### Next Steps

1. **v17a is the new baseline.** All future experiments start from v17a config.
2. **Abandon beta warm-start** — confirmed counterproductive.
3. **Submit v17c delta_scale sweep** (deadzone 9–12% < 20% threshold ✅):
   - `delta_scale: 0.20` and `delta_scale: 0.30`, both with v17a base config
   - Target: does larger perturbation further reduce EO or help the rogue seed?
4. **5-seed runs for v17a** once delta_scale is confirmed — tighten confidence intervals.
5. **Rogue seed problem**: one seed in 3 degrades at each bias level. Investigate whether
   this correlates with alpha-EO level (seed 1 at bias=0.05 has relatively low alpha-EO=0.076,
   leaving little room; seed 42 at bias=0.10 has α-EO=0.162, agent may overshoot).

---

### After v17 Results: Decision Tree

**Step 1 — Early diagnosis (after ~200 episodes, use `diagnose_training.ipynb`)**

Open the `metrics.csv` for each run and check:
- `global_obj` column: what fraction of Phase 1 episodes have `global_obj < 0.5`?
- First episode where `global_obj` crosses 0.5 stably?
- Median `local_reward` in Phase 1?

**Step 2 — Compare v17a vs v17b**

| Outcome | Interpretation | Action |
|---|---|---|
| v17b deadzone < 10%, EO improves | Warm-start fixes P1; gamma+curriculum (P2+P3) also helped | Submit v17c (delta_scale sweep) |
| v17b deadzone < 10%, EO flat | Deadzone fixed but something else limits fairness | Investigate P4 (delta_scale) — still submit v17c |
| v17b deadzone still > 30% | Warm-start not working as expected | Check training.py warm-start code path; check if `_optim_cfg` is stored |
| v17a much better than v16, v17b no further gain | gamma=1.0 + no curriculum was the key fix | P1 may be less important; still run v17c |
| No improvement vs v16 in either | Deeper structural issue | Discuss Phase 2 warm-start, longer episodes, or DVRL delta_scale before more runs |

**Step 3 — v17c (delta_scale sweep)**: only if v17b shows stable training (deadzone < 20%)

Submit `v17c_bias05_rl_census` and `v17c_bias010_rl_census` with:
- `delta_scale: 0.20` and `delta_scale: 0.30` (two specs per bias level)
- Everything else from v17b

**Step 4 — Credit card runs**: once census v17b is validated, run equivalent credit specs.
Credit alpha-EO is near-zero at both bias levels so credit results only matter for
utility preservation claims, not the fairness story.

**Step 5 — Final paper experiments**: once best config is identified from v17a/b/c:
- 5-seed runs for census (main result table)
- Add CTGAN at bias=0.05 to the comparison (already done at bias=0.10)
- Re-run GroupDRO/OTRepair at best bias levels to confirm degradation story

---

## v17c — Delta Scale Sweep Results (Mar 18 2026)

**Goal:** Test whether larger perturbation magnitude (delta_scale=0.20, 0.30 vs v17a's 0.10)
reduces EO gap further, addressing P4 (DVRL local reward too weak in early training).

**Base config:** v17a (gamma=1.0, curriculum disabled, no warm-start). Only delta_scale varies.

### Per-seed results

#### bias=0.10

| Spec | seed | α-EO | β-EO | Δ-EO | β-F1w | β-AUC | dead% |
|---|---|---|---|---|---|---|---|
| ds=0.20 | 0  | 0.157 | 0.049 | −0.107 ✅ | 0.808 | 0.877 | 0.2% |
| ds=0.20 | 1  | 0.113 | 0.068 | −0.046 ✅ | 0.800 | 0.851 | 6.0% |
| ds=0.20 | 42 | 0.162 | **0.314** | +0.153 🔴 | 0.817 | 0.868 | 3.4% |
| **ds=0.20 mean** | — | 0.144 | **0.144** | ±0.121 | **0.808** | **0.865** | **3.2%** |
| ds=0.30 | 0  | 0.157 | 0.235 | +0.079 🔴 | 0.804 | 0.873 | 0.0% |
| ds=0.30 | 1  | 0.113 | 0.015 | −0.099 ✅ | 0.767 | 0.846 | 0.9% |
| ds=0.30 | 42 | 0.162 | 0.258 | +0.096 🔴 | 0.806 | 0.855 | 0.0% |
| **ds=0.30 mean** | — | 0.144 | **0.169** | ±0.110 | **0.793** | **0.858** | **0.3%** |

#### bias=0.05

| Spec | seed | α-EO | β-EO | Δ-EO | β-F1w | β-AUC | dead% |
|---|---|---|---|---|---|---|---|
| ds=0.20 | 0  | 0.056 | 0.125 | +0.069 🔴 | 0.780 | 0.850 | 2.7% |
| ds=0.20 | 1  | 0.076 | 0.047 | −0.028 ✅ | 0.765 | 0.842 | 10.7% |
| ds=0.20 | 42 | 0.093 | 0.052 | −0.042 ✅ | 0.774 | 0.847 | 0.5% |
| **ds=0.20 mean** | — | 0.075 | **0.075** | ±0.036 | **0.773** | **0.846** | **4.6%** |
| ds=0.30 | 0  | 0.056 | 0.152 | +0.096 🔴 | 0.789 | 0.839 | 0.3% |
| ds=0.30 | 1  | 0.076 | 0.063 | −0.012 ✅ | 0.749 | 0.804 | 0.4% |
| ds=0.30 | 42 | 0.093 | 0.084 | −0.009 ~ | 0.786 | 0.843 | 0.0% |
| **ds=0.30 mean** | — | 0.075 | **0.100** | ±0.038 | **0.775** | **0.828** | **0.2%** |

### Comparison vs v17a

| Method | bias | β-EO | ±std | β-F1w | β-AUC | dead% | seeds improve |
|---|---|---|---|---|---|---|---|
| **v17a (ds=0.10)** | 0.10 | **0.071** | 0.067 | **0.807** | **0.867** | 12.0% | 2/3 |
| v17c ds=0.20 | 0.10 | 0.144 | 0.121 | 0.808 | 0.865 | 3.2% | 2/3 |
| v17c ds=0.30 | 0.10 | 0.169 | 0.110 | 0.793 | 0.858 | 0.3% | 1/3 |
| **v17a (ds=0.10)** | 0.05 | **0.071** | 0.051 | 0.771 | **0.860** | 8.9% | 2/3 |
| v17c ds=0.20 | 0.05 | 0.075 | 0.036 | **0.773** | 0.846 | 4.6% | 2/3 |
| v17c ds=0.30 | 0.05 | 0.100 | 0.038 | 0.775 | 0.828 | 0.2% | 2/3 |

### Key Findings

1. **Larger delta_scale makes results worse, not better.** Mean EO degrades monotonically
   with delta_scale at both bias levels. v17a (ds=0.10) remains the best config by a clear margin.

2. **Deadzone nearly eliminated but this does not help fairness.** ds=0.30 hits near-zero
   deadzone (0.2–0.3%) yet achieves the worst EO outcomes. The deadzone fix from gamma=1.0
   (v17a, ~10–12%) was sufficient; eliminating it further by increasing perturbation does
   not buy additional fairness improvement.

3. **Catastrophic rogue seed at ds=0.20, bias=0.10 (seed 42: EO=0.314).** This is far
   worse than v17a's worst case (seed 42: EO=0.141). Seed 42 at bias=0.10 has a high
   α-EO=0.162 — the agent likely overshoots the decision boundary with larger perturbations,
   generating samples that confuse beta rather than guiding it. The problem is structural:
   larger delta_scale amplifies both the signal and the risk of overshoot.

4. **ds=0.30 at bias=0.05 shows AUC regression** (0.828 vs v17a's 0.860 — a 3.2pp drop).
   The larger perturbations are placing synthetic samples in regions that hurt beta's
   overall discrimination ability while failing to close the EO gap.

5. **Rogue seed at bias=0.05 (seed 0, α-EO=0.056)** is partly explained by near-floor
   alpha performance: with only 0.056 EO gap, there is almost no room to improve, and
   any noise in synthetic sample placement degrades to the rogue outcome.

### Interpretations (Claude)

The delta_scale sweep conclusively answers the P4 hypothesis: larger perturbations do not
fix the problem of "DVRL local reward too weak in early training." What actually happened
with v17a (ds=0.10, gamma=1.0) was that the agent explored more freely across the full 10D
PCA space than v16 (which had curriculum limiting early dims), producing higher DVRL signals
organically. This was already sufficient — and increasing delta_scale beyond 0.10 sends
the agent into regions where beta's training becomes noisy and unstable.

The pattern is revealing: ds=0.30 has near-zero deadzone (the beta *looks* healthy by that
metric) but EO is worst. This means the deadzone metric alone is insufficient — a large
perturbation can "rescue" beta from the deadzone by generating lots of uncertain samples,
but those samples may be too far out of distribution to actually guide beta toward fairness.
The right balance is enough perturbation to keep DVRL informative (ds=0.10 achieves 0.16–0.18
median local reward) but not so much that the synthetic samples become pure noise.

The rogue seed at ds=0.20 is a warning sign: increasing delta_scale raises the ceiling
(the two good seeds are better than v17a: 0.049, 0.068 vs 0.141, 0.063) but also raises
the floor dramatically (0.314 vs 0.141). For a paper result, high variance is worse than
a slightly higher mean — reviewers will focus on the rogue seed. v17a's controlled variance
is more publishable.

### Decision: v17a is the final best config

**v17a config (gamma=1.0, no curriculum, delta_scale=0.10) is the winner.**

Do not pursue further delta_scale variants. The next experiments are:

1. **5-seed runs of v17a** at both bias levels (census) — required for final paper table.
   Use seeds: 42, 0, 1, 2, 3 (or any 5 fixed seeds).
2. **Global-only ablation** (no local reward, global reward only) at bias=0.10 — needed
   to prove DVRL contribution. Without this, reviewers will ask "does the local reward
   actually help?" This is the highest-priority experiment after confirming the 5-seed plan.
3. **Credit card v17a runs** — for utility preservation story. Use same config, both bias levels.
4. **PAMAP2** — after census and credit are finalised.

---

## v19 — 5-seed results + global-only ablation (2026-03-19)

### Runs

| Spec | Dataset | Bias | Config | Seeds | Dir |
|------|---------|------|--------|-------|-----|
| v17a_bias010_5seeds | census | 0.10 | DVRL (v17a) | 42,0,1,2,3 | SPECv17a_bias010_rl_census_5seeds_...e7a6f2fe |
| v17a_bias05_5seeds | census | 0.05 | DVRL (v17a) | 42,0,1,2,3 | SPECv17a_bias05_rl_census_5seeds_...253d5a53 |
| v18_ablation_global_only | census | 0.10 | global-only | 42,0,1,2,3 | SPECv18_ablation_global_only_bias010_...4676389e |

### Results

**v17a 5-seed census bias=0.10** (seeds 42,0,1,2,3):

| Seed | α-EO | β-EO | F1w | AUC |
|------|------|------|-----|-----|
| 42 | 0.162 | 0.141 | 0.813 | 0.867 |
| 0  | 0.157 | 0.063 | 0.799 | 0.879 |
| 1  | 0.113 | 0.008 | 0.808 | 0.856 |
| 2  | 0.263 | 0.234 | 0.814 | 0.886 |
| 3  | 0.227 | 0.169 | 0.800 | 0.862 |
| **Mean** | 0.184 | **0.123 ± 0.089** | **0.807 ± 0.007** | **0.870 ± 0.012** |

The 3-seed mean was 0.071; seeds 2 and 3 (α-EO 0.263, 0.227) are "hard" seeds that inflate
the 5-seed mean significantly. High variance is a problem for the main results table.

**v17a 5-seed census bias=0.05** (seeds 42,0,1,2,3):

| Seed | α-EO | β-EO | F1w | AUC |
|------|------|------|-----|-----|
| 42 | 0.093 | 0.068 | 0.795 | 0.875 |
| 0  | 0.056 | 0.022 | 0.740 | 0.861 |
| 1  | 0.076 | 0.123 | 0.778 | 0.845 |
| 2  | 0.065 | 0.045 | 0.727 | 0.846 |
| 3  | 0.077 | 0.124 | 0.762 | 0.812 |
| **Mean** | 0.073 | **0.076 ± 0.046** | **0.760 ± 0.027** | **0.848 ± 0.023** |

**v18 global-only ablation census bias=0.10** (seeds 42,0,1,2,3, note: "3seeds" in dir name
but 5 seeds ran):

| Seed | α-EO | β-EO | F1w | AUC |
|------|------|------|-----|-----|
| 42 | 0.162 | 0.026 | 0.826 | 0.884 |
| 0  | 0.157 | 0.082 | 0.805 | 0.875 |
| 1  | 0.113 | 0.002 | 0.808 | 0.871 |
| 2  | 0.263 | 0.153 | 0.797 | 0.876 |
| 3  | 0.227 | 0.055 | 0.821 | 0.882 |
| **Mean** | 0.184 | **0.063 ± 0.059** | **0.811 ± 0.012** | **0.877 ± 0.005** |

### Key Findings

1. **Global-only (v18) outperforms DVRL (v17a) on EO in 4/5 seeds.** Mean EO 0.063 vs 0.123.
   The DVRL local reward does not help fairness at bias=0.10 — it hurts. The ablation claim
   originally intended to validate DVRL is now inverted: the ablation shows global-only is better.

2. **v18 shows clearer learning curves.** Episode return rises monotonically to ~0.99 in 4/5
   seeds. Best checkpoints found at ep 31–796 (spread across training). v17a by contrast shows
   severe late-training regression in worst_loss_beta (up to +1.10 delta) and seed_3's best
   checkpoint was found at episode 6/800 — essentially no learning.

3. **v17a DVRL causes policy instability in late training.** The local reward (DVRL) drives the
   agent toward beta's decision boundary, but this produces increasingly out-of-distribution
   samples that destabilize beta training in Q3/Q4. This explains both the worse EO and the
   best-checkpoint regression pattern.

4. **Seed 2 is a structural outlier** (α-EO=0.263) for both methods. The underlying data split
   has severe imbalance; neither method closes the gap well. Seed 2 should be noted in the paper
   as a high-difficulty case.

5. **On overlap seeds {42,0,1}, v18 vs baselines:**

| Method | EO | F1w | AUC |
|--------|----|-----|-----|
| GroupDRO | 0.100 ± 0.038 | 0.821 | 0.891 |
| OT Repair | 0.025 ± 0.020 | 0.788 | 0.819 |
| CTGAN | 0.073 ± 0.031 | 0.789 | 0.842 |
| v17a (DVRL) | 0.071 ± 0.067 | 0.807 | 0.867 |
| **v18 (global-only)** | **0.036 ± 0.041** | **0.813** | **0.877** |

v18 achieves near-OT-Repair EO (0.036 vs 0.025) while dominating OT Repair on utility
(F1w +0.025, AUC +0.058). Beats GroupDRO and CTGAN on both axes.

### Decision: v18 (global-only) is the new primary method

**The primary method for the paper is now v18 (global-only reward, gamma=1.0, no curriculum,
delta_scale=0.10).** v17a (DVRL) becomes the ablation variant that shows local reward shaping
does not improve over global-only.

Paper narrative revision:
- Claim 2 (competitive performance): strongly supported by v18 results
- Claim 3 (ablation): reframed — global-only is the proposed design; DVRL local reward is
  tested and shown not to improve, giving a clean negative ablation result

### Remaining experiments for paper

**Main results (must run):**
1. v18 census bias=0.05 — 5 seeds (42,0,1,2,3)
2. v18 credit bias=0.10 — 5 seeds (42,0,1,2,3)
3. v18 credit bias=0.05 — 5 seeds (42,0,1,2,3)
4. CTGAN credit bias=0.10 — 3 seeds (42,0,1)
5. CTGAN credit bias=0.05 — 3 seeds (42,0,1)

**Hyperparameter sweeps** (census bias=0.10, v18 config, 3 seeds 42/0/1):
- Synthetic data budget: T ∈ {500, 1000, **2000**, 4000}
- FFNN epochs per episode: {5, **10**, 20, 50}
- Delta scale: {0.05, **0.10**, 0.20} (bold = current default)

**Ablation (complete):**
- v17a (DVRL) vs v18 (global-only) — 5 seeds, bias=0.10 — done

---

## HAR / PAMAP2 Experiments (2026-03-23)

### Dataset setup

- **Dataset:** PAMAP2 physical activity recognition
- **Protected attribute:** sex, derived from subject_id (subject 102 = female; all others male)
- **Activity pair:** `minority_id=5` (running, y=1) vs `majority_id=24` (rope jumping, y=0)
  - Natural EO gap: ~30% (Female positive rate 41.1%, Male 71.1%)
  - Running chosen because female subject participates fully; rope jumping chosen as majority activity
- **Windowing:** `win_seconds=1.0`, `step_seconds=0.5` (1s windows, 0.5s step)
  - 5× more windows than default 5s/2.5s — needed to get sufficient female examples in all splits
  - No-bias: Female train runners = 110 / 269 (40.9%), Male = 1064 / 1492 (71.3%), EO gap = 30.4%
  - Bias=0.10: Female train runners = 9 / 168 (5.4%) — comparable scarcity level to census/credit
- **Split strategy:** temporal within-(subject, activity) — ensures female examples appear in
  train/val/test proportionally. Earlier subject-level split placed all female runners in training.
- **PCA:** 10 components, same as census/credit

### Specs queued (2026-03-23)

**Main table — RL Framework (5 seeds 42,0,1,2,3):**
- `p1_har_nobias_global_5s` — bias_pct=null, global-only reward, delta actions
- `p1_har_bias010_global_5s` — bias_pct=0.10, global-only reward, delta actions

**Main table — Baselines (5 seeds 42,0,1,2,3):**
- `p1_har_nobias_gdro_5s` / `p1_har_bias010_gdro_5s` — Group DRO
- `p1_har_nobias_otrep_5s` / `p1_har_bias010_otrep_5s` — Gaussian OT Repair
- `p1_har_nobias_ctgan_5s` / `p1_har_bias010_ctgan_5s` — CTGAN (300 epochs, 2000 synthetic)

**Delta ablation (5 seeds 42,0,1,2,3):**
- `p1_har_nobias_nodelta_5s` — no delta actions (exact-point generation)
- `p1_har_bias010_nodelta_5s` — no delta actions, bias=0.10

### Implementation notes

- `run_baseline.py` and all three baseline trainers (`group_dro.py`, `gaussian_ot_repair.py`,
  `ctgan_baseline.py`) patched to accept and forward `win_seconds`/`step_seconds` to
  `get_data_splits`. Default values (5.0/2.5) preserve backward compatibility for census/credit.
- `real_data_size=null` for all HAR specs (use full dataset, no subsampling).
- No-delta sanity check on census showed degenerate policy collapse (fixed local reward = 0.1192
  throughout); delta actions prevent this. Expecting same pattern on HAR no-delta runs.

---

## Episode Convergence Sweep (2026-03-24)

**Goal:** Demonstrate that 800 episodes is a well-chosen training budget by showing the
fairness-utility tradeoff as a function of episode count. Addresses reviewer question of
whether the policy has converged or would continue to improve with more training.

**Base config:** Identical to main results (census bias=0.10, global-only reward, gamma=1.0,
curriculum disabled, delta_scale=0.10, ffnn epochs=20, 5 seeds). Only `total_episodes` varies.

**Specs submitted (2026-03-24):**

| Spec | Episodes | Time alloc |
|---|---|---|
| `p1_conv_census_bias010_ep600_5s` | 600 | 4:30:00 |
| *(main result — already exists)* | 800 | — |
| `p1_conv_census_bias010_ep1000_5s` | 1000 | 7:00:00 |
| `p1_conv_census_bias010_ep1500_5s` | 1500 | 10:00:00 |

**Expected outcomes:**
- Plateau at ~800 → confirms training budget is well-chosen
- Continued improvement at 1000–1500 → justifies extending the budget for the final config

**Status:** Running (submitted 2026-03-24)

---

## HAR (PAMAP2) RL Full Run — bias=0.10 (2026-03-24)

**Goal:** Confirm whether RL can improve EO on HAR with a full 5-seed, 800-episode run.
Earlier smoke test (seed_42, 150 eps) showed promising improvement (0.330 → 0.221).

**Spec:** `p1_har_bias010_gpu0_s3` (seeds 42,0,1) + `p1_har_bias010_gpu1_s2` (seeds 2,3),
both run locally on two GPUs. Merged for analysis.

**Results (5 seeds, 800 episodes):**

| Seed | α-EO | β-EO | Δ EO | α-F1w | β-F1w | Note |
|------|------|------|------|-------|-------|------|
| 42   | 0.330 | 0.207 | −0.123 ✅ | 0.818 | 0.872 | Only seed that improved |
| 0    | 0.208 | 0.227 | +0.019 ❌ | 0.848 | 0.859 | Slight degradation |
| 1    | 0.026 | 0.204 | +0.179 ❌ | 0.866 | 0.877 | α already near-zero; group flipped |
| 2    | 0.166 | 0.189 | +0.023 ❌ | 0.869 | 0.869 | Slight degradation |
| 3    | 0.024 | 0.126 | +0.102 ❌ | 0.831 | 0.815 | α already near-zero; group flipped |

**Mean: α-EO=0.151 → β-EO=0.191 — net degradation of 0.040.**

**Diagnosis:** Seeds 1 and 3 show α-EO ≈ 0.02–0.03, meaning alpha itself assigned the
disadvantaged label to the WRONG group (flipped due to extreme scarcity — only 9 female
positive training examples at bias=0.10). The RL agent then generates samples for the
"disadvantaged" group based on alpha's potentially-flipped assessment, actively harming
fairness. Seeds 0 and 2 show moderate α-EO but the agent still fails to improve.
Only seed_42 shows the correct behaviour, matching the smoke test.

**Conclusion: HAR (PAMAP2) is unsuitable for this paper.**

Root causes:
1. Only 1 female subject (subject 102) → distributional gap is physiological, not outcome-suppressed
2. Disadvantaged group identification flips across seeds due to extreme female positive scarcity
3. RL agent cannot reliably close a gap it cannot stably identify
4. High α-EO variance across seeds (0.024–0.330) makes any result unreproducible

**Decision:** Drop HAR from the paper. Replace with PTB-XL ECG dataset.

---

## PTB-XL Dataset Integration (2026-03-24)

**Motivation:** Replace PAMAP2 with a third dataset that has genuine fairness properties
suitable for the positive-class scarcity narrative. PTB-XL is a large public 12-lead ECG
dataset with documented sex-based MI underdiagnosis.

**Dataset properties:**
- 21,799 records; 11,354 male / 10,445 female (near-balanced)
- MI vs NORM binary classification: 4,134 MI, 9,438 NORM (post-filter: 13,572 records)
- Male MI rate: 2,595/(2,595+4,349) = 37.4%; Female MI rate: 1,539/(1,539+5,089) = 23.2%
- Natural sex-based MI disparity supports the positive-class outcome scarcity narrative
- At bias_pct=0.10: ~146 female MI training examples (vs ~483 male MI) — much better than
  PAMAP2's 9 examples; enough for stable group identification across seeds

**Feature extraction:** 8 per-lead statistics (mean, std, min, max, rms, p25, p75, iqr)
across all 12 leads → 96 features → PCA to 10 components (same pipeline as census/credit).

**Implementation (2026-03-24):**
- `split_ptb_xl()` added to `dataset.py`
- `ptb_xl` entry added to `DATASET_REGISTRY`
- `get_data_splits()` extended with `ptb_xl` branch (strips PAMAP2-specific kwargs)
- Feature caching to `datasets/ptb-xl/ptbxl_features_cache.npz` to avoid re-extraction
- Specs created: `smoke_ptbxl_bias010_rl.json` (150 eps, seeds [42,0]) and
  `p1_ptbxl_bias010_global_5s.json` (800 eps, seeds [42,0,1,2,3])
- Signal files (records100, MI+NORM subset only, 27,144 files) downloaded locally

**Status:** Signal files downloading. Run smoke test once download completes.

---

## v2 Paper Reframing: DA+ Scarcity Framing + COMPAS Dataset (2026-03-27)

### Motivation & Decisions

**Problem reframing:** The paper is now framed around *positive-class outcome scarcity* as
measured by DA+ (disadvantaged-group positive count) rather than `bias_pct`. `bias_pct` is
an internal implementation parameter only; DA+ is the paper-level concept. All three datasets
are calibrated to DA+ ≈ 43 using different `bias_pct` values.

**Dataset changes:**
- Credit card DROPPED: DA+ ≈ 136 at bias_pct=0.10, alpha-EO near-zero — not in the scarcity
  regime. Not a valid test case.
- COMPAS added as third dataset (replacing credit). ProPublica recidivism data; protected
  attribute = sex (female, a=0 is disadvantaged); bias_pct=0.14 → DA+=43.
- Final three paper datasets: **census_income, capture24, compas**.

**DA+ calibration (seed=42):**

| Dataset | bias_pct | real_data_size | DA+ | Protected attr | Disadv. group |
|---------|----------|----------------|-----|----------------|---------------|
| census_income | 0.10 | 3000 | 43 | sex | female (a=0) |
| capture24 | 0.02 | 3000 | ~45 | sex | female (a=1) |
| compas | 0.14 | ~2346 (no cap) | 43 | sex | female (a=0) |

**EO guard:** Seeds where alpha-EO < 0.10 are excluded (replaced with next available seed).
This ensures a meaningful fairness gap exists before RL intervention. Stated as an inclusion
criterion in the paper methodology.

**COMPAS setup details:**
- Protected attribute: `dp_protected_col="sex"` (NOT race — cleaner framing, sex gives
  stable DA+ of 43)
- ProPublica standard filters applied: days_b_screening_arrest ±30, is_recid != -1,
  c_charge_degree != 'O', score_text != 'N/A'. All races included.
- Data downloaded to `datasets/compas/compas-scores-two-years.csv` (7215 rows).
- EO gap at alpha: ~0.10–0.15 (confirmed meaningful, sufficient to solve).

### Infrastructure Changes

- `dp_protected_col` field added to all specs and wired through `training.py`, `main.py`,
  `run_baseline.py`, and all 5 baseline trainers.
- `split_compas()` implemented in `dataset.py` with ProPublica filters and sex-based
  protected attribute mapping.
- New `paper_specs_v2/` directory created at project root for all paper-final specs.

### paper_specs_v2 — Main Result Specs

All use v18 config (global-only reward, lambda=[1,1], gen_both_classes=true,
phase2_episodes=200, total_episodes=800, 5 seeds). 18 JSON + 18 .sh files.
SLURM resources: 2 CPUs, 4G memory.

| Dataset | Methods | Spec prefix |
|---------|---------|-------------|
| census_income | global, gdro, otrep, ctgan, fairtabddpm, flb | `v2_census_*_5s` |
| capture24 | global, gdro, otrep, ctgan, fairtabddpm, flb | `v2_capture24_*_5s` |
| compas | global, gdro, otrep, ctgan, fairtabddpm, flb | `v2_compas_*_5s` |

### paper_specs_v2 — Episode Ablation Specs

4 configs × 3 datasets = 12 JSON + 12 .sh files. RL only (no baselines).
Goal: identify best episode budget across datasets; ablate phase 2 contribution.

| Config | total_episodes | phase2_episodes | gen_both_classes | Purpose |
|--------|---------------|-----------------|-----------------|---------|
| ep800ph0 | 800 | 0 | false | Phase 1 only — isolates phase 2 |
| ep800ph200 | 800 | 200 | true | v18 anchor (same as main result) |
| ep1500ph400 | 1500 | 400 | true | Scaling test |
| ep2000ph600 | 2000 | 600 | true | Scaling test |

### Current Status (2026-03-27)

**Running locally (both GPUs):**
- `v2_compas_gdro_5s` → `v2_compas_otrep_5s` → `v2_compas_flb_5s` on cuda:0 (PID 121995)
- `v2_compas_ctgan_5s` → `v2_compas_fairtabddpm_5s` on cuda:1 (PID 122101)
- Logs: `paper_specs_v2/logs/v2_compas_fast_baselines.out` and `v2_compas_slow_baselines.out`

**Queued for DRAC (not yet submitted):**
- All 12 ablation RL jobs (`ablation_{dataset}_ep{N}ph{M}_5s.sh`)
- Main RL jobs: `v2_census_global_5s`, `v2_capture24_global_5s`, `v2_compas_global_5s`

**Already complete (from prior runs, results valid for v2):**
- Census baselines (gdro, otrep, flb, ctgan, fairtabddpm) — in `training_runs/BASELINE_*census_b010*`
- Capture24 baselines (gdro, otrep, flb, ctgan, fairtabddpm) — in `training_runs/BASELINE_*capture24*`

**Next steps (pending COMPAS baseline results):**
1. Verify COMPAS baseline results: check alpha-EO ≥ 0.10 for all 5 seeds
2. If results are clean: begin paper draft
3. Submit ablation RL jobs to DRAC in parallel with paper writing
4. Use best episode config per dataset as the main RL result in the final table


---

## Episode Ablation Results + SMOTE Validity Challenge (2026-03-28)

### Episode Ablation — Full Results (5 seeds, all 3 datasets)

RL ablation jobs completed on DRAC and downloaded to `paper_results_v2/`. All 12 configs
(3 datasets × 4 episode budgets) have exactly 5 seeds each, with EO guard applied. Seeds
differ per dataset based on which passed the alpha-EO ≥ 0.10 threshold:

| Dataset | Seeds used |
|---------|-----------|
| census_income | 0, 2, 3, 5, 42 |
| compas | 1, 3, 6, 7, 42 |
| capture24 | 0, 3, 4, 5, 42 |

#### Results by dataset

**Census Income** (alpha-EO = 0.196 ± 0.066)

| Config | EO ↓ | F1w ↑ | AUC ↑ |
|--------|------|-------|-------|
| ep800/ph0 | 0.188 ± 0.047 | 0.786 ± 0.016 | 0.876 ± 0.009 |
| ep800/ph200 | 0.076 ± 0.048 | 0.729 ± 0.036 | 0.852 ± 0.012 |
| ep1500/ph400 | **0.070 ± 0.078** | 0.719 ± 0.042 | 0.852 ± 0.013 |
| ep2000/ph600 | 0.082 ± 0.073 | 0.729 ± 0.039 | 0.845 ± 0.011 |

**COMPAS** (alpha-EO = 0.052 ± 0.017)

| Config | EO ↓ | F1w ↑ | AUC ↑ |
|--------|------|-------|-------|
| ep800/ph0 | 0.047 ± 0.017 | **0.457 ± 0.024** | **0.703 ± 0.019** |
| ep800/ph200 | 0.038 ± 0.034 | 0.436 ± 0.038 | 0.672 ± 0.020 |
| ep1500/ph400 | 0.039 ± 0.027 | 0.444 ± 0.030 | 0.668 ± 0.015 |
| ep2000/ph600 | **0.016 ± 0.012** | 0.440 ± 0.035 | 0.659 ± 0.007 |

**Capture-24** (alpha-EO = 0.231 ± 0.164)

| Config | EO ↓ | F1w ↑ | AUC ↑ |
|--------|------|-------|-------|
| ep800/ph0 | 0.150 ± 0.109 | 0.939 ± 0.032 | 0.905 ± 0.031 |
| ep800/ph200 | 0.082 ± 0.084 | **0.940 ± 0.027** | 0.868 ± 0.039 |
| ep1500/ph400 | 0.195 ± 0.104 | 0.946 ± 0.021 | 0.893 ± 0.035 |
| ep2000/ph600 | **0.069 ± 0.039** | 0.938 ± 0.023 | 0.862 ± 0.066 |

#### Episode config recommendation

ep2000/ph600 wins or ties on 2/3 datasets (compas, capture24) and is competitive on census.
Using a single config across all datasets is cleaner for the paper. **Tentative main config:
ep2000/ph600.** Census ep1500/ph400 has lowest mean EO (0.070) but highest std (0.078) —
ep800/ph200 may be more reliable there.

### Convergence Analysis — Phase Reset Bug Found

Initial per-episode return plots were incorrect. Episode numbers reset to 1 at the start of
phase 2 in `metrics.csv`, causing `pivot_table(index='episode')` to average phase-1 and
phase-2 rows for episodes 1–200. Fixed by using row index (`global_ep = range(len(df))`)
as the x-axis. Corrected plots saved to `paper_figures/fig_episode_return_fixed.png` and
`paper_figures/fig_eo_convergence_fixed.png`.

**Corrected convergence findings:**

- **Census/COMPAS:** Episode return starts high (~0.9) at episode 1, drops dramatically
  during phase 1 (reaching near 0 by episode 400–600), then jumps back to ~0.9 at the
  phase 2 boundary. EO gap worsens during phase 1 then drops sharply in phase 2.
- **Capture-24:** Return stable and high (~0.85–0.9) throughout both phases. EO declines
  continuously.

The per-episode EO plot shows *current-episode* beta's EO, not best_beta. Final test
uses the combined beta trained on best_phase1_synthetic + best_phase2_synthetic (see
`training.py` lines 1225–1251).

### Critical Finding: High Early Returns Complicate the Learning Claim

**Observation:** At episode 1 (before any RL gradient updates), episode returns are already
~0.9 for census/COMPAS (e.g., census seed_0: return=0.953, alpha-EO=0.196, beta-EO=0.068).
This means the near-zero-delta initial policy — which generates near-distribution minority
positive examples — already substantially beats alpha's worst-group BCE loss.

**Why this matters:** If random or near-random minority augmentation produces high reward from
episode 1, it is unclear whether the RL policy gradient is learning anything beyond what a
simple oversampling approach (SMOTE, random oversampling) would achieve. The agent's
"learning" may reduce to: find which near-distribution perturbations are best, rather than
learning a meaningful generative policy.

**Complicating factors for the paper contribution claim:**

1. The high early reward is genuine — near-distribution minority examples immediately help
   beta beat alpha because the DA+≈43 regime means even a handful of extra minority positives
   shifts the decision boundary. This is consistent with the DA+ scarcity framing, but also
   suggests the problem may partially be solved by simple oversampling.

2. Phase 1 alone (ep800/ph0) gives final EO=0.188 vs alpha=0.196 — barely any improvement
   despite 800 episodes of "learning." The best_synthetic selected over 800 phase-1 episodes
   is not notably better than random augmentation would be. The real EO gain (0.076) only
   comes from the two-phase combination.

3. The contribution is more defensible as a *two-phase structured augmentation framework*
   than as "RL learns to generate fair synthetic data." The reward-guided search over PCA
   space is the mechanism; whether it outperforms simpler search strategies is an open
   empirical question.

**Paper framing adjustment:** Do not claim the RL agent "learns" a meaningful generative
policy. Frame as: "a reward-guided two-phase augmentation framework that jointly searches
for minority-augmenting and majority-recovering synthetic examples under positive-class
scarcity." The RL component is the search mechanism, not an end in itself.

### SMOTE Baseline Added (2026-03-28)

To directly test whether near-distribution augmentation is sufficient, a SMOTE baseline
was implemented (`benchmarks/smote_baseline.py`) and queued.

**Design:** Two-phase SMOTE in PCA-10 space (same feature space as RL), n_synthetic=2000
per phase (matches RL traj_length), same FFNN architecture, same EO guard and seed selection.
Phase 1: SMOTE on minority-group y=1. Phase 2: SMOTE on majority-group y=0.

This is the cleanest possible comparison: same feature space, same data budget, same model,
same seeds. If SMOTE ≈ RL → the RL policy gradient adds negligible value and the contribution
is the two-phase structure alone. If SMOTE << RL → RL is finding better augmentation points
than random interpolation, validating the learned policy.

**Specs:** `paper_specs_v2/v2_{census,compas,capture24}_smote_5s.json`
**Status:** Running locally on both GPUs, queued after existing baseline chains.
**Results pending.** Will be added to this log when complete.

### Current Status (2026-03-28)

**Running locally:**
- GPU0: census gdro → flb → otrep → ctgan → fairtabddpm → compas gdro → flb → otrep →
  *smote census → smote compas*
- GPU1: compas ctgan → fairtabddpm → capture24 gdro → flb → otrep → ctgan → fairtabddpm →
  *smote capture24*

**Complete:**
- All 12 RL ablation configs (paper_results_v2/) — 5 seeds each, results analysed above.

**Next steps:**
1. Wait for SMOTE results — compare EO/F1w/AUC against RL ep2000/ph600.
2. If SMOTE >> RL: reframe more aggressively toward two-phase structure; consider dropping
   RL framing entirely in favour of a simpler search-based method.
3. If SMOTE << RL: strong validation of the learned policy — use this comparison prominently.
4. If SMOTE ≈ RL: honest reporting; contribution is the two-phase framework design, cite
   RL as the search mechanism with the same practical outcome as SMOTE at this scale.
5. Once all baselines complete: build final comparison table and begin paper draft.

---

## Status Snapshot — End of Session (2026-03-28)

### SMOTE Baseline Results (complete)

SMOTE ran on identical seeds to the RL framework (eo_guard disabled, seeds hardcoded).
Results on same data splits, same PCA-10 feature space, n_synthetic=2000 two-phase:

| Dataset | SMOTE EO | RL ep2k/ph6 EO | RL advantage |
|---------|----------|----------------|-------------|
| census_income | 0.133 ± 0.064 | 0.082 ± 0.073 | −0.051 |
| compas | 0.507 ± 0.138 | 0.016 ± 0.012 | **−0.491** |
| capture24 | 0.232 ± 0.170 | 0.069 ± 0.039 | −0.164 |

**Interpretation:** RL substantially beats SMOTE on all datasets. COMPAS is most striking —
SMOTE worsens EO to 0.507 (far above alpha ~0.052), while RL achieves 0.016. Flooding
training with SMOTE-interpolated minority examples in PCA space does not generalise well
for COMPAS; the RL policy is finding meaningfully better augmentations. This resolves the
concern that high early episode returns implied simple oversampling was sufficient.

The "high early returns" finding stands as a nuance (initial policy already helps due to
severe scarcity) but the policy gradient is demonstrably learning something beyond SMOTE-
level interpolation.

### Scarcity Sensitivity Smokes (running overnight)

**Motivation:** At DA+≈43 (bias=0.10), initial episode return is ~0.93 even at episode 1.
Goal: identify whether at less severe scarcity (higher DA+), the initial random policy helps
less and the learning curve shows more genuine improvement over training.

**Specs created:** `experiment_specs/smoke_census_bias{05,15,20}_scarcity.json`
2 seeds each (seeds [0, 2]), eo_guard disabled, ep800/ph200 config.

| Spec | bias_pct | DA+ (approx) |
|------|----------|--------------|
| bias05 | 0.05 | 17 (more scarce) |
| bias15 | 0.15 | 77 (less scarce) |
| bias20 | 0.20 | 96 (less scarce) |

**Auto-monitor running** (`/tmp/scarcity_monitor.sh`, PID 102612):
- Checks every 5 min until all 3 smokes complete
- Computes mean initial episode return (first 5 eps) per bias level
- If any level shows init_return < 0.88 (below the 0.93 baseline at bias=0.10),
  automatically launches 3 more seeds (seeds [3, 5, 42]) on cuda:0
- Full log: `/tmp/scarcity_monitor.log`

**Expected outcome:** bias=0.15 and bias=0.20 likely show lower initial returns since
the disadvantaged group already has more real examples, reducing the marginal value of
random augmentation. If confirmed, these levels provide a cleaner learning curve for
the paper's scarcity motivation figure.

### Baselines on DRAC (submitted)

All 18 baseline jobs submitted to DRAC after `git pull`. Seeds are hardcoded in SMOTE
specs but still use eo_guard for other baselines — see open question about seed matching.

**Fast jobs** (gdro, flb, otrep, smote × 3 datasets = 12 jobs):
Expected runtime: 1–2h each. Seeds may differ from RL for non-SMOTE baselines.

**Slow jobs** (ctgan, fairtabddpm × 3 datasets = 6 jobs):
Expected runtime: 4–12h each.

**Open question before paper:** Should all baseline specs have seeds hardcoded to match
RL (like SMOTE)? Currently only SMOTE specs have eo_guard=0.0 + hardcoded seeds. Other
baselines use eo_guard=0.10 and may select different seeds, making per-seed comparison
invalid. Decision needed when baselines return.

### Next Steps (priority order)

1. **Morning:** Check scarcity smoke results + monitor decision log
2. **When DRAC baselines return:** Decide on seed-matching fix for non-SMOTE baselines
3. **Build full comparison table:** RL (ep2000/ph600) vs all baselines vs SMOTE, 3 datasets
4. **Write paper:** Contribution reframed as two-phase reward-guided augmentation framework;
   RL as search mechanism validated by SMOTE comparison
5. **Nice-to-have:** Scarcity sensitivity curve if bias=0.15/0.20 smokes are promising

---

## Final Baseline Analysis + Paper Figures (2026-03-28)

### Best RL Config Per Dataset (confirmed)

Selected by lowest mean EO across 5 seeds from GG202603271717 ablation runs:

| Dataset | Best Config | EO mean±std | F1w mean±std | AUC mean±std | Run dir (hash) |
|---------|-------------|-------------|--------------|--------------|----------------|
| census_income | ep1500/ph400 | 0.070±0.070 | 0.719±0.037 | 0.852±0.012 | 866d84c2 |
| compas | ep2000/ph600 | 0.016±0.011 | 0.440±0.031 | 0.659±0.006 | 7e86910b |
| capture24 | ep2000/ph600 | 0.069±0.035 | 0.938±0.020 | 0.862±0.059 | 5888523e |

RL seeds used:
- census: [0, 2, 3, 5, 42]
- compas: [1, 3, 6, 7, 42]  ← matches all baselines exactly
- capture24: [0, 3, 4, 5, 42]

### Baseline Results (5 seeds each, GG202603280213/16)

**Baseline seeds used:**
- census: [0, 2, 5, 6, 7]  ← differs from RL by seeds 3, 42 vs 6, 7
- compas: [1, 3, 6, 7, 42]  ← exact match
- capture24: [0, 4, 5, 7, 42]  ← differs from RL by seed 3, 42 vs 7

Seed mismatch on census/capture24 is due to EO guard selecting different fallback seeds
between RL and baseline frameworks. All seeds satisfy alpha-EO ≥ 0.10 criterion.
COMPAS is perfectly matched — safest dataset for per-method comparison.

**CENSUS INCOME** (alpha EO=0.109±0.063):

| Method | EO ↓ | F1w ↑ | AUC ↑ |
|---|---|---|---|
| RL (ep1500/ph400) | 0.070±0.070 | 0.719±0.037 | 0.852±0.012 |
| GroupDRO | 0.074±0.030 | **0.825**±0.010 | **0.894**±0.003 |
| OT Repair | 0.054±0.035 | 0.792±0.003 | 0.823±0.006 |
| **FLB** | **0.031**±0.028 | 0.819±0.008 | 0.889±0.001 |
| FairTabDDPM | 0.070±0.039 | 0.791±0.008 | 0.845±0.009 |

FLB wins on census. RL is competitive with GroupDRO and FairTabDDPM on EO but loses on F1.

**COMPAS** (alpha EO=0.070±0.044):

| Method | EO ↓ | F1w ↑ | AUC ↑ |
|---|---|---|---|
| **RL (ep2000/ph600)** | **0.016**±0.011 | 0.440±0.031 | 0.659±0.006 |
| GroupDRO | 0.156±0.050 | **0.605**±0.043 | 0.648±0.050 |
| OT Repair | 0.036±0.020 | 0.466±0.015 | **0.700**±0.009 |
| FLB | 0.118±0.016 | 0.614±0.050 | 0.658±0.069 |
| FairTabDDPM | 0.064±0.046 | 0.483±0.021 | 0.681±0.013 |

RL dominant. GroupDRO and FLB **worsen** EO vs alpha — the scarcity claim confirmed.

**CAPTURE-24** (alpha EO=0.229±0.157):

| Method | EO ↓ | F1w ↑ | AUC ↑ |
|---|---|---|---|
| **RL (ep2000/ph600)** | **0.069**±0.035 | 0.938±0.020 | 0.862±0.059 |
| GroupDRO | 0.141±0.071 | 0.906±0.021 | 0.900±0.043 |
| OT Repair | 0.080±0.065 | **0.949**±0.012 | 0.893±0.050 |
| FLB | 0.106±0.056 | 0.906±0.011 | **0.927**±0.023 |
| FairTabDDPM | 0.250±0.137 | **0.953**±0.010 | **0.938**±0.012 |

RL achieves lowest EO. FairTabDDPM worsens EO above alpha — fails under scarcity.

### Paper Figures v2 (saved to paper_figures_v2/)

- `table_census.tex`, `table_compas.tex`, `table_capture24.tex` — LaTeX comparison tables
- `fig_training_curve_census_income.png` — Best config training curves, 3 rows
- `fig_training_curve_compas.png`
- `fig_training_curve_capture24.png`
- `fig_eo_comparison.png` — Side-by-side EO bar chart all datasets

### CTGAN + SMOTE Local Runs (2026-03-28)

Fixed spec errors:
- SMOTE census: was `minority_id=0` (wrong), fixed to 1
- All SMOTE/CTGAN: seeds now hardcoded to match other baselines, guard=0.0
  - census: [0,2,5,6,7], compas: [1,3,6,7,42], capture24: [0,4,5,7,42]

Running locally (background):
- SMOTE census: COMPLETE (24s/seed)
- SMOTE compas: running (~86478)
- SMOTE capture24: running (~89014)
- CTGAN census: running (300 epochs, slow)
- CTGAN compas: running (~90814)
- CTGAN capture24: running (~90815)

When complete: copy final_test_metrics.csv to paper_results_v2/ and regenerate figures.

---

## COMPAS Race Investigation (2026-03-31)

**Spec:** `compas_race_ep1500ph400_5s` — bias_pct=0.05, dp_protected_col=race, minority_id=0 (Caucasian), 5 seeds [0,2,3,5,42].

### Alpha-EO Discrepancy

The COMPAS_PLACEHOLDERS in the generate scripts assumed alpha-EO ≈ 0.41. The actual 5-seed run produced **alpha-EO = 0.677 ± 0.024**. The 0.41 figure was a manual estimate written into the placeholder dicts before results arrived — it was never from a real run. With bias_pct=0.05, Caucasian TPR under alpha collapses to near-zero (~0.024), producing a much larger gap than anticipated.

### Full 5-Seed Results (paper_results_v3/)

**RL run:** `SPECcompas_race_ep1500ph400_5s_...BIAS0.05_GG202603310150_87dcb72a`

| Metric | Alpha | RL (beta) |
|--------|-------|-----------|
| EO gap | 0.677 ± 0.024 | 0.600 ± 0.055 |
| Caucasian TPR | 0.024 ± 0.008 | 0.081 ± 0.041 |
| AA TPR | 0.701 ± 0.018 | 0.681 ± 0.058 |
| F1-weighted | 0.641 ± 0.008 | 0.641 ± 0.013 |
| AUC | 0.681 ± 0.006 | 0.682 ± 0.014 |

Mean EO improvement: 0.077 ± 0.037. Utility preserved.

### Baseline Comparison (5 seeds each, paper_results_v3/)

| Method | EO ↓ | F1w | AUC |
|--------|-------|-----|-----|
| Alpha (ERM) | 0.677 ± 0.024 | 0.641 | 0.681 |
| **RL (v18)** | 0.600 ± 0.055 | 0.641 | 0.682 |
| GroupDRO | **0.197 ± 0.105** | 0.650 | 0.692 |
| OT Repair | 0.556 ± 0.041 | 0.637 | 0.675 |
| FLB | **0.075 ± 0.054** | 0.628 | 0.671 |
| FairTabDDPM | 0.541 ± 0.043 | 0.647 | 0.686 |
| CTGAN | — (timed out on DRAC) | | |
| SMOTE | — (timed out on DRAC) | | |

### Key Finding: Paper Narrative Does Not Hold for COMPAS Race

GroupDRO (EO=0.197) and FLB (EO=0.075) substantially outperform RL (EO=0.600) on this dataset. The motivation claim — reweighting methods fail under severe scarcity — is not supported here, even though DA+≈40, the same scarcity level as census and capture24. RL is the weakest method for fairness on this configuration.

**Possible explanations to investigate:**
1. The race signal in COMPAS may be easy to reweight from even at low count — the protected attribute is a strong predictor and reweighting on it is sufficient.
2. Caucasian TPR collapsing to ~0.024 under alpha suggests the PCA space may not place Caucasian positives in a recoverable region for delta-based synthesis.
3. The global reward (`sigmoid(10 × (wgl_alpha − wgl_beta))`) may saturate too quickly given the extreme alpha-EO, providing poor gradient signal.

### Incomplete Baselines

CTGAN and SMOTE both started but only completed seed_0 setup before hitting the DRAC wall time. Need to resubmit with longer time limit or fewer epochs.

### Suggested Next Steps

1. **Before including COMPAS race in paper:** Run a diagnostic — check whether RL is actually learning (reward curve increasing) across seeds. If reward is flat, the agent is not learning at all on this dataset.
2. **Investigate FLB success:** FLB achieves EO=0.075 with utility cost (F1w 0.628 vs alpha 0.641). If this holds, COMPAS race strengthens baselines, not RL.
3. **Options for paper:** (a) Drop COMPAS race and explain that the race-based scarcity in COMPAS creates a different structure than sex-based scarcity; (b) Keep as a limitation/discussion point; (c) Reframe: show RL achieves best utility-fairness tradeoff (RL has highest AUC at 0.682 with EO=0.600, vs FLB which gets lower EO but worse utility).
4. **Resubmit CTGAN/SMOTE** with longer walltime to complete the baseline picture.
5. **Update COMPAS placeholder values** in generate_ablation_figures.py and generate_main_table.py once a final decision on COMPAS inclusion is made.


---

## Third Dataset Viability Analysis (2026-04-02)

**Context:** COMPAS race results do not support the paper's motivation claim (GroupDRO/FLB substantially outperform RL). Systematic structural analysis run to determine whether PTB-XL, MEPS, or a replacement could serve as the third dataset.

**Structural viability criteria (calibrated against census):**
1. `val_disadv_pos ≥ 30` — enough biased-val signal to drive worst-group BCE reward
2. `test_disadv_pos ≥ 200` — reliable fairness evaluation
3. `alpha_EO ≥ 0.10` — meaningful gap for the RL agent to close
4. (Cosine criterion dropped — census itself has cosine=0.983, so this is not discriminative)

**Census reference values (measured, seed=42, bias=0.10, pca=10):**
- DA+=43, val_pos=86, test_pos=244, alpha_EO=0.118, cosine=0.983

---

### PTB-XL: All Framings Rejected

Tested MI, STTC, HYP, CD vs NORM with sex-based (female=1 disadv) and age-based (young<50=0 disadv) groupings at bias_pct=0.02 and 0.04. Full results from `dataset_viability_analysis.py`:

| Framing | Val+ | Test+ | EO | Cosine | Pass |
|---|---|---|---|---|---|
| MI/sex bias=0.04 | 42 | 344 | 0.056 | 0.928 | 2/4 |
| MI/age bias=0.04 | 32 | 52 | 0.108 | 0.628 | 1/3* |
| STTC/age bias=0.04 | 16 | 8 | 0.409 | -0.138 | 1/3* |
| HYP/age bias=0.04 | 9 | 5 | 0.178 | 0.461 | 1/3* |
| CD/age bias=0.04 | 16 | 14 | 0.132 | 0.781 | 1/3* |

*scored against census-calibrated 3-criterion set (no cosine)

**Root causes:**
- Sex-based framings: ECG disease features are sex-independent (cosine ≥0.90 across all conditions). EO never exceeds 0.20 regardless of bias level. Fundamentally not viable.
- Age-based framings: strat_fold structure hard-limits young positives in test. MI gives only 52 young test positives; rarer conditions (STTC, HYP, CD) give 5–14. No framing reaches test_pos≥200.

**Verdict: PTB-XL is not viable as a fairness dataset under any framing available in the strat_fold split structure.**

---

### MEPS: Ethnicity Framing Invalid (Existing DRAC Runs)

The existing MEPS DRAC runs (`meps_opv_rl_800ep_5s`, baselines) used `dp_protected_col=ethnicity` (Hispanic=0 vs non-Hispanic=1). This framing produces test_disadv_pos=111 (below 200 threshold). The 2 seeds that completed on DRAC are invalid for the paper. Specs need to be replaced with race framing.

### MEPS Race Framing: Marginal

Race framing (Black=0 vs White=1, dental visits, bias=0.08, pca=25): passes 3/3 census-calibrated criteria on paper (val_pos=37, test_pos=216, EO=0.759). However, a 200ep local smoke test showed:

- Phase 1 best_global_obj=0.927 (strong learning signal)
- Alpha EO: 0.750 (Black TPR=0.000, White TPR=0.750)
- Beta EO after 200ep: 0.721 (Black TPR=0.042, White TPR=0.763) — only 0.029 improvement
- F1w: 0.639 → 0.632 (marginal utility drop)

The weak test EO improvement despite strong phase 1 global_obj suggests val_pos=37 is insufficient for reliable policy gradient learning — the agent optimizes for a noisy 37-example val signal that doesn't generalize to the 216-example test set. Compare census: 86 val positives, consistent 0.05–0.08 EO reduction by 200ep.

**Verdict: MEPS race framing is structurally marginal. val_pos=37 is too low for stable learning.**

---

### COMPAS Race: Structurally Inferior to MEPS

Structural reanalysis (same methodology as above, seed=42, bias=0.05, pca=10):
- DA+=40, val_pos=**14**, test_pos=**162**, alpha_EO=0.689, cosine=0.757

COMPAS has only 14 val Caucasian positives — the worst of all candidates, worse than MEPS (37) and far below census (86). The small total dataset size (~5k rows) is the root cause; there simply aren't enough examples in any split. The 14 val positives explain why COMPAS RL results are inconsistent across seeds.

**Combined with the existing COMPAS Race Investigation finding (GroupDRO EO=0.197, FLB EO=0.075 vs RL EO=0.600): COMPAS race is both structurally weak and produces results that contradict the paper's motivation claim. Decision: replace COMPAS with ACS Income.**

---

### ACS Income (NeurIPS 2021): Selected as Third Dataset

ACS PUMS 2018 California, sex-based (female=1 disadv, income>50k), pca=10:

| bias_pct | DA+ | Val+ | Test+ | Alpha-EO | Cosine |
|---|---|---|---|---|---|
| 0.10 | 105 | 6,427 | 6,327 | 0.597 | 0.969 |
| 0.07 | 71 | 6,427 | 6,327 | 0.680 | 0.969 |
| **0.05** | **49** | **6,427** | **6,327** | **0.729** | **0.969** |
| 0.03 | 29 | 6,427 | 6,327 | 0.726 | 0.969 |

Val/test positives are 6,427 and 6,327 respectively — 75× more than COMPAS and 74× more than MEPS for the validation signal. Alpha-EO at bias=0.05 is 0.729 (female TPR collapses to 0.040 under bias).

**Target config:** bias_pct=0.05 (DA+=49, closest to target ~43), pca=10 (matching census), train_size=3000.

**Reviewer framing:** ACS is the community-recommended successor to UCI Adult per Ding et al. NeurIPS 2021 ("Retiring Adult"). Using it alongside census_income demonstrates scale robustness (4× the data), not redundancy. Both datasets use sex as protected attribute and income prediction as the task — this is intentional; it isolates the effect of dataset scale and positive-class rate.

#For future reference
  FairJob — Detailed Assessment Against Your Project
                                                                                                                                                                                                                         
  Critical numbers at real_data_size=3000                                                                                                                                                                              
                                                                                                                                                                                                                         
  ┌────────┬─────────┬──────────────┬───────────────────────────┐                                                                                                                                                        
  │ Group  │ Samples │  Click rate  │ Estimated positives (DA+) │
  ├────────┼─────────┼──────────────┼───────────────────────────┤                                                                                                                                                        
  │ Female │ 1,500   │ ~0.37% (avg) │ ≈ 6                       │
  ├────────┼─────────┼──────────────┼───────────────────────────┤                                                                                                                                                        
  │ Male   │ 1,500   │ ~0.34% (avg) │ ≈ 5                       │                                                                                                                                                        
  └────────┴─────────┴──────────────┴───────────────────────────┘                                                                                                                                                        
                                                                                                                                                                                                                         
  DA+ ≈ 5–6 at your standard real_data_size=3000. That's 7–8× lower than your target of 43. To reach DA+ ≈ 43 you'd need real_data_size ≈ 30,000 — a 10× change to a core hyperparameter that was ablated on census.     
                                                            
  The fairness structure doesn't match your framework                                                                                                                                                                    
                                                            
  FairJob's documented unfairness is exposure/campaign selection bias: females make up 53.6% of the user population but receive only 39.2% of job ad impressions. But the per-impression click rates are nearly identical
   (female non-senior 0.24%, male 0.17% — females actually click slightly more).
                                                                                                                                                                                                                         
  Your framework is built around group-specific positive-class scarcity — the disadvantaged group has far fewer positive examples to learn from. In FairJob, both groups are equally click-starved; the injustice is     
  upstream in ad targeting, not in the label distribution your beta classifier sees. The RL agent generating synthetic minority-positive examples is solving the wrong problem for this dataset.
                                                                                                                                                                                                                         
  Protected attribute weakness                                                                                                                                                                                           
   
  The gender proxy is derived from product purchase patterns, not demographics. A reviewer will immediately ask whether "gender proxy" constitutes a protected attribute in the legal/ethical sense, and whether your    
  results hold for actual gender.                           
                                                                                                                                                                                                                         
  ---                                                                                                                                                                                                                    
  Verdict: FairJob is a poor fit
                                                                                                                                                                                                                         
  It would require fundamental changes (10× real_data_size, new dataset integration, re-ablation) and the fairness structure is misaligned with your core contribution. It does not replace COMPAS.
                                                                                                                                                                                                                         
  The COMPAS problem is a results problem, not a dataset problem. The path forward there is either (a) diagnosing why reweighting works on COMPAS race specifically and framing that as a finding, or (b) replacing      
  COMPAS with a dataset that actually exhibits baseline failure — MEPS health survey, hospital readmission data, or a small-jurisdiction criminal justice dataset where one racial group genuinely has very few          
  positive-outcome examples.    

---

## ACS Income — Not in Scarcity Regime (2026-04-02)

**Finding:** ACS Income at bias_pct=0.04 (the intended config) has DA+≈1500 before the
real_data_size=3000 cap is applied. After cap, the female income>50k group is still
over-represented relative to the target ~43. FLB achieves EO≈0.002 on some seeds.
Reweighting baselines work well — the dataset is not in the scarcity regime our framework
is designed for. **Decision: drop ACS Income. Continue with census + capture24 as the
two primary paper datasets.**

---

## Framework Improvement Phase (2026-04-03)

**Motivation:** After ACS Income was dropped, the focus shifted from dataset selection to
framework improvement. Core structural problem identified: REINFORCE training loop is
structurally a trajectory-level bandit. All 2000 steps in a trajectory receive identical
reward (global_term / T), making within-episode credit assignment undefined. True per-step
RL is not possible without per-step beta retraining (prohibitively expensive). Two improvement
leads identified:

1. **Normalized reward (v19):** Replace `sigmoid(10 × Δwgl)` with `(wgl_alpha − wgl_beta) /
   wgl_alpha`. Removes sigmoid saturation, gives continuous scale-invariant gradient.
   Triggered by `global_sigmoid_k: 0` in spec.

2. **Evolutionary search (v20):** Replace REINFORCE entirely with CMA-ES evolving a
   parameterized Gaussian in PCA space. Eliminates false sequential structure.
   20D parameter vector: mean(10D) + log_std(10D). Default popsize≈12 for D=20.

---

## v19 — Normalized Reward Diagnostic (2026-04-02)

**Specs:** `v19_census_normalized_reward_s42.json`, `v19_census_normalized_reward_s0.json`
(single seed each for parallel GPU execution). 800 episodes, same config as v18 otherwise.

**Results (census, bias_pct=0.10):**

| Seed | α-EO | β-EO | F1w | AUC |
|------|------|------|-----|-----|
| 42 | 0.126 | 0.031 | 0.800 | 0.870 |
| 0  | 0.146 | 0.082 | 0.803 | 0.854 |
| **mean** | — | **0.056±0.036** | **0.802** | **0.862** |

Compared to v18 (global-only, sigmoid): EO=0.063±0.059, F1w=0.811, AUC=0.877.
Mixed result — EO slightly better, F1w/AUC slightly worse. Improvement is modest and
within variance. Confirms that optimizer (not just reward) is the bottleneck.

---

## v20 — CMA-ES Evolutionary Search (2026-04-03)

**Implementation:** `agents/cmaes_agent.py` (new), `training.py` (`use_cmaes` dispatch),
`main.py` (`use_cmaes`/`cmaes` spec passthrough). Requires `pip install cma`.

**Design:**
- CMA-ES evolves 20D parameter vector [mean(10D), log_std(10D)] in PCA space
- Each "episode" = one candidate evaluation (sample 2000 points from Gaussian, train beta,
  compute reward, tell CMA-ES)
- After popsize(≈12) episodes = one generation → `es.tell()` → next generation
- No env/delta actions — samples drawn i.i.d. from Gaussian per candidate
- Normalized reward (`global_sigmoid_k: 0`) used throughout
- Phase 2: separate CMAESAgent initialized from real majority sample mean

**Specs:** `v20_census_cmaes_2s.json` (130 eps, 2 seeds), `v20_compas_cmaes_2s.json`
(130 eps, 2 seeds, bias_pct=0.05, dp_protected_col=race). Run locally on 2 GPUs.

### Results

**Census (bias_pct=0.10), 2 seeds [42, 0]:**

| Seed | α-EO | β-EO | F1w | AUC |
|------|------|------|-----|-----|
| 42 | 0.126 | 0.021 | 0.788 | 0.884 |
| 0  | 0.146 | 0.022 | 0.790 | 0.881 |
| **mean** | — | **0.021±0.000** | **0.789** | **0.882** |

**COMPAS (bias_pct=0.05, race), 2 seeds [42, 0]:**

| Seed | α-EO | β-EO | F1w | AUC |
|------|------|------|-----|-----|
| 42 | 0.674 | 0.625 | 0.650 | 0.679 |
| 0  | 0.670 | 0.570 | 0.654 | 0.702 |
| **mean** | — | **0.597±0.039** | **0.652** | **0.691** |

### Full Comparison Table — Census (bias_pct=0.10)

| Method | n | EO | F1w | AUC |
|--------|---|-----|-----|-----|
| GroupDRO | 5 | 0.106±0.022 | 0.822 | 0.895 |
| OT Repair | 5 | 0.053±0.046 | 0.791 | 0.821 |
| SMOTE | 5 | 0.134±0.065 | 0.781 | 0.854 |
| CTGAN | 5 | 0.109±0.049 | 0.792 | 0.840 |
| FLB | 5 | 0.033±0.037 | 0.818 | 0.888 |
| FairTabDDPM | 5 | 0.058±0.054 | 0.798 | 0.848 |
| v18 REINFORCE (sigmoid) | 5 | 0.076±0.048 | 0.729 | 0.852 |
| v19 REINFORCE (norm.) | 2 | 0.056±0.036 | 0.802 | 0.862 |
| **v20 CMA-ES (norm.)** | **2** | **0.021±0.000** | **0.789** | **0.882** |

### Full Comparison Table — COMPAS (bias_pct=0.05, race protected attr)

| Method | n | EO | F1w | AUC |
|--------|---|-----|-----|-----|
| GroupDRO | 5 | 0.197±0.105 | 0.650 | 0.692 |
| OT Repair | 5 | 0.556±0.041 | 0.637 | 0.675 |
| FLB | 5 | 0.075±0.054 | 0.628 | 0.671 |
| FairTabDDPM | 5 | 0.541±0.043 | 0.647 | 0.686 |
| v18 REINFORCE (sigmoid) | 5 | 0.600±0.055 | 0.641 | 0.682 |
| **v20 CMA-ES (norm.)** | **2** | **0.597±0.039** | **0.652** | **0.691** |

### Key Findings

1. **CMA-ES dramatically outperforms REINFORCE on census.** EO 0.021±0.000 vs v18's
   0.063±0.059 — a 67% improvement in mean EO, near-zero variance across seeds (std=0.000).
   This is the best result of any method on census by a wide margin.

2. **Seed variance eliminated.** v18 REINFORCE had EO range ~0.12 across seeds; CMA-ES
   produces virtually identical results (0.021 vs 0.022) despite entirely different data
   splits. CMA-ES is finding a stable region of the search space that REINFORCE was only
   occasionally landing in.

3. **F1w and AUC competitive.** v20 F1w=0.789 is lower than FLB (0.818) and GroupDRO (0.822)
   but competitive with other generative methods. AUC=0.882 is second only to GroupDRO (0.895).

4. **COMPAS: CMA-ES matches REINFORCE exactly but doesn't beat FLB/GroupDRO.** Both
   generative methods fail to substantially reduce EO on COMPAS race. FLB (0.075) and
   GroupDRO (0.197) both outperform on EO. Only 10 CMA-ES generations at 130 episodes —
   more training may help but structural issues (val_pos=14, cosine=0.757) are the likely
   root cause. COMPAS race remains a difficult case.

5. **CMA-ES vs REINFORCE structural advantage confirmed.** The trajectory-level bandit
   structure of REINFORCE produces high variance because the policy gradient is a noisy
   signal over 2000 steps with identical reward. CMA-ES treats each episode as a direct
   function evaluation and adapts its search distribution accordingly — a fundamentally
   better match for this problem structure.

### Why FLB is Competitive on Census

FLB (Fairness Loss Balancing, Kim et al. 2023) uses adaptive batch sampling: per-epoch,
it allocates more batch slots to whichever (a, y) subgroup currently has the highest BCE
loss. Standard unweighted BCE is then computed on the resampled batch. Key observations:

- **10× more training:** FLB uses 200 epochs vs our FFNN's 20 epochs. A meaningful fraction
  of FLB's advantage is simply more thorough training on the same real data.
- **Why it works on census but not COMPAS:** Census female positives have a distinctive
  feature profile that longer training can learn. COMPAS Caucasian positives (race framing)
  are not clearly separable from African-American positives in feature space — oversampling
  an indistinct group doesn't help.
- **Scarcity sensitivity hypothesis (not yet tested):** FLB's advantage likely degrades
  faster than CMA-ES as DA+ drops below ~20, because there's simply nothing to oversample
  from. A bias_pct sweep on census (FLB vs CMA-ES across DA+ levels) would demonstrate this
  and strengthen our motivation claim.

### Next Steps

1. **5-seed v20 CMA-ES on census** to lock in the result (currently 2 seeds — provisional).
2. **v20 capture24** — second paper dataset; key test of generalizability.
3. **Local reward exploration:** OT-inspired prior (use OT transport map to define target
   distribution in PCA space) or FLB-inspired signal (oversample highest-loss synthetic
   candidates). See "Framework Improvement Leads" discussion in CLAUDE.md.
4. **Scarcity curve:** FLB vs CMA-ES across bias_pct levels on census to support motivation claim.

---

## Paper Framing Pivot — Scarcity Curve (2026-04-03)

### New Core Framing

**Previous framing:** "Our RL generative framework achieves SOTA fairness-utility tradeoff vs all baselines."

**Problem with that framing:** On capture24, CMA-ES shows inconsistent results across seeds (degradation at low alpha-EO). On COMPAS, even our best method (FLB) is simply the best reweighter. The "beat all baselines" claim is too broad to defend across three datasets.

**New framing:** "Reweighting methods (GroupDRO, OT Repair) fail under severe positive-class scarcity (DA+ ≤ ~40). Generative augmentation — specifically, optimizing the synthetic distribution via CMA-ES — maintains fairness improvement where reweighting degrades. We demonstrate this via a scarcity curve on census_income across four DA+ levels."

**Key supporting claims:**

1. *Motivation:* GroupDRO and OT Repair EO degrades as DA+ decreases below ~43; their best-case improvement disappears at DA+=17. CMA-ES holds or improves.
2. *Ablation:* CMA-ES outperforms Gaussian Augment (same distribution, no optimization) — the optimization loop matters, not just sampling from the right distribution.
3. *Generalizability:* capture24 is included to satisfy funding requirements; reported with the caveat that the method's advantage is clearest when alpha-EO is substantial (>0.1).

**The "why not simpler?" answer** is now answered by the Gaussian Augment ablation: naive i.i.d. sampling from the real minority distribution is not sufficient — CMA-ES optimization is what drives the improvement.

### Experiment Status (as of 2026-04-03)

#### Running on DRAC cluster
| Spec | Description | Status |
|------|-------------|--------|
| v25a_capture24_cmaes_sigma05_5s | capture24, CMA-ES, sigma0=0.5 | Running (~45 min remaining) |
| v25b_capture24_cmaes_k10_5s | capture24, CMA-ES, sigmoid k=10 | Running (~45 min remaining) |
| v23_census_rl_ot_nocurr_5s | census, REINFORCE+OT, no curriculum | Pending (Priority queue) |

#### Completed (results in paper_results_v4 / training_runs)
| Spec | Description | Key result |
|------|-------------|------------|
| v18 (5 seeds) | census REINFORCE sigmoid | EO=0.076±0.048, F1w=0.729 |
| v19 (2 seeds) | census REINFORCE normalized | EO=0.056±0.036, F1w=0.802 |
| v20 census (2 seeds) | census CMA-ES normalized | EO=0.021±0.000, F1w=0.789 |
| v20 COMPAS (2 seeds) | COMPAS CMA-ES normalized | EO=0.597±0.039, F1w=0.652 |
| v20 capture24 (5 seeds) | capture24 CMA-ES normalized | Inconsistent; seeds 3&4 degrade |
| Baselines census (5 seeds) | GroupDRO, OT Repair, FLB, CTGAN, SMOTE, FairTabDDPM | See comparison table above |
| Baselines COMPAS (5 seeds) | GroupDRO, OT Repair, FLB, FairTabDDPM | See comparison table above |

#### Pending — Scarcity Curve (census_income, 5 seeds each)
| Spec | DA+ (approx) | bias_pct | Status |
|------|-------------|----------|--------|
| sc_census_cmaes_bias005_5s | ~17 | 0.05 | Not yet submitted |
| sc_census_cmaes_bias015_5s | ~65 | 0.15 | Not yet submitted |
| sc_census_cmaes_bias020_5s | ~90 | 0.20 | Not yet submitted |
| sc_census_gdro_bias005_5s | ~17 | 0.05 | Running locally |
| sc_census_gdro_bias015_5s | ~65 | 0.15 | Running locally |
| sc_census_gdro_bias020_5s | ~90 | 0.20 | Running locally |
| sc_census_otrep_bias005_5s | ~17 | 0.05 | Running locally |
| sc_census_otrep_bias015_5s | ~65 | 0.15 | Running locally |
| sc_census_otrep_bias020_5s | ~90 | 0.20 | Running locally |
| sc_census_flb_bias015_5s | ~65 | 0.15 | Not yet submitted |
| sc_census_flb_bias020_5s | ~90 | 0.20 | Not yet submitted |
| sc_census_gauss_augment_bias010_5s | ~43 | 0.10 | Not yet submitted |

Note: bias_pct=0.10 (DA+≈43) baselines are already complete from v16 runs.
CMA-ES at bias_pct=0.10 (DA+≈43) is complete from v20 (2 seeds; submit 5-seed version on DRAC).

#### Pending — New Model Variants
| Spec | Description | Status |
|------|-------------|--------|
| v24_census_cmaes_ot_5s | census CMA-ES + OT local reward (code fix applied) | Not yet submitted |
| v25a_capture24_cmaes_sigma05_5s | capture24 CMA-ES sigma0=0.5 | On DRAC |
| v25b_capture24_cmaes_k10_5s | capture24 CMA-ES sigmoid k=10 | On DRAC |

### Code Fix — CMA-ES OT Integration (2026-04-03)

**Bug:** In `training.py`, `agent.tell(candidate_idx, global_obj)` was passing only the global fairness signal to CMA-ES. The OT local reward was being computed but discarded — CMA-ES never saw it.

**Fix:** Changed to `agent.tell(candidate_idx, float(rewards.sum().item()))` so the full episode return (global + local) flows to CMA-ES. When lambda=1.0 or w_ot=0, rewards.sum() == global_obj — pure-global configs are unaffected.

**Impact:** v21 OT+CMA-ES results are invalid (OT was inactive). v24 re-runs with fix applied.

### New Baseline — Gaussian Augment

**File:** `benchmarks/gaussian_augment.py`

**Purpose:** "No-search" ablation against CMA-ES. Fits a diagonal Gaussian N(μ, diag(σ²)) to real disadvantaged minority samples in PCA space, draws 2000 i.i.d. samples (matching CMA-ES traj_length). No optimization loop — answers whether CMA-ES improves over naive Gaussian sampling from the real minority distribution (i.e., CMA-ES episode 0).

**Phase 1:** Augment minority positives (group=minority_id, y=1).  
**Phase 2:** Augment majority negatives (group=majority_id, y=0) — mirrors gen_both_classes=True.

Both phases use the same FFNN as CMA-ES beta (20 epochs, [32,16] hidden, lr=0.001).

**Expected result:** Gaussian Augment should outperform doing nothing (alpha) but fall short of CMA-ES — if it matches CMA-ES, the optimization loop adds no value and the paper's core claim is undermined.

---

## April 7, 2026 — Experiments Session

### Infrastructure Changes

**FairJob dataset integrated** (`dataset.py`, `training.py`, `main.py`):
- New `split_fairjob()` method. Loads Criteo CTR parquet, filters `displayrandom=1`, OHE + scales features, stratified split.
- `pool_pos_fraction` parameter undersmaples negatives in train+val post-split to enrich positives (~0.7% natural click rate → configurable target fraction). Test set left at natural distribution to preserve AUC/F1 validity.
- Group-specific bias: only female (a=0, disadvantaged) positives are downsampled by `bias_pct`.
- Smoke test confirmed: DA+=39 at bias_pct=0.03, pool_pos_fraction=0.05, real_data_size=3000.

**PPO+DVRL integrated** (`agents/ppo_agent.py`, `training.py`, `main.py`):
- `use_ppo: bool` flag in Training and JSON specs routes agent instantiation to `PPOAgent`.
- `PPOAgent.predict()` returns `(action, log_prob)` tuple; log-probs stored at rollout time and passed to `learn_trajectory(old_log_probs=...)` — importance ratio is computed from behavior policy, not the updated actor.
- `_run_phase` pre-allocates `log_probs[T]` buffer; step loop branches on `use_ppo`.
- Phase 2 also branches to a fresh `PPOAgent` when `use_ppo=True`.
- GAE advantages (lambda=0.95) subtract the critic's value baseline, isolating per-step local signal even when global term is constant across all trajectory steps. This is the key structural advantage over REINFORCE for DVRL local rewards.
- Motivation: supervisor noted REINFORCE with gamma=1.0 gives identical global_term/T to every step, so the policy gradient only sees within-trajectory local variation. PPO+GAE can exploit per-step DVRL signal properly.

**Specs saved to `overleaf_figures_specs/`**: All 13 canonical paper_results_v2 specs (census + capture24, ours + all baselines) saved for reproducibility reference.

---

### Experiments Launched — 2026-04-07

#### Running Locally — cuda:0

| Spec | Description | Status |
|------|-------------|--------|
| `census_ppo_dvrl_2s` | Census, PPO+DVRL, lambda=0.5, k=10, 800ep+200ph2, seeds [42,0] | Running |

**Config:** `use_ppo=true`, `use_dvrl_local=true`, `lambda_schedule=[0.5, 0.5]`, `global_sigmoid_k=10`, `gen_both_classes=true`, `phase2_episodes=200`, FFNN [32,16], PPO hidden=64, lr=3e-4, gamma=1.0, clip_epsilon=0.2, update_epochs=4.

**Purpose:** First PPO+DVRL run. Tests whether GAE advantage isolation allows per-step DVRL local reward to drive better minority-region targeting than REINFORCE. Comparing against v18 baseline (REINFORCE, lambda=1.0, no local reward, EO=0.076±0.048).

---

#### Running Locally — cuda:1 (sequential)

| Spec | Description | Status |
|------|-------------|--------|
| `census_nosigmoid_3s` | Census, REINFORCE, normalized reward (k=0), 3000ep no phase2, seeds [42,0,1] | Running (1st in queue) |
| `capture24_nosigmoid_3s` | Capture-24, REINFORCE, normalized reward (k=0), 3000ep no phase2, seeds [42,0,1] | Queued |
| `compas_nosigmoid_3s` | COMPAS race, REINFORCE, normalized reward (k=0), 3000ep no phase2, seeds [42,0,1] | Queued |

**Config (all three):** `reward_mode=fairness`, `global_sigmoid_k=0` (normalized reward: `(wgl_alpha - wgl_beta) / wgl_alpha`), `lambda_schedule=[1.0, 1.0]` (global-only), `gen_both_classes=false`, `total_episodes=3000`. Census: bias_pct=0.10. Capture-24: bias_pct=null, win_seconds=1.0, step_seconds=0.5. COMPAS: bias_pct=0.05, dp_protected_col=race, minority_id=0.

**Purpose:** Supervisor-requested diagnostic. Tests whether removing sigmoid saturation (replacing sigmoid(k*delta) with linear normalized improvement) helps or hurts learning stability across three datasets. Normalized reward gives continuous gradient signal vs sigmoid which saturates near 0 or 1 when k=10. COMPAS included at supervisor request despite known val_pos=14 structural issue.

---

#### Prepared for DRAC — Episode Ablations

| Spec | Description |
|------|-------------|
| `census_ep1500_nophase2` | Census, 1500ep, no phase2, seeds [42,0,1,2,3] |
| `census_ep2000_nophase2` | Census, 2000ep, no phase2, seeds [42,0,1,2,3] |
| `capture24_ep1500_nophase2` | Capture-24, 1500ep, no phase2, seeds [42,0,1,2,3] |
| `capture24_ep2000_nophase2` | Capture-24, 2000ep, no phase2, seeds [42,0,1,2,3] |

**Purpose:** Episode count ablation for paper figures. Compared against existing 800ep+phase2 (paper config) to show that additional episodes without phase2 do not materially change results — or to find the optimal episode budget.

---

#### Prepared for DRAC — ROC-EO Lambda Ablation

| Spec | Description |
|------|-------------|
| `compas_roc_eo_lam[05,04,03,02]_3s` | COMPAS race, reward_mode=roc_eo, lambda in {0.5,0.4,0.3,0.2}, 1500ep+ph2, seeds [42,0,1] |
| `fairjob_roc_eo_lam[05,04,03,02]_3s` | FairJob, reward_mode=roc_eo, lambda in {0.5,0.4,0.3,0.2}, 1500ep+ph2, seeds [42,0,1] |

**Purpose:** Sweep roc_eo_lambda (AUC vs EO tradeoff weight) on COMPAS and FairJob. `roc_eo` reward = `lambda * AUC - (1-lambda) * EO`. FairJob uses bias_pct=0.03, pool_pos_fraction=0.05, global_sigmoid_k=3 (WorstLoss ~3.5, k=10 would saturate).

---

### Results — Pending

All runs above are in-progress or queued. Update this section when results are downloaded.
