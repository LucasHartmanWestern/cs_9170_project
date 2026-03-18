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
Median Phase 1 `local_reward` = 0.12 (low, not saturated at 1 as initially feared).
Generated samples land near the training distribution (delta_scale=0.10) where beta already
has low loss → low DVRL → weak gradient signal in early episodes. Phase 1 local-global
correlation is **+0.765** (strong alignment), so DVRL is correct in principle. The weakness
is spatial: samples are too close to training data to produce meaningful DVRL variance.

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

**Status:** v17a/v17b specs ready, code changes merged — awaiting submission
