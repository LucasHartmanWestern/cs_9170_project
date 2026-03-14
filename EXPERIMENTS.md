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

### v7 — Hard-Positive Anchors ⬜ Next to run

**Goal:** Fix the root cause of local/global reward misalignment identified in Training
Dynamics Analysis Finding 4. Make anchor proximity and hard_reward point at the same region.

**Key insight:** Current "all" anchor selection includes easy minority-positive training
points where alpha is already confident (p_alpha ≈ 1). `anchor_reward` then pulls generation
toward these easy regions. But `hard_reward` rewards generating where alpha is uncertain
(p < 0.65). These two terms pull in opposite directions; since `w_anchor=0.80` dominates,
the agent ends up near easy anchors where hard_reward ≈ 0. This structural tension is why
`corr(anchor_reward, EO) = +0.36` in credit run 3 — the anchor reward is tracking the
wrong thing.

**Fix:** `anchor_selection_mode: "hard_positive"` selects the N most misclassified
minority-positive training points (lowest p_alpha). Now anchor proximity and hard_reward
both target the decision boundary region → local reward coherently aligned with EO.
Combined with `sigma_calibration_factor: 1.0` to prevent anchor death.

**Specs created:**

| Spec | Runs | Purpose |
|---|---|---|
| `v7_census_hard_anchors_smoke` | 1 seed, 200 eps | Verify anchor_reward stays healthy and correlates with EO |
| `v7_credit_hard_anchors_smoke` | 1 seed, 200 eps | Same for credit |
| `v7_census_hard_anchors` | 3 seeds, 6000 eps | Full run for census |
| `v7_credit_hard_anchors` | 3 seeds, 6000 eps | Full run for credit |

All use: `use_pca=true`, `lambda=[0.5,0.5]`, `sigma_calibration_factor=1.0`,
`use_uncertainty_anchors=false` — isolates the anchor selection change cleanly.

**Run order:** both smoke tests first (fast, ~10 min each), then check
`corr(anchor_reward, EO)` in metrics.csv. If negative (aligned), launch full runs.

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
