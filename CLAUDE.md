# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Paper Goal

**Target venue:** Neurocomputing (Elsevier journal)

**Core contribution:** An RL-based framework for generating synthetic training data that improves classifier fairness under *positive-class outcome scarcity* — a regime where standard reweighting methods (GroupDRO, OT Repair) degrade because there are too few minority-group positive examples to reweight from. This is an underexplored area; existing fairness literature focuses on reweighting or in-processing, not generative augmentation under severe label scarcity.

**Three claims that must be supported by results:**

1. **Motivation claim** — Reweighting baselines (GroupDRO, OT Repair) fail under severe positive-class scarcity (DA+ ≤ 43 training examples). We do not. Well-supported by v18 results on census.

2. **Competitive performance claim** — Our method achieves comparable or better fairness-utility tradeoff vs all baselines including CTGAN. v18 at census (EO=0.063±0.059, F1w=0.811, AUC=0.877) is the best result. Beats GroupDRO and CTGAN on both axes; near-OT-Repair EO with much better utility.

3. **Ablation / design validation claim** — v17a (DVRL local reward) vs v18 (global-only) is the key ablation. Result is a clean negative: global-only is better. Paper narrative: global-only is the proposed design; DVRL is tested and shown not to improve, validating the simpler design.

**Current best config — v18 (global-only):**
```
reward_mode: fairness
lambda_schedule: [1.0, 1.0]          ← pure global reward, no local
use_dvrl_local: false
curriculum_learning: false
gen_both_classes: true
phase2_episodes: 200
gamma: 1.0
delta_scale: 0.1
delta_clip: 0.2
radius_clip: 3.0
traj_length: 2000
real_data_size: 3000
total_episodes: 800
ffnn: hidden=[32,16], lr=0.001, batch=64, epochs=20
reinforce: hidden=[64,64], lr=0.0003, entropy_start=0.02, entropy_end=0.005
pca_components: 10
```
**Do NOT change these unless a new experiment explicitly beats v18.** Apply this config to all new datasets — only change dataset-specific fields (dataset_name, bias_pct, dp_protected_col, minority_id, win_seconds, step_seconds).

**Current paper status:** v18 is the confirmed primary method. Main results (census, CAPTURE-24) done. COMPAS race experiments validated (2026-03-30) — alpha-EO≈0.41 confirmed across 5 seeds, all 7 job specs ready. Credit dataset DROPPED.

**Results still needed before submission:**
- COMPAS race 5-seed full run — all methods (queue for DRAC, specs ready as of 2026-03-30)
- v18 census bias=0.05 — 5 seeds (queue for DRAC)

## Experiment Log

See **`EXPERIMENTS.md`** for the full chronological record of every experiment: configs run, results, what worked, what failed and why, and planned next steps. **Update EXPERIMENTS.md after every significant result.** It serves as the ground truth for paper claims — any number cited in the paper should be traceable to a specific entry there.

## Project Overview

Fairness-aware synthetic data generation using reinforcement learning. An RL agent (REINFORCE) learns to generate synthetic minority-positive training samples that improve classifier fairness (Equal Opportunity gap) while preserving utility (F1-weighted, AUC).

**Problem setting:** *positive-class outcome scarcity* — the disadvantaged group has very few positive-class examples in the training data (DA+ ≤ ~50), making reweighting-based baselines ineffective. Each dataset is configured via `bias_pct` (an internal implementation parameter) to achieve a target DA+ count. Do NOT lead with `bias_pct` in paper text — always frame in terms of DA+.

## Commands

### Run an experiment
```bash
python main.py --spec experiment_specs/<spec>.json --device cuda:0
```
Falls back to CPU automatically if CUDA is unavailable.

### Run on SLURM cluster
```bash
sbatch experiment_specs/<spec>.sh
```
Each spec has a corresponding `.sh` batch file in `experiment_specs/`. Requires `~/envs/rl` virtualenv.

**Important workflow note:** `.sh` batch files are submitted manually by the user on the DRAC cluster. Claude Code does not submit jobs directly. When results are ready, the user downloads them locally and notifies Claude Code to begin analysis. Do not ask the user to run commands on DRAC — just prepare the spec and batch files.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run tests
There is no automated test suite. `test_suite.py` is a post-training evaluation module (fairness/utility metrics), not a pytest test file.

## Architecture

### Three-Model System
1. **Alpha model** (`FFNNAgent`): Trained on real data only. Used to identify disadvantaged groups and compute fairness baselines.
2. **Beta model** (`FFNNAgent`): Trained on real + synthetic data. The target model whose fairness the RL agent tries to improve.
3. **RL Agent** (`ReinforceAgent`): Generates synthetic samples by perturbing PCA-space coordinates via delta actions. PPO exists in the codebase but is not active.

### Training Loop (training.py)
Each episode: generate synthetic trajectory → train beta on real+synthetic → compute reward (fairness gap + local terms) → update RL agent via policy gradient → log diagnostics.

### Reward Structure (current best config: v18 global-only)
- **Global term**: `sigmoid(10 × (wgl_alpha − wgl_beta))` where wgl = worst-group BCE loss on validation. Range (0,1); above 0.5 means beta is better than alpha.
- **Lambda schedule**: `λ * global + (1−λ) * local`. v18 uses `λ=1.0` (pure global, no local). Reward per step = global / T.
- **Local term (DVRL)**: Disabled in v18. Tested in v17a and shown to hurt — drives agent toward decision boundary producing OOD samples that destabilize beta training in late episodes.
- **gamma=1.0**: All 2000 trajectory steps contribute equally to the policy gradient return. gamma=0.99 caused a ~50% deadzone and was abandoned.
- **Curriculum**: Disabled. Start directly in full 10D PCA space.
- **gen_both_classes=true**: Agent generates synthetic samples for both minority and majority class (phase 1 = minority, phase 2 = majority recovery, 200 episodes).

### Key Modules
- **`training.py`**: Core `Training` class with full training loop
- **`env.py`**: `Environment` class with delta actions, radius clipping, and curriculum learning (disabled in current best config — full 10D PCA from episode 1)
- **`dataset.py`**: `Dataset` class handling census_income, compas, capture24, ptb_xl. Includes bias injection, PCA encoding, stratified splitting
- **`reward_helpers.py`**: Loss functions, fairness metrics (EO gap, worst-group loss), local reward components
- **`episode_tracker.py`**: `EpisodeTracker` context manager for CSV logging, checkpointing, console mirroring
- **`test_suite.py`**: `TestSuite` class for post-training evaluation (DP, EO, EOd, F1, Brier)
- **`agents/`**: `FFNNAgent` (classifier), `ReinforceAgent` (policy gradient), `PPOAgent` (actor-critic)

### Notebooks
- **`plot_results.ipynb`**: Main results comparison notebook. Loads `final_test_metrics.csv` from selected `training_runs/` directories, produces fairness (EO/DP/EOd) and utility (F1/AUC) bar charts per dataset, and a full tradeoff table with UAFI scores. Edit the `DATASETS` dict in cell 6 to add new runs.
- **`dataset_visualizations.ipynb`**: PCA projections and class/group distribution plots for raw datasets.
- **`diagnose_training.ipynb`**: Per-episode `metrics.csv` analysis — reward curves, deadzone fraction, local/global correlation. Use this to check if a new run is learning before waiting for full results.
- **`analyze_datasets.ipynb`**: Dataset statistics, bias injection effects, group imbalance summaries.

### Experiment Configuration
JSON specs in `experiment_specs/` define all hyperparameters: dataset, reward mode, lambda schedule, PCA components, trajectory length, curriculum settings, network architectures, seeds. Multi-seed runs execute sequentially from a single spec.

The `dp_protected_col` field in a spec selects which column is used as the protected attribute (passed through `training.py` and all baseline trainers to `get_data_splits`). Defaults to each dataset's natural default if omitted.

### Output Structure
Results go to `training_runs/SPEC_{name}_{hash}__G{timestamp}/seed_{N}/` containing:
- `metrics.csv`: Per-episode metrics (50+ columns, flattened from nested dicts with prefixes: global, utility, fairness, local, extra, align)
- `test_results.json`: Final fairness/utility evaluation
- `best_synthetic.npz`: Best synthetic data checkpoint
- `best_beta_state_dict.pt`, `alpha_state_dict.pt`: Model weights

## Datasets

**Active datasets (paper):** census_income, capture24, compas. Credit card DROPPED.

### Scarcity Metric: DA+
**DA+** = number of disadvantaged-group positive (y=1) training examples. This is the primary metric defining the scarcity regime. All three datasets are configured so DA+ ≈ 43, the level at which reweighting methods demonstrably fail. `bias_pct` is an internal implementation parameter used to achieve the target DA+; it is NOT a paper-level framing variable.

**DA+ log — confirmed values (seed=42, mean across seeds similar):**

| Dataset | bias_pct | real_data_size | DA+ | Protected attr | Disadv. group |
|---------|----------|----------------|-----|----------------|---------------|
| census_income | 0.10 | 3000 | **43** | sex | female (a=0) |
| capture24 | 0.02 | 3000 | **45** | sex | female (a=1) |
| compas | 0.05 | — (no cap) | **~40** | race | Caucasian (a=0) |

Secondary scarcity level (census only, for motivation curve):

| Dataset | bias_pct | DA+ |
|---------|----------|-----|
| census_income | 0.05 | 17 |

### Dataset Details

- **census_income**: Adult income (UCI). Protected attr: sex. Female (a=0) is disadvantaged — naturally low positive rate (~7%) creates scarcity at bias_pct=0.10. Primary dataset; most ablations here. Path: `datasets/census+income/adult.data`.

- **capture24**: Wearable accelerometer sleep/activity (Oxford). Protected attr: sex. Female (a=1) is disadvantaged. Requires windowing (win_seconds=1.0, step_seconds=0.5). Path: `datasets/capture24/`. bias_pct=0.02 (real_data_size=3000 cap drives DA+≈45).

- **compas**: COMPAS recidivism (ProPublica). Protected attr: **race** (dp_protected_col="race"). Caucasian (a=0) is disadvantaged — lower recidivism positive rate vs African-American (~47%). Bias injection is group-specific: only Caucasian positives are reduced (AA positives kept at natural rate). bias_pct=0.05 → DA+≈40 (Caucasian positives). Alpha-EO gap ≈ 0.41 (strong racial disparity). Race is included as a feature in cat_cols_all. Path: `datasets/compas/compas-scores-two-years.csv`. Specs: `compas_race_ep1500ph400_5s.json` (RL), `compas_race_bias005_*_5s.json` (baselines).

- **PAMAP2**: DROPPED — disadvantaged group identification unstable across seeds (only 1 female subject).
- **credit_card**: DROPPED — DA+ too high (~136 at bias=0.10), alpha-EO near-zero, not in the scarcity regime.
- **ptb_xl**: PAUSED — ECG implementation complete; pending decision on whether it satisfies IoT funding requirement.

## Paper-Writing Guidance

**Style rules (apply to all paper text):**
- Do not use emphasis dashes (em-dashes) in prose. Use commas, semicolons, or restructure the sentence instead.
- Do not make model-agnostic claims unless directly supported by experiments (the current results use a fixed FFNN classifier; model-agnostic generality is not demonstrated).
- Do not introduce specific numeric thresholds (e.g., "fewer than 15%") without a citation. Use qualitative language instead.
- Frame scarcity in terms of positive-class rate percentages (e.g., "~11% for the disadvantaged group") rather than raw DA+ counts in paper text. DA+ is for internal tracking only.

When making decisions about experiments, code changes, or analysis, prioritise in this order:
1. **Does this strengthen or weaken a specific paper claim?** If a result is ambiguous, flag it explicitly rather than presenting it optimistically.
2. **Is it reproducible?** Always report seed count, mean ± std, and range. 3 seeds is provisional; 5+ seeds is required for the final results table.
3. **Is the ablation clean?** Change one thing at a time between compared configs. If multiple things changed between versions, note the confounds.
4. **Reviewer questions to anticipate:** Why RL over simpler generative methods? Why DVRL local reward vs no local reward? Why PCA space? Does the agent actually learn (vs random search)? Have answers or experiments ready for each.

**On the DA+ scarcity framing:** Reviewers may ask whether DA+≈43 is realistic. It is — hospital datasets for rare conditions, small-jurisdiction criminal justice records, and internal HR datasets routinely produce comparable minority-positive counts. The key point is that DA+≈43 is the regime where reweighting methods empirically fail (shown in our census baseline degradation curve); the specific mechanism creating that scarcity (historical underrecording, small group size, rare outcome) does not affect the algorithmic behavior we study. Framing: "we study the regime where DA+ is severely limited; our bias injection *simulates* this condition, following standard practice in fairness ML."
