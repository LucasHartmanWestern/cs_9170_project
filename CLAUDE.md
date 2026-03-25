# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Paper Goal

**Target venue:** Neurocomputing (Elsevier journal)

**Core contribution:** An RL-based framework for generating synthetic training data that improves classifier fairness under *positive-class outcome scarcity* — a regime where standard reweighting methods (GroupDRO, OT Repair) degrade because there are too few minority positive examples to reweight from. This is an underexplored area; existing fairness literature focuses on reweighting or in-processing, not generative augmentation under severe label scarcity.

**Three claims that must be supported by results:**

1. **Motivation claim** — Reweighting baselines (GroupDRO, OT Repair) fail under severe positive-class scarcity (bias_pct ≤ 0.10). We do not. Well-supported by v18 results.

2. **Competitive performance claim** — Our method achieves comparable or better fairness-utility tradeoff vs all baselines including CTGAN. v18 at census bias=0.10 (EO=0.063±0.059, F1w=0.811, AUC=0.877) is the best result. Beats GroupDRO and CTGAN on both axes; near-OT-Repair EO with much better utility.

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
**Do NOT change these unless a new experiment explicitly beats v18.** Apply this config to all new datasets (CAPTURE-24, etc.) — only change dataset-specific fields (dataset_name, bias_pct, minority_id, win_seconds, step_seconds).

**Current paper status:** v18 is the confirmed primary method. Main results (census) done. CAPTURE-24 results in progress (2026-03-25). Credit card results needed.

**Results still needed before submission:**
- CAPTURE-24 5-seed v18 run (in progress locally, 2026-03-25)
- v18 credit bias=0.10 and bias=0.05 — 5 seeds each (queue for DRAC)
- v18 census bias=0.05 — 5 seeds (queue for DRAC)

## Experiment Log

See **`EXPERIMENTS.md`** for the full chronological record of every experiment: configs run, results, what worked, what failed and why, and planned next steps. **Update EXPERIMENTS.md after every significant result.** It serves as the ground truth for paper claims — any number cited in the paper should be traceable to a specific entry there.

## Project Overview

Fairness-aware synthetic data generation using reinforcement learning. An RL agent (REINFORCE) learns to generate synthetic minority-positive training samples that improve classifier fairness (Equal Opportunity gap) while preserving utility (F1-weighted, AUC). The problem setting is *positive-class outcome scarcity*: the disadvantaged group has very few positive-class examples due to simulated historical bias (`bias_pct` downsampling), making reweighting-based baselines ineffective.

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
- **`dataset.py`**: `Dataset` class handling census_income, credit_card, PAMAP2. Includes bias injection, PCA encoding, stratified splitting
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

### Output Structure
Results go to `training_runs/SPEC_{name}_{hash}__G{timestamp}/seed_{N}/` containing:
- `metrics.csv`: Per-episode metrics (50+ columns, flattened from nested dicts with prefixes: global, utility, fairness, local, extra, align)
- `test_results.json`: Final fairness/utility evaluation
- `best_synthetic.npz`: Best synthetic data checkpoint
- `best_beta_state_dict.pt`, `alpha_state_dict.pt`: Model weights

## Datasets

- **census_income**: Adult income with sex/race/age/country protected attributes. Primary dataset for all fairness experiments. Most results and ablations are on this dataset.
- **credit_card**: Credit default with sex/age protected attributes. Secondary fairness dataset. Alpha-EO is near-zero at both bias levels so it contributes primarily to the utility preservation story, not the fairness story.
- **PAMAP2**: Physical activity recognition. DROPPED — full 5-seed run at bias=0.10 showed mean EO degradation (α=0.151 → β=0.191). Root cause: only 1 female subject (sub102), 9 female positive examples at bias=0.10, disadvantaged group identification flips across seeds. Not suitable for the positive-class scarcity narrative.

- **Third dataset (IoT/wearable, REQUIRED for funding)**: Replacement for PAMAP2 pending. See "Third Dataset — Work In Progress" note below.

Bias is injected via `bias_pct` parameter which downsamples y=1 (positive class) examples to simulate historical outcome bias. At bias_pct=0.05 only 17 minority-positive training examples remain; at 0.10, 43 remain (census numbers).

**Dataset priority:** Validate and finalise all census and credit results first. Third dataset (IoT) comes last.

## Third Dataset — Work In Progress (2026-03-25)

**Funding requirement:** Third dataset must be IoT / wearable sensor data (not tabular). PAMAP2 dropped; replacement needed.

**PTB-XL ECG (paused, not confirmed for use):**
- 21,799 records, sex-balanced (11,354 male / 10,445 female)
- Binary: MI (y=1) vs NORM (y=0), 13,572 records post-filter
- Female MI rate naturally lower (23.2% vs 37.4% male) → positive-class scarcity narrative
- Implementation COMPLETE: `split_ptb_xl()` in `dataset.py`, strat_fold-based splits (no patient leakage), feature cache, all experiment specs created
- Signal files: 13,395/13,572 downloaded to `datasets/ptb-xl/records100/` (98.7% complete — ~177 files remain, can resume with filelist at `/tmp/ptbxl_filelist.txt`)
- **Status: PAUSED** — ECG is clinical, not traditional IoT; user reviewing whether this satisfies funding requirement
- To resume: delete partial cache (`datasets/ptb-xl/ptbxl_features_cache.npz` if it exists), finish download, run `smoke_ptbxl_bias010_rl.json`

**Open question for user:** Does PTB-XL (clinical ECG) qualify as "IoT" for funding? If not, alternative IoT options being evaluated (see EXPERIMENTS.md).

## Paper-Writing Guidance

When making decisions about experiments, code changes, or analysis, prioritise in this order:
1. **Does this strengthen or weaken a specific paper claim?** If a result is ambiguous, flag it explicitly rather than presenting it optimistically.
2. **Is it reproducible?** Always report seed count, mean ± std, and range. 3 seeds is provisional; 5+ seeds is required for the final results table.
3. **Is the ablation clean?** Change one thing at a time between compared configs. If multiple things changed between versions, note the confounds.
4. **Reviewer questions to anticipate:** Why RL over simpler generative methods? Why DVRL local reward vs no local reward? Why PCA space? Does the agent actually learn (vs random search)? Have answers or experiments ready for each.
