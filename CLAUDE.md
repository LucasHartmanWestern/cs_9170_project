# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Experiment Findings

See **`EXPERIMENTS.md`** for the full record of what has worked, what has not, baseline
comparisons, and planned next experiments. Update it after each significant result.

## Project Overview

Fairness-aware synthetic data generation using reinforcement learning. An RL agent (REINFORCE/PPO) learns to generate synthetic training samples that improve classifier fairness (e.g., Equal Opportunity, Equalized Odds) while maintaining utility (accuracy/F1). Uses curriculum learning to progressively increase PCA dimensionality during training.

## Commands

### Run an experiment
```bash
python main.py --spec experiment_specs/<spec>.json --device cuda:0
```
Falls back to CPU automatically if CUDA is unavailable.

### Run on SLURM cluster
```bash
sbatch run.sh
```
Edit the `SPECS` array in `run.sh` to select which experiment specs to run. Requires `~/envs/rl` virtualenv.

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
3. **RL Agent** (`ReinforceAgent` or `PPOAgent`): Generates synthetic samples by perturbing PCA-space coordinates via delta actions.

### Training Loop (training.py)
Each episode: generate synthetic trajectory → train beta on real+synthetic → compute reward (fairness gap + local terms) → update RL agent via policy gradient → log diagnostics.

### Reward Structure
- **Global term**: Fairness objective (e.g., `exp(-worst_group_loss)`)
- **Local term**: Anchor proximity + hard-positive confidence + diversity penalty
- **Lambda schedule**: Interpolates `λ * global + (1-λ) * local` over training

### Key Modules
- **`training.py`**: Core `Training` class with full training loop
- **`env.py`**: `Environment` class with curriculum learning (progressive dim expansion), delta actions, radius clipping
- **`dataset.py`**: `Dataset` class handling census_income, credit_card, PAMAP2. Includes bias injection, PCA encoding, stratified splitting
- **`reward_helpers.py`**: Loss functions, fairness metrics (EO gap, worst-group loss), local reward components
- **`episode_tracker.py`**: `EpisodeTracker` context manager for CSV logging, checkpointing, console mirroring
- **`test_suite.py`**: `TestSuite` class for post-training evaluation (DP, EO, EOd, F1, Brier)
- **`agents/`**: `FFNNAgent` (classifier), `ReinforceAgent` (policy gradient), `PPOAgent` (actor-critic)

### Notebooks
- **`plot_results.ipynb`**: Main results comparison notebook. Loads `final_test_metrics.csv` from selected `training_runs/` directories, produces fairness (EO/DP/EOd) and utility (F1/AUC) bar charts per dataset, and a full tradeoff table with UAFI scores. Edit the `DATASETS` dict in cell 6 to add new runs.
- **`dataset_visualizations.ipynb`**: PCA projections and class/group distribution plots for raw datasets.
- **`diagnose_training.ipynb`**: Per-episode `metrics.csv` analysis — reward curves, anchor health, local/global correlation. Use this to check if a new run is learning before waiting for full results.
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
- **census_income**: Adult income with sex/race/age/country protected attributes
- **credit_card**: Credit default with sex/age protected attributes
- **PAMAP2**: Activity recognition (no protected attributes)

Bias is injected via `bias_pct` parameter which downsamples minority groups to create class imbalance.
