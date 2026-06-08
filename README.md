# FORGE: Fairness-Oriented Reward-Guided Generation

FORGE is a generative augmentation framework for improving classifier fairness under *positive-class outcome scarcity* — a regime where standard reweighting methods degrade because there are too few minority-group positive examples to reweight from. The framework uses REINFORCE to generate synthetic minority-positive training samples in PCA space, guided by a worst-group-loss reward signal.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. GPU recommended (CUDA) but CPU works for smoke tests.

## Datasets

Two datasets are required before running any experiments.

### Census Income (UCI Adult)

Download directly from the UCI repository:

```bash
mkdir -p datasets/census+income
curl -o datasets/census+income/adult.data \
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
```

### Capture-24 (Oxford wearable accelerometer)

The raw dataset is ~6.9 GB. The download script fetches a 40-participant subset, extracts sliding-window features, saves a compact cache (~100–300 MB), then deletes the raw files.

```bash
python scripts/download_capture24.py --data-dir datasets/capture24
```

This requires `wget` or `curl` and takes 10–30 minutes depending on connection speed. The resulting cache (`datasets/capture24/capture24_features_cache.npz`) is all that the training pipeline needs.

## Quick Start

```bash
# Run FORGE on Census Income (single seed, paper config)
python main.py --spec experiment_specs/census/paper_final_seed42.yaml --device cuda:0

# Run FORGE on Capture-24 (fold 0, paper config)
python main.py --spec experiment_specs/capture24/paper_final_fold0.yaml --device cuda:0

# Run baseline comparisons
python run_baseline.py --spec experiment_specs/census/baselines/group_dro.yaml --device cpu

# Analyse a completed run
python analysis/check_run.py training_runs/<run_dir>
```

Results are written to `training_runs/<spec_name>_<hash>/seed_<N>/`.

## Project Structure

```
.
├── main.py                      # Entry point — loads spec, launches Training
├── run_baseline.py              # Run GroupDRO, FLB, CTGAN, OT Repair, SMOTE
│
├── FORGE/                       # Core pipeline
│   ├── training.py              # Training class (RL loop, reward, checkpointing)
│   ├── env.py                   # Environment (delta-action walk, PCA clipping)
│   ├── dataset.py               # Dataset class (census_income, capture24, k-fold)
│   ├── episode_tracker.py       # CSV logging, checkpointing, console mirroring
│   └── test_suite.py            # Post-training evaluation (EO, DP, F1, AUC)
│
├── agents/
│   ├── reinforce_agent.py       # REINFORCE policy gradient agent
│   └── ffnn_agent.py            # Feed-forward classifier (alpha / beta models)
│
├── utilities/
│   ├── reward_helpers.py        # Loss functions, fairness metrics, reward
│   ├── spec_helpers.py          # Spec loading and experiment group naming
│   ├── make_spec.py             # Generate spec variants from a base config
│   └── make_search_specs.py     # Generate grid/random hyperparameter search specs
│
├── benchmarks/                  # Baseline implementations
│   ├── group_dro.py
│   ├── fairness_loss_balancing.py
│   ├── gaussian_ot_repair.py
│   ├── smote_baseline.py
│   ├── ctgan_baseline.py
│   └── fairtabddpm_baseline.py
│
├── analysis/                    # Post-run analysis scripts
│   ├── check_run.py             # Learning curves, summary table, gen curves
│   ├── analyze_grid.py          # Aggregate grid-search results
│   ├── analyze_kfold.py         # Aggregate k-fold results (FORGE + baselines)
│   ├── analyze_reward_signal.py
│   └── eval_checkpoint.py
│
├── figure_generation_scripts/   # Reproduce paper figures (Figures 2–5)
│
├── scripts/
│   └── download_capture24.py    # Capture-24 dataset acquisition
│
├── experiment_specs/            # YAML configs + SLURM batch files
│   ├── census/                  # Census Income specs
│   │   ├── paper_final_seed42.yaml
│   │   ├── grid_search/
│   │   ├── random_search/
│   │   └── baselines/
│   └── capture24/               # Capture-24 specs
│       ├── paper_final_fold0.yaml
│       ├── grid_search/
│       ├── random_search/
│       └── baselines/
│
└── search_configs/              # Hyperparameter search space definitions
```

## Configuration

Experiment specs are YAML files. A canonical vanilla configuration is embedded in `utilities/make_spec.py` (`VANILLA_CONFIG`). Generate variants with:

```bash
# Patch a single parameter
python utilities/make_spec.py --base vanilla --name my_exp \
    --patch dataset_name=census_income da_pct=0.01433 minority_id=0 majority_id=1

# Sweep over a parameter
python utilities/make_spec.py --base experiment_specs/census/paper_final_seed42.yaml \
    --sweep ffnn.epochs 10 20 30

# k-fold specs (capture24)
python utilities/make_spec.py --base experiment_specs/capture24/grid_search/base.yaml \
    --fold-sweep 5
```

## Reproducing Paper Results

Paper configs are provided directly:

| Dataset | Spec | Seeds/Folds |
|---------|------|-------------|
| Census Income | `experiment_specs/census/paper_final_seed42.yaml` | seeds 0, 1, 3, 42, 99 |
| Capture-24 | `experiment_specs/capture24/paper_final_fold0.yaml` | folds 0–4 |

Run all seeds/folds, then aggregate:

```bash
# Aggregate Census results
python analysis/check_run.py training_runs/<census_run_dir>

# Aggregate Capture-24 k-fold results
python analysis/analyze_kfold.py training_runs/<c24_run_dir_fold0> \
    training_runs/<c24_run_dir_fold1> ...
```

Baselines use the same FFNN classifier and dataset splits. Run via:

```bash
python run_baseline.py --spec experiment_specs/census/baselines/<method>.yaml --device cpu
```
