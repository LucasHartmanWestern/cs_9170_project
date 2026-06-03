# FORGE: Fairness-Oriented Reward-Guided Generation

FORGE is a generative augmentation framework for improving classifier fairness under *positive-class outcome scarcity* — a regime where standard reweighting methods degrade because there are too few minority-group positive examples to reweight from. The framework uses REINFORCE to generate synthetic minority-positive training samples in PCA space, guided by a worst-group-loss reward signal.

## Quick Start

```bash
pip install -r requirements.txt

# Run FORGE on a dataset
python main.py --spec experiment_specs/census_grid/census_k10_r04_e30.yaml --device cuda:0

# Run a baseline comparison
python run_baseline.py --spec experiment_specs/capture24_rand10_baselines/group_dro_fold0.yaml --device cuda:0

# Analyse results
python analysis/check_run.py training_runs/<run_dir>
```

## Project Structure

```
.
├── main.py                  # Entry point — loads spec, launches Training
├── training.py              # Core Training class (RL loop, reward, logging)
├── env.py                   # Environment (delta actions, PCA clipping)
├── dataset.py               # Dataset class (census_income, capture24, k-fold)
├── reward_helpers.py        # Loss functions, fairness metrics, reward components
├── episode_tracker.py       # CSV logging, checkpointing, console mirroring
├── test_suite.py            # Post-training evaluation (EO, DP, F1, AUC)
├── spec_helpers.py          # Spec loading, validation, experiment group naming
├── run_baseline.py          # Run GroupDRO, FLB, CTGAN, OT Repair and other baselines
├── vanilla_config.json      # Base configuration — all experiments are deltas from this
│
├── agents/
│   ├── reinforce_agent.py   # REINFORCE policy gradient (primary RL agent)
│   ├── ffnn_agent.py        # Feed-forward classifier (alpha / beta models)
│   └── cmaes_agent.py       # CMA-ES evolution strategy (ablation baseline)
│
├── benchmarks/              # Baseline method implementations
│   ├── group_dro.py
│   ├── fairness_loss_balancing.py
│   ├── gaussian_ot_repair.py
│   ├── smote_baseline.py
│   ├── ctgan_baseline.py
│   └── fairtabddpm_baseline.py
│
├── analysis/                # Post-run analysis scripts
│   ├── check_run.py         # Learning curves, generalizability curves, summary table
│   ├── analyze_grid.py      # Aggregate grid-search results across configs
│   ├── analyze_kfold.py     # Aggregate k-fold results (FORGE + baselines)
│   ├── analyze_reward_signal.py
│   └── eval_checkpoint.py
│
├── tools/                   # Experiment setup utilities
│   ├── make_spec.py         # Generate spec variants from a base spec
│   ├── make_search_specs.py # Generate grid/random hyperparameter search specs
│   └── dataset_viability.py # Pre-flight dataset structural checks
│
├── paper_figures/           # Paper figure generation scripts (Figures 2–5)
│
├── experiment_specs/        # Experiment configurations (YAML + SLURM scripts)
│   ├── census_grid/         # Stage 1 grid search — Census Income
│   ├── census_random/       # Stage 2 random search — Census Income
│   ├── capture24_kfold_grid/    # Stage 1 grid search — Capture-24
│   ├── capture24_random/        # Stage 2 random search — Capture-24
│   ├── capture24_rand10_baselines/  # Baseline specs — Capture-24
│   └── capture24_rand10_folds34/    # Final FORGE fold specs — Capture-24
│
├── search_configs/          # Hyperparameter search space definitions
└── scripts/                 # Data acquisition scripts
    └── download_capture24.py
```

## Datasets

Two datasets are evaluated:

| Dataset | Protected attr | Disadvantaged group | DA+ |
|---------|---------------|---------------------|-----|
| Census Income (UCI Adult) | sex | female | ~43 |
| Capture-24 (Oxford wearable) | sex | female | ~60 |

Both datasets are subsampled so that the disadvantaged group has approximately 1–2% positive-class rate in training, simulating the scarcity regime studied in the paper.

## Configuration

All experiments are defined as deltas from `vanilla_config.json`:

```bash
# Generate a spec variant
python tools/make_spec.py --base vanilla_config.json --name my_exp \
    --patch dataset_name=census_income global_sigmoid_k=10

# Generate a hyperparameter search
python tools/make_search_specs.py search_configs/census_random.yaml
```

## Reproducing Paper Results

See `EXPERIMENTS.md` for the full experiment log with configurations and results for every experiment reported in the paper.

Key experiment specs:
- **Census Income** — `experiment_specs/census_grid/` (Stage 1) + `experiment_specs/census_random/` (Stage 2)
- **Capture-24** — `experiment_specs/capture24_kfold_grid/` (Stage 1) + `experiment_specs/capture24_random/` (Stage 2)
- **Baselines** — `experiment_specs/capture24_rand10_baselines/`
