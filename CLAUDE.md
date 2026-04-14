# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. Paper Goal

**Target venue:** Neurocomputing (Elsevier journal)

**Core contribution:** A generative augmentation framework for improving classifier fairness under *positive-class outcome scarcity* — a regime where standard reweighting methods (GroupDRO, OT Repair) degrade because there are too few minority-group positive examples to reweight from. The framework uses REINFORCE to generate synthetic minority-positive training samples in PCA space, guided by a worst-group-loss reward signal.

### Claims

1. **Motivation claim** — Reweighting baselines (GroupDRO) fail under severe positive-class scarcity (DA+ ≤ 43 training examples). We do not. Well-supported by v18 results on census.

2. **Competitive performance claim** — Our method achieves comparable or better fairness-utility tradeoff vs all baselines including CTGAN. v18 at census (EO=0.063±0.059, F1w=0.811, AUC=0.877) is the current best. Beats GroupDRO and CTGAN on both axes; near-OT-Repair EO with much better utility.

3. **Ablation / design validation claim** — Under active development. v17a (DVRL local reward) vs v18 (global-only) was the prior ablation. Sigmoid (v18, k=10) confirmed better than nosigmoid (v19, k=0) on census: v18 β-EO=0.079±0.015 vs v19 β-EO=0.093±0.068, 3/3 vs 2/3 seeds improved.

### Current Status

Dataset selection complete (census_income, capture24 confirmed; 3rd dataset paused pending capture24 stabilization).

**Best confirmed results (census, 3 seeds, 3000 ep):**
- v18 sigmoid k=10: β-EO=0.079±0.015, F1w=0.810, Deadzone=2.3%, 3/3 seeds improved
- v19 nosigmoid k=0: β-EO=0.093±0.068, F1w=0.801, Deadzone=100%, 2/3 seeds improved → v18 wins

**Capture24 status:** Unstable across seeds. α-EO ranges 0.06–0.33 across seeds at bias_pct=0.02, causing 1/5 seeds to improve with v18 k=3. Root cause under investigation.

**Active investigation leads:**
1. **Capture24 stabilization** — α-EO instability across seeds is the primary blocker. Need to diagnose whether this is a data split issue or bias injection issue.
2. **Evolutionary search** — REINFORCE is structurally a trajectory-level bandit (all 2000 steps get identical reward). CMA-ES or NES is theoretically better suited and under consideration.

### Datasets

**Active (paper):** census_income, capture24.

**DA+** = number of disadvantaged-group positive (y=1) training examples. Both datasets are configured so DA+ ≈ 43–45, the level at which reweighting methods demonstrably fail. `da_pct` is an internal implementation parameter (fraction of train set that should be disadvantaged-group positives) — do NOT use it in paper text. Always frame in terms of DA+ or positive-class rate percentages (e.g., "~11% for the disadvantaged group").

`da_pct` uses group-specific subsampling: only the disadvantaged group's positives are reduced; advantaged-group positives and all negatives are kept intact. This gives identical DA+ across all seeds. The old `bias_pct` parameter (which subsampled all positives and produced variable DA+ across seeds) is retained for backward compatibility with archived runs.

**DA+ log — confirmed values (seed=42, mean across seeds similar):**

| Dataset | da_pct | real_data_size | DA+ | Protected attr | Disadv. group |
|---------|--------|----------------|-----|----------------|---------------|
| census_income | 0.01433 | 3000 | **43** | sex | female (a=0) |
| capture24 | 0.015 | 3000 | **45** | sex | female (a=1) |

**Dataset selection criteria** (all four must pass before committing to RL experiments):
1. val_disadv_pos ≥ 30 — stable reward signal
2. test_disadv_pos ≥ 200 — reliable fairness evaluation
3. alpha_EO ≥ 0.10 — meaningful pre-intervention gap
4. Feature-space distinctiveness — disadvantaged-group positives must be spatially distinct from advantaged-group positives in PCA space; otherwise synthetic samples cannot carry group-specific signal

**Dropped datasets:**
- **COMPAS** — val_pos=14 (below threshold), positive-class overlap in PCA space, RL results contradict motivation claim
- **PAMAP2** — only 1 female subject, unstable group identification across seeds
- **credit_card** — DA+ too high (~136), alpha-EO near-zero, not in scarcity regime
- **PTB-XL** — no framing reaches test_pos≥200
- **MEPS** — val_pos=37 marginal; ethnicity framing test_pos=111 fails evaluation threshold

### Paper-Writing Rules

- Do not use em-dashes in prose. Use commas, semicolons, or restructure.
- Do not make model-agnostic claims — current results use a fixed FFNN classifier.
- Do not introduce specific numeric thresholds without a citation. Use qualitative language.
- Frame scarcity in terms of positive-class rate percentages, not raw DA+ counts.
- Reviewer framing on DA+ realism: DA+≈43 is realistic (hospital rare-condition datasets, small-jurisdiction criminal justice records, internal HR data). Our bias injection simulates this condition, following standard practice in fairness ML.
- Reviewer questions to anticipate: Why RL over simpler generative methods? Why PCA space? Does the agent actually learn vs random search? Have experiments or arguments ready for each.

When assessing results, prioritise in this order:
1. Does this strengthen or weaken a specific claim? Flag ambiguous results explicitly.
2. Is it reproducible? 3 seeds is provisional; 5+ seeds required for final results table.
3. Is the ablation clean? Note any confounds if multiple things changed between versions.

---

## 2. Codebase

### Framework Overview

Three-model system: **alpha** (trained on real data only, fairness baseline), **beta** (trained on real + synthetic, the target model), **RL agent** (REINFORCE, generates synthetic minority-positive samples by perturbing PCA-space coordinates via delta actions).

Each training episode: generate synthetic trajectory → train beta on real+synthetic → compute reward (worst-group-loss improvement) → update RL agent via policy gradient → log diagnostics.

### Reward Structure (current best: v18 global-only)

- **Global term**: `sigmoid(k × (wgl_alpha − wgl_beta))` where wgl = worst-group BCE loss on validation. k=10 (vanilla). Range (0,1); above 0.5 means beta is better than alpha. k=0 gives normalized reward: (wgl_alpha − wgl_beta) / wgl_alpha.
- **Lambda schedule**: `λ * global + (1−λ) * local`. Vanilla uses λ=1.0 (pure global, no local).
- **gamma=1.0**: All 2000 trajectory steps contribute equally. gamma=0.99 caused ~50% deadzone and was abandoned.
- **Local term (DVRL)**: Disabled. `use_dvrl_local=false` in vanilla.
- **Curriculum**: Disabled. Start directly in full 10D PCA space.
- **gen_both_classes**: Disabled in vanilla. When enabled: phase 1 = minority augmentation, phase 2 = majority recovery.

### Core Modules

| File | Purpose |
|------|---------|
| `training.py` | Core `Training` class with full training loop |
| `env.py` | `Environment` class — delta actions, radius clipping, curriculum (disabled) |
| `dataset.py` | `Dataset` class — census_income, capture24, compas, ptb_xl. Bias injection, PCA encoding, stratified splitting. Stores `self.pca_transform` after `get_data_splits`. |
| `reward_helpers.py` | Loss functions, fairness metrics (EO gap, worst-group loss), local reward components |
| `episode_tracker.py` | `EpisodeTracker` context manager — CSV logging, checkpointing, console mirroring |
| `test_suite.py` | `TestSuite` class — post-training evaluation (DP, EO, EOd, F1, Brier). Not a pytest file. |
| `agents/` | `FFNNAgent` (classifier), `ReinforceAgent` (policy gradient), `PPOAgent` (inactive) |

### Analysis Scripts

| File | Purpose |
|------|---------|
| `check_run.py` | Standard post-run analysis — summary table, learning curves, generalizability curves |
| `dataset_viability.py` | Dataset structural viability checker — DA+ scan, alpha-EO baseline, feature separability |
| `make_spec.py` | Generate experiment spec JSON + SLURM batch files from a base spec with `--patch` / `--sweep` |
| `run_baseline.py` | Run GroupDRO, FLB, CTGAN, OT Repair, and other baselines |

### Notebooks

| File | Purpose |
|------|---------|
| `plot_results.ipynb` | Paper figures — fairness/utility bar charts and tradeoff table. Edit `DATASETS` dict in cell 6 to add new runs. |
| `analyze_datasets.ipynb` | Dataset statistics, bias injection effects, group imbalance summaries |
| `dataset_visualizations.ipynb` | PCA projections and class/group distribution plots |
| `visualize_generation.ipynb` | Visualises where the RL agent places synthetic points in PCA space across training. Use to answer "does the agent actually learn?" |

### Dataset Implementation Details

- **census_income**: Adult income (UCI). Protected attr: sex. Female (a=0) disadvantaged. Path: `datasets/census+income/adult.data`.
- **capture24**: Wearable accelerometer (Oxford). Protected attr: sex. Female (a=1) disadvantaged. Requires windowing: `win_seconds=1.0`, `step_seconds=0.5`. Path: `datasets/capture24/`.

### Output Structure

Results go to `training_runs/SPEC_{name}_{hash}__G{timestamp}/seed_{N}/`:

| File | Contents |
|------|----------|
| `metrics.csv` | Per-episode metrics (50+ columns, prefixes: global, utility, fairness, local, extra, align) |
| `final_test_metrics.csv` | Final test-set fairness/utility evaluation |
| `meta.json` | Run config: dataset, seed, bias_pct, episodes, global_sigmoid_k, win_seconds, etc. |
| `best_beta_meta_phase1_class1.json` | Best checkpoint episode + metric value for phase 1 |
| `synthetic_snapshots/synthetic_ep{N:04d}_phase1_class1.npz` | Synthetic data snapshots every 5 episodes |
| `best_beta_state_dict_phase1_class1.pt`, `alpha_state_dict.pt` | Model weights |
| `analysis/` | Created by `check_run.py` — `summary.txt`, `fig_learning.png`, `fig_gen_curve.png` |

**Directory layout:**
- `training_runs/` — active runs from current experiments
- `archive_runs/` — all runs prior to April 2026 cleanup

---

## 3. Research Process

### Experiment Log

See **`EXPERIMENTS.md`** for the full record of every experiment: config delta from vanilla, results, takeaway, next steps. **Update EXPERIMENTS.md after every significant result.** Every number cited in the paper must be traceable to a specific entry there.

### Vanilla Config

`vanilla_config.json` (project root) is the canonical base for all experiments. All specs are deltas from this. Do NOT modify it unless a new result explicitly establishes a better default across both datasets.

### Workflow: Designing a New Experiment

1. Identify the question being asked and which paper claim it bears on.
2. Write an EXPERIMENTS.md entry (PLANNED status) with the config delta from vanilla and purpose.
3. Generate the spec: `python make_spec.py --base vanilla_config.json --name <name> --patch key=value ...`
4. For new datasets: first run `python dataset_viability.py ...` and confirm all four criteria pass.

### Workflow: Running Experiments

```bash
# Local
python main.py --spec experiment_specs/<spec>.json --device cuda:0

# SLURM (user submits manually on DRAC — Claude Code does not submit jobs)
sbatch experiment_specs/<spec>.sh
```

DRAC workflow: prepare spec and `.sh` batch files locally, user submits on DRAC, user downloads results and notifies Claude Code to begin analysis. Do not ask the user to run commands on DRAC.

### Workflow: Analysing Results

Run after every completed experiment:
```bash
python check_run.py training_runs/<run_dir> [--interval 150] [--device cpu] [--no-gen-curve]
```

Outputs go to `<run_dir>/analysis/`:
- `summary.txt` — per-seed and mean α/β EO, F1w, AUC, EO-Δ, deadzone %, best checkpoint episode
- `fig_learning.png` — episode return + val EO per seed + mean band (phase 1 only)
- `fig_gen_curve.png` — test-set EO/F1w/AUC vs episode at `--interval` snapshot intervals

Key diagnostics to check:
- **Seeds improved** — how many seeds show β-EO < α-EO. Below 3/5 warrants investigation before claiming the config works.
- **Deadzone %** — fraction of phase-1 episodes where global_obj < 0.5 (applies only to sigmoid reward; nosigmoid will always show ~100%). Above ~20% indicates the reward signal is too weak.
- **Best checkpoint episode** — if consistently near episode 0 or episode max, the agent is not converging cleanly.
- **α-EO variance** — high variance across seeds means the data split is unstable, not the RL config.

### Workflow: Recording Results

Update the EXPERIMENTS.md entry (change status to COMPLETE, fill in Result, Takeaway, Next steps). Any number that will appear in the paper should be in EXPERIMENTS.md first.

### Spec Naming and Organisation

- Specs live in `experiment_specs/`. Each has a `.json` and a `.sh` SLURM batch file.
- Use `make_spec.py` to generate; only specify fields that differ from vanilla.
- `dp_protected_col` selects the protected attribute column. Omit to use each dataset's natural default.

### SLURM Defaults (DRAC rorqual)

1 CPU, 3 GB RAM, 1 thread per library. See existing `.sh` files for the template.
