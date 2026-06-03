# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. Paper Goal

**Target venue:** Neurocomputing (Elsevier journal)

**Core contribution:** A generative augmentation framework for improving classifier fairness under *positive-class outcome scarcity* — a regime where standard reweighting methods (GroupDRO, OT Repair) degrade because there are too few minority-group positive examples to reweight from. The framework uses REINFORCE to generate synthetic minority-positive training samples in PCA space, guided by a worst-group-loss reward signal.

### Claims

1. **Motivation claim** — Naive generative baselines (CTGAN, FairTabDDPM) degrade under severe positive-class scarcity (DA+ ≈ 43): both show EO *higher* than alpha on capture24, and CTGAN EO=0.270 vs alpha EO≈0.36 on census (modest reduction). Reward-guided generation avoids this failure mode. Reweighting methods (GroupDRO, FLB) do not fail catastrophically — the claim is that RL achieves *better EO than all baselines* on census, not that reweighting collapses.

2. **Competitive performance claim** — FORGE achieves best EO of all methods on both confirmed datasets. Census: β-EO=0.018±0.005, F1w=0.817, AUC=0.876 (k=10, pca=10, ep=30, traj=2000; EXP-021). Beats GroupDRO (0.057), FairTabDDPM (0.081), OT Repair (0.162), FLB (0.247), CTGAN (0.270), SMOTE (0.285) — all baselines from EXP-050 (matched 20-epoch FFNN, da_pct protocol). Capture24: β-EO=0.069±0.030, F1w=0.948, AUC=0.945 (k=5, pca=15, ep=10, rand_0010 secondary params; EXP-049/EXP-050, 5 folds). Beats GroupDRO (0.122), CTGAN (0.128), FairTabDDPM (0.168), OT Repair (0.176), FLB (0.219), SMOTE (0.285) — EXP-050 baselines, k-fold protocol. α-EO=0.234±0.212 (GroupDRO reference, lr=0.001).

3. **Ablation / design validation claim** — Grid search (EXP-021) confirms sigmoid sharpness matters: k=10 (β-EO=0.018) substantially outperforms k=3 (0.039) and k=0 (0.039) on census. Global-only reward (no DVRL local term) confirmed by EXP-007/008. ep=30 classifier epochs per episode outperforms ep=20 vanilla.

### Current Status

Datasets confirmed: census_income, capture24. **Wildfire (FPA-FOD, BLM vs PRIVATE land, large-fire prediction) adopted as 3rd dataset candidate 2026-05-21; re-framed 2026-05-22** — passes all 5 viability criteria (alpha-EO=0.136–0.147, sep_ratio=3.30, targeted aug delta=+0.097). Framing corrected: BLM is disadvantaged (minority_id=1), PRIVATE is reference (majority_id=0). EXP-042 (PRIVATE-as-disadvantaged) terminated due to WGL-EO disconnect; EXP-044 (BLM-as-disadvantaged) launched — soft_eo_alpha=0.1284, WGL now correctly points to BLM. MEPS dropped (see Datasets section). ACS Employment (disability) dropped after FORGE failure.

**Best confirmed results — census (3 seeds, EXP-021 grid search):**
- k=10, pca=10, ep=30, traj=2000: β-EO=0.018±0.005, EOd=0.037±0.013, F1w=0.817±0.005, AUC=0.876±0.008 — beats all baselines on EO; confirmed 2026-05-20
- Supersedes k=5 result (β-EO=0.031±0.018); k=10 grid on Huron now complete for all confirmed configs

**Capture24 status:** COMPLETE — 5/5 folds, final number confirmed 2026-06-03. Config: k=5, pca=15, ep=10, rand_0010 secondary params (ffnn_lr=2.4e-4, rl_lr=1.4e-4, δ_scale=0.110, adam/adamw). β-EO=0.069±0.030, F1w=0.948, AUC=0.945. α-EO=0.234±0.212 (GroupDRO reference). Beats all EXP-050 baselines: GroupDRO 0.122, CTGAN 0.128, FairTabDDPM 0.168, OT Repair 0.176, FLB 0.219, SMOTE 0.285.

### Datasets

**Active (paper):** census_income, capture24. **Candidate 3rd dataset:** wildfire (FPA-FOD, BLM vs PRIVATE land, large-fire prediction) — viability confirmed 2026-05-21, re-framed 2026-05-22 (BLM as disadvantaged group, minority_id=1). EXP-044 in progress. **Backup candidate:** bank_marketing (UCI Portuguese bank, working-age 30–55 vs young adults 20–29, term deposit subscription) — marginal pass (EXP-043): targeted aug delta=+0.010 (10x weaker than wildfire), hold until wildfire EXP-044 outcome known.

**DA+** = number of disadvantaged-group positive (y=1) training examples. Both datasets are configured so DA+ ≈ 43–60, the level at which reweighting methods demonstrably fail. `da_pct` is an internal implementation parameter (fraction of train set that should be disadvantaged-group positives) — do NOT use it in paper text. Always frame in terms of DA+ or positive-class rate percentages (e.g., "~11% for the disadvantaged group").

`da_pct` uses group-specific subsampling: only the disadvantaged group's positives are reduced; advantaged-group positives and all negatives are kept intact. This gives identical DA+ across all seeds. The old `bias_pct` parameter (which subsampled all positives and produced variable DA+ across seeds) is retained for backward compatibility with archived runs.

**DA+ log — confirmed values (seed=42, mean across seeds similar):**

| Dataset | da_pct | real_data_size | DA+ | Protected attr | Disadv. group |
|---------|--------|----------------|-----|----------------|---------------|
| census_income | 0.01433 | 3000 | **43** | sex | female (a=0) |
| capture24 | 0.015 | 4000 | **~60** | sex | female (a=1) |
| wildfire | 0.01433 | 3000 | **43** | owner_descr | BLM (a=1) |

**New viability criterion (6th):** Disadvantaged group natural positive-class rate in unbiased val/test must be <~15-20%. This ensures reweighting methods (FLB) cannot trivially equalize TPRs. census (11% ✓), capture24 (5.6% ✓). MEPS failed this criterion (male rx+ = 54.8%) — FLB achieves β-EO≈0.033 vs FORGE 0.575, which is structural not addressable by more training. **Wildfire (BLM framing):** BLM val positive rate is ~55%, which nominally fails this criterion. However wildfire differs from MEPS structurally: BLM is biased to DA+=43, meaning FLB has only 43 BLM positives to reweight from — the same scarcity that prevents CTGAN/FairTabDDPM from working. Whether FLB can trivially equalize under this constraint must be verified by running baselines under the BLM-as-disadvantaged framing (baselines from EXP-041 used PRIVATE framing and cannot be reused).

**Dataset selection criteria** (all must pass before committing to RL experiments):
1. val_disadv_pos ≥ 30 — stable reward signal
2. test_disadv_pos ≥ 200 — reliable fairness evaluation
3. alpha_EO: at least one seed clearly non-zero in viability — meaningful pre-intervention gap exists. The viability FFNN systematically underestimates FORGE's actual alpha-EO (by roughly 3–5×) because it trains on heavily imbalanced biased data without class weighting and collapses toward all-negative predictions. Census passes (FORGE alpha-EO ~0.34) despite viability showing only 0.047. Treat the viability alpha-EO as a relative signal, not an absolute threshold: near-zero on all seeds = reject; clearly non-zero on best seed = pass.
4. Feature-space distinctiveness — disadvantaged-group positives must be spatially distinct from advantaged-group positives in PCA space (sep_ratio > 1, computed with drop_protected=True); otherwise synthetic samples cannot carry group-specific signal
5. Targeted augmentation probe (Step 4 of viability script) — inject synthetic DA+ near centroid, train beta, check delta EO > 0. Hard FAIL if delta ≤ 0: even perfect RL targeting cannot help. PASS does not guarantee RL success (ACS Employment sex passed at delta=+0.10 yet failed in practice).

**WGL-EO alignment — no reliable pre-flight test exists.** The viability script now computes: (a) cosine similarity of discriminant vectors (WARN if > 0.95 — both census at 0.97 and ACS Employment sex at 0.98 trigger this; not a hard threshold); (b) WGL group dominance ratio — informational only, commonly inverted even for working datasets. The only reliable WGL-EO check is monitoring β-EO vs α-EO at ep ~500 in the actual RL run. If β-EO is not tracking below α-EO on at least one seed by ep ~1000, the run should be terminated.

**Viability script note:** Always run with drop_protected=True (the default as of 2026-05-21). Earlier runs with drop_protected=False produced misleading results — ACS Income + race appeared viable (alpha-EO=0.131) but failed completely once drop_protected=True was applied (alpha-EO=0.007, sep_ratio=0.92).

**Dropped datasets:**
- **COMPAS** — val_pos=14 (below threshold), positive-class overlap in PCA space, RL results contradict motivation claim
- **PAMAP2** — only 1 female subject, unstable group identification across seeds
- **credit_card** — DA+ too high (~136), alpha-EO near-zero, not in scarcity regime
- **PTB-XL** — no framing reaches test_pos≥200
- **MEPS (ethnicity framing)** — val_pos=37 marginal; ethnicity framing test_pos=111 fails evaluation threshold
- **MEPS (sex framing)** — passes viability but fails 6th criterion: natural male positive rate = 54.8% in unbiased val/test; FLB trivially equalizes (β-EO≈0.033 vs FORGE 0.575). Structural failure, not addressable by more training.
- **Covertype** — targeted aug delta = -0.030 (hard fail on criterion 5); EO increases when injecting near DA+ centroid. Do not retry.
- **BRFSS (sex/cvdinfr4, race/cvdinfr4, sex/depression)** — near-zero alpha-EO and sep_ratio < 1.0 across all framings. Features carry no group-specific signal.
- **ACS Employment (disability)** — FORGE beta-EO 0.62–0.71; WGL-EO disconnect; narrow repair gap (1.7x); AA+/DA+ ratio 30x
- **ACS Employment (sex/age/nativity/race/veteran)** — sex framing: WGL-EO disconnect (beta-EO never beats alpha across 2918 eps, same pattern as disability); other framings: alpha-EO near zero or narratively inappropriate. ACS Employment ruled out entirely.
- **ACS Income (race)** — passes with drop_protected=False but collapses with drop_protected=True (alpha-EO=0.007, sep_ratio=0.92); proxy features carry no race-specific signal after dropping RAC1P
- **Diabetes130 (age)** — test_disadv_pos=60 (fails ≥200); alpha-EO near zero
- **Diabetes130 (race)** — identical readmission rates for Black/White (8.5% vs 9.0%), no EO signal

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

### Reward Structure (current best: global-only, k=5 for census)

- **Global term**: `sigmoid(k × (wgl_alpha − wgl_beta))` where wgl = worst-group BCE loss on validation. k=10 (vanilla default); k=5 is best for census (EXP-021), k=3 provisional best for capture24 (EXP-025). Range (0,1); above 0.5 means beta is better than alpha. k=0 gives normalized reward: (wgl_alpha − wgl_beta) / wgl_alpha.
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

Located in `analysis/`:

| File | Purpose |
|------|---------|
| `analysis/check_run.py` | Standard post-run analysis — summary table, learning curves, generalizability curves |
| `analysis/analyze_grid.py` | Aggregate and rank grid-search results across completed runs |
| `analysis/analyze_k0_grid.py` | Grid analysis specific to k=0 (no-sigmoid) runs |
| `analysis/analyze_kfold.py` | Aggregate FORGE and baseline results across k-fold runs |
| `analysis/analyze_reward_signal.py` | Reward signal diagnostics across training |
| `analysis/eval_checkpoint.py` | Evaluate best saved checkpoint on full val/test sets |

Other tools (project root):

| File | Purpose |
|------|---------|
| `dataset_viability.py` | Dataset structural viability checker — DA+ scan, alpha-EO baseline, feature separability |
| `make_spec.py` | Generate experiment spec JSON + SLURM batch files from a base spec with `--patch` / `--sweep` |
| `make_search_specs.py` | Generate random/grid hyperparameter search specs and SLURM batch files |
| `run_baseline.py` | Run GroupDRO, FLB, CTGAN, OT Repair, and other baselines |

Paper figure scripts (in `paper_figures/`):

| File | Purpose |
|------|---------|
| `paper_figures/fig2_training_curves.py` | Figure 2 — episode return training curves (Census + Capture-24) |
| `paper_figures/fig3_centroid_drift.py` | Figure 3 — centroid drift toward disadvantaged-positive cluster |
| `paper_figures/fig4_comparative_results.py` | Figure 4 — comparative fairness-utility profile across all methods |
| `paper_figures/fig5_radar.py` | Figure 5 — multi-metric radar trade-off profiles |
| `paper_figures/fig_grid_sensitivity.py` | Exp 1 pending — sensitivity heatmap (k × PCA, k × traj) |
| `paper_figures/fig_grid_scatter.py` | Exp 1 pending — grid scatter plot |
| `paper_figures/plot_centroid_drift.py` | Per-run centroid drift (input to fig3_centroid_drift.py) |

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
python analysis/check_run.py training_runs/<run_dir> [--interval 150] [--device cpu] [--no-gen-curve]
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

---

## 4. Server Infrastructure

Three local GPU servers. SSH from the Windows client uses the keys listed below. SSH between servers (e.g. Huron → Aulavik) requires Huron's `~/.ssh/id_rsa.pub` to be in Aulavik/Lambda's `~/.ssh/authorized_keys`.

| Host | IP | Port | User | Windows key |
|------|----|------|------|-------------|
| Huron | 129.100.226.162 | 2021 | epigou | `C:\Users\epigo\Documents\Summer2024\id_rsa_huron` |
| Lambda | 129.100.226.208 | 2023 | epigou | `C:\Users\epigo\Documents\Summer2024\id_rsa_lambda` |
| Aulavik | 129.100.226.194 | 2023 | epigou | `C:\Users\epigo\Documents\Summer2024\id_rsa_aulavik` |
| Oneida | 129.100.226.232 | 2023 | epigou | `~/.ssh/id_rsa_oneida` |

**Storage layout (Huron):** `/storage_1/epigou_storage/FORGE/` contains `training_runs/`, `training_runs_k10/`, `aulavik_runs/`, `lambda_runs/`.

**Python environment:** `~/envs/rl/` — activate with `source ~/envs/rl/bin/activate` before running any project scripts.
