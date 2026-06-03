import gc
import sys
import time

import numpy as np
import torch
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

_project_root = Path(__file__).parent.parent
for _p in [str(_project_root), str(_project_root / 'utilities'), str(_project_root / 'FORGE')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env import Environment
from dataset import Dataset
from agents.reinforce_agent import ReinforceAgent
from agents.ffnn_agent import FFNNAgent
from episode_tracker import EpisodeTracker
import reward_helpers as rh


class Training:
    def __init__(
        self,
        exp_group,
        spec_name,
        spec,
        output_dir,
        seed=42,
        process_label="Training process -0",
        device='cpu'
    ):
        self.exp_group = exp_group
        self.spec_name = spec_name
        self.seed = seed
        self.process_label = process_label
        self.device = torch.device(device)
        self.save_dir = output_dir

        self._get_specs(spec)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.set_float32_matmul_precision("highest")
        print(f"[Training {self.process_label}] ---- running seed={self.seed} ----")

        self.state_dim = 1 + 2 * self.pca_components

        self.ffnn_overrides = dict(self.ffnn or {})
        if spec.get("epochs") is not None:
            self.ffnn_overrides["epochs"] = int(spec["epochs"])
        self.reinforce_overrides = self.reinforce or {}

        self.dataset = Dataset(
            self.dataset_name,
            minority_id=self.minority_id,
            majority_id=self.majority_id,
            pca_components=self.pca_components,
            seed=self.seed,
            device=self.device,
            use_pca=self.use_pca,
        )

        DEFAULT_FFNN = {
            "input_size": self.pca_components,
            "hidden_sizes": [32, 16],
            "output_size": 2,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 10,
            "optimizer": "adam",
            "type": "classification",
            "classes": [0, 1],
            "device": self.device,
            "seed": self.seed,
        }
        self.ffnn_config = {**DEFAULT_FFNN, **self.ffnn_overrides}
        self.ffnn_config["input_size"] = self.pca_components
        self.ffnn_config["output_size"] = 2
        self.ffnn_config["classes"] = [0, 1]
        self.ffnn_config["device"] = self.device
        self.ffnn_config["seed"] = self.seed

        DEFAULT_REINFORCE = {
            "state_size": self.state_dim,
            "action_size": self.pca_components,
            "hidden_sizes": [64, 64],
            "total_episodes": self.episodes,
            "lr": 3e-4,
            "gamma": 0.99,
            "entropy_start": 1e-2,
            "entropy_end": 0.0,
            "optimizer": "adam",
            "seed": self.seed,
            "device": self.device,
        }
        reinforce_config = {**DEFAULT_REINFORCE, **self.reinforce_overrides}
        reinforce_config["state_size"] = self.state_dim
        reinforce_config["action_size"] = self.pca_components
        reinforce_config["total_episodes"] = self.episodes
        reinforce_config["seed"] = self.seed
        reinforce_config["device"] = self.device
        self.reinforce_config = reinforce_config

        self.dl_generator = torch.Generator(device="cpu").manual_seed(self.seed)

        self.agent = ReinforceAgent(**reinforce_config)
        self.alpha_model = FFNNAgent(**self.ffnn_config)
        self.beta_model = FFNNAgent(**self.ffnn_config)

    def _get_specs(self, spec):
        rs = spec.get("reward_shaping", {})

        self.dataset_name = spec["dataset_name"]
        self.minority_id = spec.get("minority_id")
        self.majority_id = spec.get("majority_id")
        self.da_pct = spec.get("da_pct")

        self.pca_components = spec["pca_components"]
        if "total_data_size" in spec and "ratio_trajectory" in spec:
            total_data_size = spec["total_data_size"]
            self.traj_length = int(spec["ratio_trajectory"] * total_data_size)
            self.real_data_size = int(total_data_size - self.traj_length)
        else:
            self.traj_length = spec["traj_length"]
            self.real_data_size = spec["real_data_size"]
        self.episodes = spec["total_episodes"]

        self.lambda_schedule = tuple(spec["lambda_schedule"])

        self.use_delta_actions = spec.get("use_delta_actions", True)
        self.delta_scale = spec.get("delta_scale", 0.10)
        self.delta_clip = spec.get("delta_clip", 0.20)
        self.pca_clip = spec.get("pca_clip", None)
        self.radius_clip = spec.get("radius_clip", None)

        self.use_pca = spec.get("use_pca", True)

        self.ffnn = spec["ffnn"]
        self.reinforce = spec["reinforce"]

        self.dp_protected_col = spec.get("dp_protected_col", None)

        self.fold_idx = spec.get("fold_idx", None)
        self.n_folds = spec.get("n_folds", 5)
        self.fold_rng_seed = spec.get("fold_rng_seed", None)

        _k = spec.get("global_sigmoid_k") or rs.get("global_sigmoid_k")
        self.global_sigmoid_k = float(_k if _k is not None else 10.0)
        _ugmf = rs.get("utility_guard_min_factor")
        self.utility_guard_min_factor = float(_ugmf if _ugmf is not None else 1.0)

        self.beta_reset_interval = int(spec.get("beta_reset_interval", 1))

    def train_predictor_model(self, model, x_train, y_train):
        train_dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(
            train_dataset,
            batch_size=int(self.ffnn_config["batch_size"]),
            shuffle=True,
            generator=self.dl_generator,
        )
        model.train(loader)
        return model

    def compute_reward(
        self,
        alpha_model,
        beta_model,
        x_theta_val,
        y_theta_val,
        x_phi,
        progress: float,
    ):
        y_val_bin = y_theta_val.long()

        with torch.no_grad():
            p1_beta_val = rh.p1_from_agent(beta_model, x_theta_val)

            f1_minority_beta = rh.f1_from_probs(y_val_bin, p1_beta_val, 0.5)
            f1_majority_beta = rh.f1_from_probs(1 - y_val_bin, 1 - p1_beta_val, 0.5)
            f1_macro_beta    = 0.5 * (f1_minority_beta + f1_majority_beta)
            acc_beta         = rh.acc_from_probs(y_val_bin, p1_beta_val, 0.5)
            auc_beta         = rh.roc_auc_from_probs(y_val_bin, p1_beta_val)

        a_theta_val = self.dataset.a_val
        with torch.no_grad():
            loss_beta_vec = rh.bce_per_sample_from_probs(y_val_bin, p1_beta_val)
            g_ids_val = torch.as_tensor(a_theta_val, device=loss_beta_vec.device).long() * 2 + y_val_bin
            worst_b_t, per_b = rh.worst_group_loss(loss_beta_vec, g_ids_val, group_values=(0, 1, 2, 3))

            _bce_cells = [per_b.get(i, float("nan")) for i in range(4)]
            _valid_cells = [(i, v) for i, v in enumerate(_bce_cells) if v == v]
            _worst_cell_id = max(_valid_cells, key=lambda x: x[1])[0] if _valid_cells else -1

            group_loss_beta_g0 = per_b.get(0, float("nan"))
            group_loss_beta_g1 = per_b.get(1, float("nan"))
            bce_mean_beta = float(loss_beta_vec.mean().item())

            if not hasattr(self, "_cached_n_g0"):
                a_t = torch.as_tensor(a_theta_val, device=loss_beta_vec.device).long()
                self._cached_n_g0 = int((a_t == 0).sum().item())
                self._cached_n_g1 = int((a_t == 1).sum().item())
            n_g0 = self._cached_n_g0
            n_g1 = self._cached_n_g1

            if (group_loss_beta_g0 == group_loss_beta_g0) and (group_loss_beta_g1 == group_loss_beta_g1):
                worst_group_beta = 1 if group_loss_beta_g1 >= group_loss_beta_g0 else 0
                group_loss_gap_beta = float(abs(group_loss_beta_g1 - group_loss_beta_g0))
            else:
                worst_group_beta = None
                group_loss_gap_beta = float("nan")

            worst_loss_beta = float(worst_b_t.item()) if worst_b_t == worst_b_t else float("nan")

            _fcls = rh.fairness_classification_metrics(
                a_theta_val, y_val_bin, p1_beta_val, threshold=0.5
            )
            ep_dp_diff      = _fcls["dp_diff"]
            ep_eo_tpr_diff  = _fcls["eo_tpr_diff"]
            ep_eod_max_diff = _fcls["eod_max_diff"]
            ep_eod_avg_diff = _fcls["eod_avg_diff"]

            _soft_eo_b = rh.soft_eo_gap(a_theta_val, y_val_bin, p1_beta_val)
            ep_soft_eo_beta = float(_soft_eo_b.item()) if (_soft_eo_b == _soft_eo_b) else float("nan")

        # Global term: sigmoid(k * (wgl_alpha - wgl_beta))
        wgl_alpha = getattr(self, "disadv_worst_loss_alpha", float("nan"))
        wgl_beta  = worst_loss_beta
        if wgl_alpha == wgl_alpha and wgl_beta == wgl_beta:
            if self.global_sigmoid_k == 0:
                global_term = float((wgl_alpha - wgl_beta) / (wgl_alpha + 1e-8))
            else:
                global_term = float(torch.sigmoid(
                    torch.tensor(self.global_sigmoid_k * (wgl_alpha - wgl_beta))
                ).item())
        else:
            global_term = 0.0

        # Utility guard: scale global_term down if beta AUC regresses vs alpha
        auc_alpha_ref = getattr(self, "auc_alpha_overall", float("nan"))
        utility_factor = 1.0
        if (self.utility_guard_min_factor < 1.0
                and auc_alpha_ref == auc_alpha_ref and auc_alpha_ref > 0
                and auc_beta == auc_beta):
            raw_factor = auc_beta / auc_alpha_ref
            utility_factor = float(max(self.utility_guard_min_factor, min(1.0, raw_factor)))
        global_term = global_term * utility_factor

        # Spread global reward uniformly over trajectory steps
        lambda_start, lambda_end = self.lambda_schedule
        lambda_t = lambda_start + (lambda_end - lambda_start) * progress
        T_traj = float(len(x_phi))
        reward = torch.full(
            (len(x_phi),),
            lambda_t * global_term / T_traj,
            dtype=torch.float32,
            device=x_phi.device,
        )

        # Diagnostics on generated samples (no extra forward pass — alpha on x_phi)
        with torch.no_grad():
            p = rh.p1_from_agent(alpha_model, x_phi)
        try:
            diag_frac_mid_conf   = float(((p >= 0.4) & (p <= 0.6)).float().mean().item())
            diag_mean_abs_margin = float(torch.abs(p - 0.5).mean().item())
            diag_gen_radius_mean = float(torch.linalg.norm(x_phi, dim=1).mean().item())
        except Exception:
            diag_frac_mid_conf = diag_mean_abs_margin = diag_gen_radius_mean = float("nan")

        diagnostics = {
            "global": {
                "global_obj":    float(global_term),
                "utility_factor": float(utility_factor),
            },
            "utility": {
                "f1_macro_beta":   float(f1_macro_beta),
                "f1_minority_beta": float(f1_minority_beta),
                "acc_beta":        float(acc_beta),
                "auc_beta":        float(auc_beta),
            },
            "fairness": {
                "worst_loss_beta":         float(worst_loss_beta),
                "group_loss_beta_g0":      float(group_loss_beta_g0),
                "group_loss_beta_g1":      float(group_loss_beta_g1),
                "worst_group_beta":        worst_group_beta,
                "group_loss_gap_beta":     float(group_loss_gap_beta),
                "bce_mean_beta":           float(bce_mean_beta),
                "n_g0":                    int(n_g0),
                "n_g1":                    int(n_g1),
                "worst_loss_alpha_baseline": float(getattr(self, "disadv_worst_loss_alpha", float("nan"))),
                "eo_alpha_baseline":       float(getattr(self, "eo_alpha_baseline", float("nan"))),
                "dp_diff":                 float(ep_dp_diff),
                "eo_tpr_diff":             float(ep_eo_tpr_diff),
                "soft_eo_beta":            float(ep_soft_eo_beta),
                "eod_max_diff":            float(ep_eod_max_diff),
                "eod_avg_diff":            float(ep_eod_avg_diff),
                "bce_cell_00":  float(_bce_cells[0]),
                "bce_cell_01":  float(_bce_cells[1]),
                "bce_cell_10":  float(_bce_cells[2]),
                "bce_cell_11":  float(_bce_cells[3]),
                "worst_cell_id": int(_worst_cell_id),
            },
            "extra": {
                "diag_frac_mid_conf":   float(diag_frac_mid_conf),
                "diag_mean_abs_margin": float(diag_mean_abs_margin),
                "diag_gen_radius_mean": float(diag_gen_radius_mean),
            },
        }

        return reward, diagnostics

    def _run_phase(
        self,
        *,
        target_class: int,
        env,
        agent,
        x_theta_train,
        y_theta_train,
        x_theta_val,
        y_theta_val,
        phase_label: str,
        episodes: int | None = None,
    ) -> tuple:
        n_episodes = episodes if episodes is not None else self.episodes

        best_phase_reward = -float("inf")
        best_x_syn = None
        best_y_syn = None

        print(f"\n{'='*60}")
        print(f"[Phase {self.process_label}] Starting {phase_label} | target_class={target_class} | episodes={n_episodes}")
        print(f"{'='*60}")

        for episode in range(n_episodes):
            A = self.pca_components
            D = 1 + 2 * A

            states      = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
            actions     = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
            next_states = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
            dones       = torch.zeros(self.traj_length, dtype=torch.bool, device=self.device)
            x_syn_tensor = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
            y_syn_tensor = torch.zeros(self.traj_length, dtype=torch.long, device=self.device)

            state = env.reset()

            if episode % self.beta_reset_interval == 0:
                self.beta_model.reset()

            for t in range(self.traj_length):
                action = agent.predict(state)
                next_state, done, info = env.step(action, (t + 1))

                states[t]      = state
                actions[t]     = action
                next_states[t] = next_state
                dones[t]       = done
                x_syn_tensor[t] = info["current_pca"]
                y_syn_tensor[t] = info["sampled_target"]

                state = next_state
                if done:
                    break

            T = self.traj_length
            x_phi_t = x_syn_tensor[:T]
            y_phi_t = y_syn_tensor[:T]

            x_hybrid = torch.cat([x_theta_train, x_phi_t])
            y_hybrid = torch.cat([y_theta_train, y_phi_t])
            self.beta_model = self.train_predictor_model(self.beta_model, x_hybrid, y_hybrid)
            del x_hybrid, y_hybrid

            progress = (episode + 1) / n_episodes

            rewards, diagnostics = self.compute_reward(
                self.alpha_model,
                self.beta_model,
                x_theta_val,
                y_theta_val,
                x_phi_t,
                progress=progress,
            )

            states      = states[:T]
            actions     = actions[:T]
            next_states = next_states[:T]
            dones       = dones[:T]
            rewards     = rewards[:T]

            agent.learn_trajectory(states, actions, rewards, next_states, dones, episode)

            lambda_start, lambda_end = self.lambda_schedule
            lambda_t = lambda_start + (lambda_end - lambda_start) * progress

            alignment_metrics = {
                "delta_global": float(-diagnostics.get("fairness", {}).get("worst_loss_beta", float("nan"))),
                "lambda_t":     float(lambda_t),
            }
            episode_return = float(rewards.sum().item())
            meta_metrics = {
                "avg_reward":    float(torch.mean(rewards).item()),
                "episode_return": episode_return,
                "phase":          phase_label,
            }

            self.tracker.log_episode(
                episode + 1,
                diagnostics=diagnostics,
                alignment_metrics=alignment_metrics,
                extra_metrics=meta_metrics,
            )
            self.tracker.maybe_save_synthetic(
                episode_num=episode + 1,
                x_syn=x_phi_t,
                y_syn=y_phi_t,
                feature_names=[f"pca_{i}" for i in range(x_phi_t.shape[1])],
                beta_model=self.beta_model,
                phase_label=phase_label,
            )

            best_select_val = diagnostics.get("global", {}).get("global_obj", episode_return)
            if best_select_val > best_phase_reward:
                best_phase_reward = best_select_val
                best_x_syn = x_phi_t.detach().clone()
                best_y_syn = y_phi_t.detach().clone()

            del states, actions, next_states, dones, rewards, x_syn_tensor, y_syn_tensor, x_phi_t, y_phi_t
            if episode % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"[Phase {self.process_label}] Finished {phase_label} | best global_obj={best_phase_reward:.4f}")
        return (best_x_syn, best_y_syn)

    def __call__(self):
        start_time = time.time()

        run_stats = {
            "EXP_GROUP":      self.exp_group,
            "SPEC_NAME":      self.spec_name,
            "dataset_name":   self.dataset_name,
            "EPISODES":       self.episodes,
            "TRAJ_LENGTH":    self.traj_length,
            "REAL_DATA_SIZE": self.real_data_size,
            "DA_PCT":         self.da_pct,
            "lambda_schedule": self.lambda_schedule,
            "pca_components": self.pca_components,
            "minority_id":    self.minority_id,
            "majority_id":    self.majority_id,
            "use_delta_actions": self.use_delta_actions,
            "delta_scale":    self.delta_scale,
            "delta_clip":     self.delta_clip,
            "pca_clip":       self.pca_clip,
            "radius_clip":    self.radius_clip,
            "seed":           self.seed,
            "global_sigmoid_k": self.global_sigmoid_k,
            "ffnn_epochs":    self.ffnn_config.get("epochs"),
            "fold_idx":       self.fold_idx,
            "n_folds":        self.n_folds if self.fold_idx is not None else None,
            "fold_rng_seed":  self.fold_rng_seed,
        }

        beta_factory = lambda: FFNNAgent(**self.ffnn_config)
        _ckpt_every = 150

        with EpisodeTracker(
            run_stats,
            dataset=self.dataset,
            process_label=self.process_label,
            save_dir=getattr(self, "save_dir", "training_runs"),
            compare_metric="global.global_obj",
            beta_factory=beta_factory,
            seed=self.seed,
            ckpt_every=_ckpt_every,
        ) as tracker:
            self.tracker = tracker

            x_theta_train, x_theta_val, x_theta_test, y_theta_train, y_theta_val, y_theta_test = (
                self.dataset.get_data_splits(
                    train_size=self.real_data_size,
                    da_pct=self.da_pct,
                    pca_components=self.pca_components,
                    drop_protected=False,
                    protected_cols=self.dataset.protected_attributes,
                    **({"fold_idx": self.fold_idx, "n_folds": self.n_folds,
                        "fold_rng_seed": self.fold_rng_seed}
                       if self.fold_idx is not None else {}),
                    **({"dp_protected_col": self.dp_protected_col}
                       if self.dp_protected_col is not None else {}),
                )
            )

            feature_dim = x_theta_train.shape[1]
            self.pca_components = feature_dim
            self.state_dim = 1 + 2 * feature_dim

            self.ffnn_config["input_size"] = feature_dim
            self.reinforce_config["state_size"] = self.state_dim
            self.reinforce_config["action_size"] = feature_dim
            self.agent = ReinforceAgent(**self.reinforce_config)
            self.alpha_model = FFNNAgent(**self.ffnn_config)
            self.beta_model = FFNNAgent(**self.ffnn_config)

            total_data = len(x_theta_train) + self.traj_length
            print(
                f"Beta trains with: {100*len(x_theta_train)/total_data:.1f}% real, "
                f"{100*self.traj_length/total_data:.1f}% synthetic"
            )

            # Train alpha once on real data
            self.alpha_model = self.train_predictor_model(self.alpha_model, x_theta_train, y_theta_train)

            # Compute alpha baselines (used by compute_reward throughout training)
            disadv, adv, _, _ = rh.disadvantaged_group_from_alpha(
                self.alpha_model, x_theta_val, y_theta_val, self.dataset.a_val, group_values=(0, 1),
            )
            self.disadv_group_value = disadv
            self.adv_group_value = adv

            with torch.no_grad():
                _y_long = y_theta_val.long()
                _p1_a   = rh.p1_from_agent(self.alpha_model, x_theta_val)
                _loss_a = rh.bce_per_sample_from_probs(_y_long, _p1_a)
                _g_ids_a = torch.as_tensor(self.dataset.a_val, device=_loss_a.device).long() * 2 + _y_long
                _worst_4g_a, _ = rh.worst_group_loss(_loss_a, _g_ids_a, group_values=(0, 1, 2, 3))
            self.disadv_worst_loss_alpha = float(_worst_4g_a.item()) if _worst_4g_a == _worst_4g_a else float("nan")
            self.auc_alpha_overall = rh.roc_auc_from_probs(_y_long, _p1_a)

            with torch.no_grad():
                _soft_eo_a = rh.soft_eo_gap(self.dataset.a_val, y_theta_val, _p1_a)
            self.eo_alpha_baseline = float(_soft_eo_a.item()) if (_soft_eo_a == _soft_eo_a) else float("nan")

            print(f"[alpha {self.process_label}] disadv_group={disadv} "
                  f"worst_4g={self.disadv_worst_loss_alpha:.4f} soft_eo={self.eo_alpha_baseline:.4f}")

            # Seed samples: disadvantaged minority train points for env initialisation
            a_train = self.dataset.a_train
            disadv_minority_mask = (y_theta_train == 1) & (a_train == int(self.disadv_group_value))
            real_minority_samples = x_theta_train[disadv_minority_mask]

            self.tracker.save_alpha_state_dict(self.alpha_model, self.ffnn_config, self.pca_components)

            try:
                pca_means = getattr(self.dataset, "pca_means_tensor", None)
            except Exception:
                pca_means = None

            def _make_env(target, seed_samples):
                return Environment(
                    target=target,
                    max_actions=self.traj_length,
                    total_episodes=self.episodes,
                    device=self.device,
                    seed=self.seed,
                    pca_components=self.pca_components,
                    pca_means=pca_means,
                    real_minority_samples=seed_samples,
                    use_delta_actions=self.use_delta_actions,
                    delta_scale=self.delta_scale,
                    delta_clip=self.delta_clip,
                    pca_clip=self.pca_clip,
                    use_radius_clip=(self.radius_clip is not None),
                    radius_clip=self.radius_clip,
                )

            phase1_env = _make_env(target=1, seed_samples=real_minority_samples)
            self._run_phase(
                target_class=1,
                env=phase1_env,
                agent=self.agent,
                x_theta_train=x_theta_train,
                y_theta_train=y_theta_train,
                x_theta_val=x_theta_val,
                y_theta_val=y_theta_val,
                phase_label="phase1_class1",
            )

            self.tracker.log_final_test(
                alpha_model=self.alpha_model,
                x_test=x_theta_test,
                y_test=y_theta_test,
                f1_thresh=0.5,
                prefer_best_beta=True,
                beta_model=self.beta_model,
                x_train=x_theta_train,
                y_train=y_theta_train,
                jitter_n=None,
                jitter_scale=0.20,
                run_alpha_raw_original=False,
                run_alpha_plus_real=False,
                alpha_plus_real_n=2000,
                run_alpha_plus_ctgan=False,
                alpha_plus_ctgan_n=self.traj_length,
                ctgan_epochs=300,
                cap_ctgan_train=None,
                data_path=None,
                bias_pct=None,
                val_frac=0.20,
                test_frac=0.20,
                train_size=self.real_data_size,
                batch_size=64,
                pca_components=None,
                seed=self.seed,
                a_test=getattr(self.dataset, "a_test", None)
            )

        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker {self.process_label}] Finished. Run folder: {self.tracker.summary_path()}")
        return True
