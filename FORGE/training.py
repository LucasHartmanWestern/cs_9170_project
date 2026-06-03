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
        self.sigmoid_k = float(_k if _k is not None else 10.0)

        self.beta_reset_interval = int(spec.get("beta_reset_interval", 1))

    def train_classifier(self, model, x_train, y_train):
        loader = DataLoader(
            TensorDataset(x_train, y_train),
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
        x_val,
        y_val,
        x_gen,
        progress: float,
    ):
        y_val = y_val.long()

        # --- Beta performance on validation set ---
        with torch.no_grad():
            p_beta = rh.p1_from_agent(beta_model, x_val)

            f1_minority = rh.f1_from_probs(y_val, p_beta, 0.5)
            f1_majority = rh.f1_from_probs(1 - y_val, 1 - p_beta, 0.5)
            f1_macro    = 0.5 * (f1_minority + f1_majority)
            acc         = rh.acc_from_probs(y_val, p_beta, 0.5)
            auc         = rh.roc_auc_from_probs(y_val, p_beta)

        # --- Worst-group loss and fairness metrics ---
        a_val = self.dataset.a_val
        with torch.no_grad():
            beta_losses  = rh.bce_per_sample_from_probs(y_val, p_beta)
            cell_ids     = torch.as_tensor(a_val, device=beta_losses.device).long() * 2 + y_val
            wgl_beta_t, per_cell = rh.worst_group_loss(beta_losses, cell_ids, group_values=(0, 1, 2, 3))

            bce_cells      = [per_cell.get(i, float("nan")) for i in range(4)]
            valid_cells    = [(i, v) for i, v in enumerate(bce_cells) if v == v]
            worst_cell_id  = max(valid_cells, key=lambda x: x[1])[0] if valid_cells else -1

            bce_g0  = per_cell.get(0, float("nan"))
            bce_g1  = per_cell.get(1, float("nan"))
            bce_mean = float(beta_losses.mean().item())

            if not hasattr(self, "_n_g0"):
                a_t = torch.as_tensor(a_val, device=beta_losses.device).long()
                self._n_g0 = int((a_t == 0).sum().item())
                self._n_g1 = int((a_t == 1).sum().item())

            if bce_g0 == bce_g0 and bce_g1 == bce_g1:
                worst_group = 1 if bce_g1 >= bce_g0 else 0
                bce_gap     = float(abs(bce_g1 - bce_g0))
            else:
                worst_group = None
                bce_gap     = float("nan")

            wgl_beta = float(wgl_beta_t.item()) if wgl_beta_t == wgl_beta_t else float("nan")

            fcls        = rh.fairness_classification_metrics(a_val, y_val, p_beta, threshold=0.5)
            dp_diff     = fcls["dp_diff"]
            eo_gap      = fcls["eo_tpr_diff"]
            eod_max     = fcls["eod_max_diff"]
            eod_avg     = fcls["eod_avg_diff"]

            soft_eo_b   = rh.soft_eo_gap(a_val, y_val, p_beta)
            soft_eo     = float(soft_eo_b.item()) if soft_eo_b == soft_eo_b else float("nan")

        # --- Reward: sigmoid(k * (WGL_alpha - WGL_beta)) ---
        wgl_alpha = getattr(self, "wgl_alpha_baseline", float("nan"))
        if wgl_alpha == wgl_alpha and wgl_beta == wgl_beta:
            if self.sigmoid_k == 0:
                wgl_reward = float((wgl_alpha - wgl_beta) / (wgl_alpha + 1e-8))
            else:
                wgl_reward = float(torch.sigmoid(
                    torch.tensor(self.sigmoid_k * (wgl_alpha - wgl_beta))
                ).item())
        else:
            wgl_reward = 0.0

        lambda_start, lambda_end = self.lambda_schedule
        lambda_t = lambda_start + (lambda_end - lambda_start) * progress
        reward = torch.full(
            (len(x_gen),),
            lambda_t * wgl_reward / float(len(x_gen)),
            dtype=torch.float32,
            device=x_gen.device,
        )

        # --- Generation diagnostics ---
        with torch.no_grad():
            p_alpha_gen = rh.p1_from_agent(alpha_model, x_gen)
        try:
            frac_uncertain = float(((p_alpha_gen >= 0.4) & (p_alpha_gen <= 0.6)).float().mean().item())
            mean_margin    = float(torch.abs(p_alpha_gen - 0.5).mean().item())
            gen_radius     = float(torch.linalg.norm(x_gen, dim=1).mean().item())
        except Exception:
            frac_uncertain = mean_margin = gen_radius = float("nan")

        metrics = {
            "reward": {
                "wgl_reward": float(wgl_reward),
            },
            "performance": {
                "f1_macro":   float(f1_macro),
                "f1_minority": float(f1_minority),
                "acc":        float(acc),
                "auc":        float(auc),
            },
            "fairness": {
                "wgl_beta":        float(wgl_beta),
                "wgl_alpha":       float(wgl_alpha),
                "bce_g0":          float(bce_g0),
                "bce_g1":          float(bce_g1),
                "worst_group":     worst_group,
                "bce_gap":         float(bce_gap),
                "bce_mean":        float(bce_mean),
                "n_g0":            int(self._n_g0),
                "n_g1":            int(self._n_g1),
                "soft_eo_alpha":   float(getattr(self, "soft_eo_alpha", float("nan"))),
                "dp_diff":         float(dp_diff),
                "eo_gap":          float(eo_gap),
                "soft_eo":         float(soft_eo),
                "eod_max":         float(eod_max),
                "eod_avg":         float(eod_avg),
                "bce_cell_00":     float(bce_cells[0]),
                "bce_cell_01":     float(bce_cells[1]),
                "bce_cell_10":     float(bce_cells[2]),
                "bce_cell_11":     float(bce_cells[3]),
                "worst_cell_id":   int(worst_cell_id),
            },
            "generation": {
                "frac_uncertain": float(frac_uncertain),
                "mean_margin":    float(mean_margin),
                "gen_radius":     float(gen_radius),
            },
        }

        return reward, metrics

    def _run_episode_loop(
        self,
        *,
        env,
        agent,
        x_train,
        y_train,
        x_val,
        y_val,
    ) -> tuple:
        best_reward = -float("inf")
        best_x_syn  = None
        best_y_syn  = None

        print(f"\n{'='*60}")
        print(f"[FORGE {self.process_label}] Training | episodes={self.episodes}")
        print(f"{'='*60}")

        for episode in range(self.episodes):
            A = self.pca_components
            D = 1 + 2 * A

            states      = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
            actions     = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
            next_states = torch.zeros((self.traj_length, D), dtype=torch.float32, device=self.device)
            dones       = torch.zeros(self.traj_length, dtype=torch.bool, device=self.device)
            x_traj      = torch.zeros((self.traj_length, A), dtype=torch.float32, device=self.device)
            y_traj      = torch.zeros(self.traj_length, dtype=torch.long, device=self.device)

            state = env.reset()

            if episode % self.beta_reset_interval == 0:
                self.beta_model.reset()

            for t in range(self.traj_length):
                action = agent.predict(state)
                next_state, done, info = env.step(action, t + 1)

                states[t]      = state
                actions[t]     = action
                next_states[t] = next_state
                dones[t]       = done
                x_traj[t]      = info["current_pca"]
                y_traj[t]      = info["sampled_target"]

                state = next_state
                if done:
                    break

            T     = self.traj_length
            x_syn = x_traj[:T]
            y_syn = y_traj[:T]

            x_combined = torch.cat([x_train, x_syn])
            y_combined = torch.cat([y_train, y_syn])
            self.beta_model = self.train_classifier(self.beta_model, x_combined, y_combined)
            del x_combined, y_combined

            progress = (episode + 1) / self.episodes
            rewards, metrics = self.compute_reward(
                self.alpha_model, self.beta_model,
                x_val, y_val,
                x_syn, progress=progress,
            )

            states      = states[:T]
            actions     = actions[:T]
            next_states = next_states[:T]
            dones       = dones[:T]
            rewards     = rewards[:T]

            agent.learn_trajectory(states, actions, rewards, next_states, dones, episode)

            lambda_start, lambda_end = self.lambda_schedule
            lambda_t = lambda_start + (lambda_end - lambda_start) * progress

            self.tracker.log_episode(
                episode + 1,
                diagnostics=metrics,
                alignment_metrics={"lambda_t": float(lambda_t)},
                extra_metrics={
                    "episode_return": float(rewards.sum().item()),
                    "avg_reward":     float(rewards.mean().item()),
                },
            )
            self.tracker.maybe_save_synthetic(
                episode_num=episode + 1,
                x_syn=x_syn,
                y_syn=y_syn,
                feature_names=[f"pca_{i}" for i in range(x_syn.shape[1])],
                beta_model=self.beta_model,
                phase_label="phase1_class1",
            )

            wgl_reward = metrics.get("reward", {}).get("wgl_reward", float(rewards.sum().item()))
            if wgl_reward > best_reward:
                best_reward = wgl_reward
                best_x_syn  = x_syn.detach().clone()
                best_y_syn  = y_syn.detach().clone()

            del states, actions, next_states, dones, rewards, x_traj, y_traj, x_syn, y_syn
            if episode % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"[FORGE {self.process_label}] Done | best wgl_reward={best_reward:.4f}")
        return (best_x_syn, best_y_syn)

    def __call__(self):
        start_time = time.time()

        run_stats = {
            "EXP_GROUP":      self.exp_group,
            "SPEC_NAME":      self.spec_name,
            "dataset_name":   self.dataset_name,
            "episodes":       self.episodes,
            "traj_length":    self.traj_length,
            "real_data_size": self.real_data_size,
            "da_pct":         self.da_pct,
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
            "sigmoid_k":      self.sigmoid_k,
            "ffnn_epochs":    self.ffnn_config.get("epochs"),
            "fold_idx":       self.fold_idx,
            "n_folds":        self.n_folds if self.fold_idx is not None else None,
            "fold_rng_seed":  self.fold_rng_seed,
        }

        beta_factory = lambda: FFNNAgent(**self.ffnn_config)

        with EpisodeTracker(
            run_stats,
            dataset=self.dataset,
            process_label=self.process_label,
            save_dir=getattr(self, "save_dir", "training_runs"),
            compare_metric="reward.wgl_reward",
            beta_factory=beta_factory,
            seed=self.seed,
            ckpt_every=150,
        ) as tracker:
            self.tracker = tracker

            x_train, x_val, x_test, y_train, y_val, y_test = self.dataset.get_data_splits(
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

            # Resolve actual feature dimension (PCA may reduce it)
            feature_dim = x_train.shape[1]
            self.pca_components = feature_dim
            self.state_dim = 1 + 2 * feature_dim

            self.ffnn_config["input_size"] = feature_dim
            self.reinforce_config["state_size"] = self.state_dim
            self.reinforce_config["action_size"] = feature_dim
            self.agent       = ReinforceAgent(**self.reinforce_config)
            self.alpha_model = FFNNAgent(**self.ffnn_config)
            self.beta_model  = FFNNAgent(**self.ffnn_config)

            total = len(x_train) + self.traj_length
            print(f"Beta trains with: {100*len(x_train)/total:.1f}% real, "
                  f"{100*self.traj_length/total:.1f}% synthetic")

            # Train alpha on real data only
            self.alpha_model = self.train_classifier(self.alpha_model, x_train, y_train)

            # Compute alpha WGL baseline (used as reference in reward every episode)
            disadv, adv, _, _ = rh.disadvantaged_group_from_alpha(
                self.alpha_model, x_val, y_val, self.dataset.a_val, group_values=(0, 1),
            )
            self.disadv_group = disadv
            self.adv_group    = adv

            with torch.no_grad():
                y_val_long    = y_val.long()
                p_alpha       = rh.p1_from_agent(self.alpha_model, x_val)
                alpha_losses  = rh.bce_per_sample_from_probs(y_val_long, p_alpha)
                cell_ids      = torch.as_tensor(self.dataset.a_val, device=alpha_losses.device).long() * 2 + y_val_long
                wgl_alpha, _  = rh.worst_group_loss(alpha_losses, cell_ids, group_values=(0, 1, 2, 3))
            self.wgl_alpha_baseline = float(wgl_alpha.item()) if wgl_alpha == wgl_alpha else float("nan")

            with torch.no_grad():
                soft_eo_a = rh.soft_eo_gap(self.dataset.a_val, y_val_long, p_alpha)
            self.soft_eo_alpha = float(soft_eo_a.item()) if soft_eo_a == soft_eo_a else float("nan")

            print(f"[alpha {self.process_label}] disadv_group={disadv} "
                  f"wgl={self.wgl_alpha_baseline:.4f} soft_eo={self.soft_eo_alpha:.4f}")

            # Seed samples: disadvantaged minority training points for env reset
            a_train    = self.dataset.a_train
            seed_mask  = (y_train == 1) & (a_train == int(self.disadv_group))
            seed_samples = x_train[seed_mask]

            self.tracker.save_alpha_state_dict(self.alpha_model, self.ffnn_config, self.pca_components)

            try:
                pca_means = getattr(self.dataset, "pca_means_tensor", None)
            except Exception:
                pca_means = None

            env = Environment(
                target=1,
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

            self._run_episode_loop(
                env=env,
                agent=self.agent,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )

            self.tracker.log_final_test(
                alpha_model=self.alpha_model,
                x_test=x_test,
                y_test=y_test,
                f1_thresh=0.5,
                prefer_best_beta=True,
                beta_model=self.beta_model,
                a_test=getattr(self.dataset, "a_test", None),
            )

        print(f"Total time {time.time() - start_time:.2f}s")
        print(f"[Tracker {self.process_label}] Finished. Run folder: {self.tracker.summary_path()}")
        return True
