import numpy as np
import torch


class Environment:
    def __init__(
        self,
        curriculum: bool,
        target: int,
        max_actions: int,
        total_episodes: int,
        device,
        seed: int = 42,
        pca_components: int = 2,
        pca_means: torch.Tensor | None = None,
        curriculum_stages: list[tuple[float, int]] | None = None,
        real_minority_samples: torch.Tensor | None = None,

        # -----------------------------
        # Delta-action controls
        # -----------------------------
        use_delta_actions: bool = True,
        delta_scale: float = 0.10,
        delta_clip: float | None = 0.20,
        pca_clip: float | None = None,

        # Optional: keep points within a reasonable radius band
        use_radius_clip: bool = True,
        radius_clip: float | None = 8.5,
        eps: float = 1e-8,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Per-env RNG (do NOT seed global numpy)
        self.rng = np.random.default_rng(seed)

        if real_minority_samples is not None:
            self.real_minority_samples = real_minority_samples.to(self.device)
        else:
            self.real_minority_samples = None

        self.target = int(target)
        self.max_actions = int(max_actions)
        self.total_episodes = int(total_episodes)
        self.curriculum = bool(curriculum)

        # PCA dimensionality
        self.A = int(pca_components)

        # Delta-action config
        self.use_delta_actions = bool(use_delta_actions)
        self.delta_scale = float(delta_scale)
        self.delta_clip = None if delta_clip is None else float(delta_clip)
        self.pca_clip = None if pca_clip is None else float(pca_clip)

        # Radius clip config
        self.use_radius_clip = bool(use_radius_clip) and (radius_clip is not None)
        self.radius_clip = None if radius_clip is None else float(radius_clip)
        self.eps = float(eps)

        # base PCA means
        if pca_means is None:
            self.pca_means = torch.zeros(self.A, dtype=torch.float32, device=self.device)
        else:
            self.pca_means = pca_means.to(self.device).float().view(-1)
            if self.pca_means.numel() != self.A:
                raise ValueError(
                    f"pca_means must have {self.A} elements, got {self.pca_means.numel()}."
                )

        # curriculum schedule
        if curriculum_stages is None:
            default_stages = [
                (0.0, min(2, self.A)),
                (0.33, min(4, self.A)),
                (0.66, self.A),
            ]
            self.curriculum_stages = sorted(default_stages, key=lambda x: x[0])
        else:
            cs = sorted(curriculum_stages, key=lambda x: x[0])
            # clamp k to [1, A] and cast threshold to float
            self.curriculum_stages = [(float(th), max(1, min(int(k), self.A))) for th, k in cs]

        # episode-local state
        self.step_idx = 0
        self.current_pca = torch.zeros(self.A, dtype=torch.float32, device=self.device)
        self.editable_mask = torch.ones(self.A, dtype=torch.float32, device=self.device)
        self.num_active_dims = self.A

        # curriculum stage index (stable + explicit)
        self.current_stage = 0

        # debug buffers (unchanged)
        self.heterogenity_rewards = []
        self.generated_buffer = []

    # --------------------------------------------------
    # Curriculum helpers
    # --------------------------------------------------
    def _progress(self, episode_idx: int) -> float:
        if self.total_episodes <= 1:
            return 0.0
        return float(episode_idx) / float(self.total_episodes - 1)

    def _stage_and_active_from_progress(self, episode_idx: int) -> tuple[int, int]:
        """
        Returns (stage_idx, active_dims) based on curriculum_stages thresholds.
        stage_idx is the index in self.curriculum_stages of the selected stage.
        """
        if not self.curriculum:
            return 0, self.A

        prog = self._progress(episode_idx)

        stage_idx = 0
        active_dims = self.curriculum_stages[0][1]

        for i, (threshold, k) in enumerate(self.curriculum_stages):
            if prog >= threshold:
                stage_idx = i
                active_dims = k
            else:
                break

        return stage_idx, max(1, min(int(active_dims), self.A))

    def _build_state(self, frac_done: float) -> torch.Tensor:
        if not self.curriculum:
            return torch.tensor([1.0, frac_done], dtype=torch.float32, device=self.device)

        frac_done_t = torch.tensor([frac_done], dtype=torch.float32, device=self.device)
        return torch.cat([frac_done_t, self.current_pca, self.editable_mask], dim=0)

    def _apply_radius_clip(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.use_radius_clip) or (self.radius_clip is None):
            return x
        r = torch.linalg.norm(x, ord=2)
        if r <= self.radius_clip:
            return x
        scale = self.radius_clip / (r + self.eps)
        return x * scale

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def reset(self, episode_idx: int = 0) -> torch.Tensor:
        self.step_idx = 0

        if self.curriculum:
            stage_idx, new_num_active = self._stage_and_active_from_progress(episode_idx)
            self.current_stage = stage_idx
            self.num_active_dims = new_num_active

            self.editable_mask = torch.zeros(self.A, dtype=torch.float32, device=self.device)
            self.editable_mask[: self.num_active_dims] = 1.0

            # init from real minority if available
            if self.real_minority_samples is not None and self.real_minority_samples.size(0) > 0:
                idx = int(self.rng.integers(0, self.real_minority_samples.size(0)))
                self.current_pca = self.real_minority_samples[idx].detach().clone().to(self.device)
            else:
                self.current_pca = self.pca_means.clone().detach()

            # optional clamps on initial point too
            if self.pca_clip is not None:
                self.current_pca = torch.clamp(self.current_pca, -self.pca_clip, +self.pca_clip)
            self.current_pca = self._apply_radius_clip(self.current_pca)

        else:
            self.current_stage = 0
            self.current_pca = torch.zeros(self.A, dtype=torch.float32, device=self.device)
            self.editable_mask = torch.ones(self.A, dtype=torch.float32, device=self.device)
            self.num_active_dims = self.A

        frac_done = 0.0
        state_vec = self._build_state(frac_done)

        self.heterogenity_rewards = []
        self.generated_buffer = []

        return state_vec

    def step(self, action: torch.Tensor, curr_length: int):
        self.step_idx = int(curr_length)

        if self.curriculum:
            # Ensure float tensor on device
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
            else:
                action = action.to(self.device).float()

            if action.shape[-1] != self.A:
                raise ValueError(f"Expected action dim={self.A}, got {action.shape[-1]}.")

            editable = self.editable_mask.bool()

            if self.use_delta_actions:
                delta = action

                if self.delta_scale != 1.0:
                    delta = delta * self.delta_scale

                if self.delta_clip is not None:
                    delta = torch.clamp(delta, -self.delta_clip, +self.delta_clip)

                new_pca = self.current_pca.clone()
                new_pca[editable] = new_pca[editable] + delta[editable]

            else:
                new_pca = self.current_pca.clone()
                new_pca[editable] = action[editable]

            if self.pca_clip is not None:
                new_pca = torch.clamp(new_pca, -self.pca_clip, +self.pca_clip)

            new_pca = self._apply_radius_clip(new_pca)

            self.current_pca = new_pca
            self.generated_buffer.append(self.current_pca.detach().cpu().numpy())

        frac_done = float(curr_length) / float(self.max_actions)
        frac_done = max(0.0, min(1.0, frac_done))

        state_vec = self._build_state(frac_done)
        done = curr_length >= self.max_actions

        info = {
            "sampled_target": torch.tensor(self.target, device=self.device, dtype=torch.long)
        }
        if self.curriculum:
            info["current_pca"] = self.current_pca.detach().clone()

        return state_vec, done, info

    def get_episode_data(self):
        return list(self.heterogenity_rewards), np.array(self.generated_buffer)
