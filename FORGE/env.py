import numpy as np
import torch


class Environment:
    def __init__(
        self,
        target: int,
        max_actions: int,
        total_episodes: int,
        device,
        seed: int = 42,
        pca_components: int = 2,
        pca_means: torch.Tensor | None = None,
        real_minority_samples: torch.Tensor | None = None,
        use_delta_actions: bool = True,
        delta_scale: float = 0.10,
        delta_clip: float | None = 0.20,
        pca_clip: float | None = None,
        use_radius_clip: bool = True,
        radius_clip: float | None = 8.5,
        eps: float = 1e-8,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)

        if real_minority_samples is not None:
            self.real_minority_samples = real_minority_samples.to(self.device)
        else:
            self.real_minority_samples = None

        self.target = int(target)
        self.max_actions = int(max_actions)
        self.total_episodes = int(total_episodes)
        self.A = int(pca_components)

        self.use_delta_actions = bool(use_delta_actions)
        self.delta_scale = float(delta_scale)
        self.delta_clip = None if delta_clip is None else float(delta_clip)
        self.pca_clip = None if pca_clip is None else float(pca_clip)

        self.use_radius_clip = bool(use_radius_clip) and (radius_clip is not None)
        self.radius_clip = None if radius_clip is None else float(radius_clip)
        self.eps = float(eps)

        if pca_means is None:
            self.pca_means = torch.zeros(self.A, dtype=torch.float32, device=self.device)
        else:
            self.pca_means = pca_means.to(self.device).float().view(-1)
            if self.pca_means.numel() != self.A:
                raise ValueError(f"pca_means must have {self.A} elements, got {self.pca_means.numel()}.")

        self.step_idx = 0
        self.current_pca = torch.zeros(self.A, dtype=torch.float32, device=self.device)
        self.editable_mask = torch.ones(self.A, dtype=torch.float32, device=self.device)
        self.current_stage = 0  # always 0; kept for logging compatibility

        self.heterogenity_rewards = []
        self.generated_buffer = []

    def _build_state(self, frac_done: float) -> torch.Tensor:
        frac_done_t = torch.tensor([frac_done], dtype=torch.float32, device=self.device)
        return torch.cat([frac_done_t, self.current_pca, self.editable_mask], dim=0)

    def _apply_radius_clip(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.use_radius_clip) or (self.radius_clip is None):
            return x
        r = torch.linalg.norm(x, ord=2)
        if r <= self.radius_clip:
            return x
        return x * (self.radius_clip / (r + self.eps))

    def reset(self) -> torch.Tensor:
        self.step_idx = 0
        self.current_pca = torch.zeros(self.A, dtype=torch.float32, device=self.device)
        self.editable_mask = torch.ones(self.A, dtype=torch.float32, device=self.device)
        self.heterogenity_rewards = []
        self.generated_buffer = []
        return self._build_state(0.0)

    def step(self, action: torch.Tensor, curr_length: int):
        self.step_idx = int(curr_length)

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

        frac_done = max(0.0, min(1.0, float(curr_length) / float(self.max_actions)))
        state_vec = self._build_state(frac_done)
        done = curr_length >= self.max_actions

        info = {
            "sampled_target": torch.tensor(self.target, device=self.device, dtype=torch.long),
            "current_pca":    self.current_pca.detach().clone(),
        }

        return state_vec, done, info

    def get_episode_data(self):
        return list(self.heterogenity_rewards), np.array(self.generated_buffer)
