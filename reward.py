# reward_cramer.py
import torch
from typing import Tuple, Dict

class RewardCramer:
    """
    Per-sample Cramér/Energy-distance similarity + global minority ΔF1.

    Similarity term (higher is better) is derived from the minibatch Energy distance:
        E(X,Y) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||
    We compute a *per-synthetic-sample* contribution:
        e_j = 2 * mean_i ||x_i - y_j||  - mean_{j'!=j} ||y_j - y_{j'}||  - const_rr
    where const_rr = mean_{i!=i'} ||x_i - x_{i'}|| (same for all j).

    We then normalize e_j to [0,1] using stage-calibrated anchors:
        d_rr:  Energy distance between two real minibatches (ideal/low)
        d_rs:  Energy distance between real and a shuffled/mismatched real batch (noisy/high)

    Finally, we convert distance -> similarity via a smooth transform:
        s_j = exp(-alpha * clip((e_j - d_rr)/(d_rs - d_rr + eps), 0, 1))

    Reward per synthetic sample:
        R_j = λ(t) * ΔF1_minority_global + (1 - λ(t)) * s_j
    """

    def __init__(
        self,
        reward_mode: str = "cramer",
        alpha_sim: float = 2.0,           # sharpness for distance->similarity
        ema_momentum: float = 0.9,        # EMA for anchors (d_rr, d_rs)
        use_whitening: bool = True,       # whiten features within minibatch (per-dim std)
        eps: float = 1e-8,
    ):
        self.reward_mode = reward_mode
        self.alpha_sim = alpha_sim
        self.ema_momentum = ema_momentum
        self.use_whitening = use_whitening
        self.eps = eps

        # Stage/calibration EMAs (scalars)
        self._ema_d_rr = None
        self._ema_d_rs = None

    # ---------------- Shared helpers (copy kept for local cohesion) ----------------
    def _p1_from_agent(self, agent, x):
        agent.model.eval()
        with torch.no_grad():
            logits = agent.model(x)             # [N, 2]
            probs  = torch.softmax(logits, -1)  # [N, 2]
            return probs[..., 1]                # [N]

    def f1_from_probs(self, y_true, p1, threshold=0.5):
        y_true = y_true.to(p1.device).long()
        y_pred = (p1 >= threshold).long()
        tp = ((y_pred == 1) & (y_true == 1)).sum().float()
        fp = ((y_pred == 1) & (y_true == 0)).sum().float()
        fn = ((y_pred == 0) & (y_true == 1)).sum().float()
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return f1

    # ---------------- Cramér / Energy distance core ----------------
    def _maybe_whiten(self, X: torch.Tensor) -> torch.Tensor:
        """
        Optional per-batch whitening to prevent any dimension from dominating distances.
        X: [N, k]
        """
        if not self.use_whitening:
            return X
        mean = X.mean(dim=0, keepdim=True)
        std  = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        return (X - mean) / std

    def _per_sample_energy_components(
        self, Xr: torch.Tensor, Ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute:
          - cross_mean_per_y:  [n] where each entry = mean_i ||x_i - y_j||
          - self_mean_y_per_y: [n] where each entry = mean_{j'!=j} ||y_j - y_{j'}||
          - const_rr: scalar = mean_{i!=i'} ||x_i - x_{i'}||

        Shapes:
          Xr: [m, k], Ys: [n, k]
        """
        # Pairwise distances
        Dxy = torch.cdist(Xr, Ys, p=2)                  # [m, n]
        Dxx = torch.cdist(Xr, Xr, p=2)                  # [m, m]
        Dyy = torch.cdist(Ys, Ys, p=2)                  # [n, n]

        # Cross mean per synthetic sample
        cross_mean_per_y = Dxy.mean(dim=0)             # [n]

        # Self means excluding diagonal
        m = Xr.size(0)
        n = Ys.size(0)

        if m > 1:
            const_rr = (Dxx.sum() - Dxx.diag().sum()) / (m * (m - 1))
        else:
            const_rr = torch.tensor(0.0, device=Xr.device, dtype=Xr.dtype)

        if n > 1:
            self_sum_y = (Dyy.sum(dim=1) - Dyy.diag())    # [n]
            self_mean_y_per_y = self_sum_y / (n - 1)      # [n]
        else:
            self_mean_y_per_y = torch.zeros(n, device=Ys.device, dtype=Ys.dtype)

        return cross_mean_per_y, self_mean_y_per_y, const_rr

    def _energy_per_sample(
        self, Xr: torch.Tensor, Ys: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-synthetic-sample Energy distance contribution e_j.
        e_j = 2 * mean_i ||x_i - y_j|| - mean_{j'!=j} ||y_j - y_{j'}|| - const_rr
        Returns: [n]
        """
        Xr_w = self._maybe_whiten(Xr)
        Ys_w = self._maybe_whiten(Ys)

        cross_mean_y, self_mean_y, const_rr = self._per_sample_energy_components(Xr_w, Ys_w)
        e_per_y = 2.0 * cross_mean_y - self_mean_y - const_rr
        return e_per_y.clamp_min(0.0)  # numeric safety; E-distance is nonnegative

    # ---------------- Normalization & similarity ----------------
    def _update_ema(self, name: str, val: torch.Tensor):
        v = float(val.detach().item())
        if name == "d_rr":
            if self._ema_d_rr is None:
                self._ema_d_rr = v
            else:
                self._ema_d_rr = self.ema_momentum * self._ema_d_rr + (1 - self.ema_momentum) * v
        elif name == "d_rs":
            if self._ema_d_rs is None:
                self._ema_d_rs = v
            else:
                self._ema_d_rs = self.ema_momentum * self._ema_d_rs + (1 - self.ema_momentum) * v

    def _batch_energy_scalar(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Scalar Energy distance between two *batches* A,B using unbiased U-statistics.
        Used to calibrate anchors (d_rr, d_rs).
        """
        A_w = self._maybe_whiten(A)
        B_w = self._maybe_whiten(B)

        Dab = torch.cdist(A_w, B_w, p=2)  # [Na, Nb]
        Daa = torch.cdist(A_w, A_w, p=2)  # [Na, Na]
        Dbb = torch.cdist(B_w, B_w, p=2)  # [Nb, Nb]

        Na = A_w.size(0)
        Nb = B_w.size(0)

        cross = 2.0 * Dab.mean()
        self_a = (Daa.sum() - Daa.diag().sum()) / max(Na * (Na - 1), 1)
        self_b = (Dbb.sum() - Dbb.diag().sum()) / max(Nb * (Nb - 1), 1)
        return (cross - self_a - self_b).clamp_min(0.0)

    def _distance_to_similarity(self, d: torch.Tensor, d_rr: float, d_rs: float) -> torch.Tensor:
        """
        Map distances to [0,1] similarity via stage anchors.
        d: [n]
        """
        denom = max(d_rs - d_rr, self.eps)
        t = torch.clamp((d - d_rr) / denom, 0.0, 1.0)     # [n]
        return torch.exp(-self.alpha_sim * t)             # [n]

    # ---------------- Public API: compute_reward ----------------
    def compute_reward(
        self,
        alpha_model, beta_model, stale_beta_model,        # stale_beta_model kept for signature compat
        x_theta_val: torch.Tensor, y_theta_val: torch.Tensor,   # θ-space VAL set
        x_phi: torch.Tensor, y_phi: torch.Tensor,               # synthetic batch (labels expected 1 for minority)
        progress: float,                                        # (episode+1)/EPISODES in [0,1]

        # schedules / gates
        lambda_schedule=(0.30, 0.95),                   # (start, end) across the run

        # Unused but kept for compatibility with your runner
        epsilon_majority: float = 0.01,
        epsilon_weighted: float = 0.005,
        c_majority: float = 0.30,
        c_weighted: float = 0.30,
        class_mode: str = "binary",
        f1_thresh: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
          reward: [T] per-sample rewards for the synthetic batch x_phi
          diagnostics: dict of scalars for logging
        """

        # --- λ schedule ---
        lambda_start, lambda_end = lambda_schedule
        lambda_t = float(lambda_start + (lambda_end - lambda_start) * progress)

        # --- Prepare labels for minority vs rest evaluation ---
        if class_mode not in ("binary", "multiclass"):
            raise ValueError("class_mode must be 'binary' or 'multiclass'")

        if class_mode == "binary":
            y_val_bin = y_theta_val.long()   # 0/1
        else:
            # collapse 3-class into rope (1) vs non-rope ({0,2})
            y_val_bin = (y_theta_val == 1).long()

        with torch.no_grad():
            # ---- Global minority ΔF1 on θ_val (β vs α) ----
            p1_alpha_val = self._p1_from_agent(alpha_model, x_theta_val)
            p1_beta_val  = self._p1_from_agent(beta_model,  x_theta_val)

            f1_minority_alpha = self.f1_from_probs(y_val_bin, p1_alpha_val, f1_thresh)
            f1_minority_beta  = self.f1_from_probs(y_val_bin, p1_beta_val,  f1_thresh)
            f1_majority_alpha = self.f1_from_probs(1 - y_val_bin, 1 - p1_alpha_val, 1 - f1_thresh)
            f1_majority_beta  = self.f1_from_probs(1 - y_val_bin, 1 - p1_beta_val,  1 - f1_thresh)

            f1_macro_beta = 0.5 * (f1_minority_beta + f1_majority_beta)

            pos_frac = float(y_val_bin.float().mean().item())
            neg_frac = 1.0 - pos_frac
            f1_weighted_alpha = pos_frac * float(f1_minority_alpha) + neg_frac * float(f1_majority_alpha)
            f1_weighted_beta  = pos_frac * float(f1_minority_beta)  + neg_frac * float(f1_majority_beta)

            delta_f1_minority = float(f1_minority_beta - f1_minority_alpha)
            delta_f1_majority = float(f1_majority_beta - f1_majority_alpha)
            delta_f1_weighted = float(f1_weighted_beta - f1_weighted_alpha)

        # ---- Local similarity: per-sample Cramér/Energy-based score on x_phi ----
        # Expect x_phi and a *real reference minibatch* in the same PCA stage space.
        # Here we use x_theta_val as the reference real batch for similarity (common choice).
        Xr = x_theta_val.detach()
        Ys = x_phi.detach()

        # Per-sample energy distances
        e_per_y = self._energy_per_sample(Xr, Ys)  # [T]

        # Calibrate anchors (d_rr ~ ideal low; d_rs ~ mismatched high)
        # For d_rr: split/permute x_theta_val to simulate two real minibatches
        # For d_rs: compare x_theta_val to a shuffled copy (breaks alignment)
        with torch.no_grad():
            # Build a shuffled copy (no gradient)
            idx = torch.randperm(Xr.size(0), device=Xr.device)
            Xr_shuf = Xr[idx]

            d_rr = self._batch_energy_scalar(Xr, Xr_shuf[: Xr.size(0)])  # real vs shuffled-real of same size
            # For a more "noisy" anchor, compare to a different segment (still OK to use shuffled)
            d_rs = self._batch_energy_scalar(Xr, Xr_shuf)

            # Update EMAs
            self._update_ema("d_rr", d_rr)
            self._update_ema("d_rs", d_rs)

            # Fallback to instantaneous if EMAs not warm yet
            d_rr_use = self._ema_d_rr if self._ema_d_rr is not None else float(d_rr)
            d_rs_use = self._ema_d_rs if self._ema_d_rs is not None else float(d_rs)

        # Distance -> [0,1] similarity per sample
        s_sim = self._distance_to_similarity(e_per_y, d_rr_use, d_rs_use)  # [T]

        # ---- Combine global + local (per-sample) ----
        # Broadcast global ΔF1_minority across samples
        global_term = torch.full_like(s_sim, fill_value=delta_f1_minority)
        reward = lambda_t * global_term + (1.0 - lambda_t) * s_sim  # [T]

        diagnostics = {
            "reward_mode": self.reward_mode,
            "lambda_t": lambda_t,

            # Global terms
            "global_reward": float(f1_minority_beta),
            "f1_macro_beta": float(f1_macro_beta),
            "delta_f1_minority": delta_f1_minority,
            "delta_f1_majority": delta_f1_majority,
            "delta_f1_weighted": delta_f1_weighted,
            "f1_minority_alpha": float(f1_minority_alpha),
            "f1_minority_beta": float(f1_minority_beta),
            "f1_weighted_alpha": float(f1_weighted_alpha),
            "f1_weighted_beta": float(f1_weighted_beta),

            # Local similarity stats
            "sim_mean": float(s_sim.mean().item()),
            "sim_std": float(s_sim.std(unbiased=False).item()),
            "energy_mean": float(e_per_y.mean().item()),
            "energy_min": float(e_per_y.min().item()),
            "energy_max": float(e_per_y.max().item()),

            # Anchors (EMAs)
            "anchor_d_rr": d_rr_use,
            "anchor_d_rs": d_rs_use,
            "alpha_sim": self.alpha_sim,
            "use_whitening": self.use_whitening,
        }

        return reward, diagnostics
