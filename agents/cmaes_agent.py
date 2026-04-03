"""
CMA-ES agent for synthetic data generation.

Replaces the REINFORCE policy network with a CMA-ES optimizer over a
parameterized Gaussian distribution in PCA space.

Parameter vector (2*D dimensional):
  params[:D]  = mean        (D-dimensional centroid in PCA space)
  params[D:]  = log_std     (per-dimension log standard deviation)

Each "episode" is one candidate evaluation:
  - Decode params -> Gaussian(mean, diag(exp(log_std)^2))
  - Draw traj_length i.i.d. samples
  - External code trains beta and computes reward
  - tell() records the reward

Each "generation" is popsize episodes. After a full generation,
step_generation() calls es.tell() and advances to the next generation.
"""

import numpy as np
import torch


LOG_STD_MIN = -4.0
LOG_STD_MAX =  3.0


class CMAESAgent:
    def __init__(
        self,
        pca_components: int,
        n_samples: int,
        init_mean: np.ndarray,
        init_log_std: np.ndarray,
        sigma0: float = 1.0,
        popsize: int | None = None,
        seed: int = 42,
        device="cpu",
        cmaes_opts: dict | None = None,
    ):
        try:
            import cma as _cma
            self._cma = _cma
        except ImportError:
            raise ImportError(
                "CMA-ES requires the 'cma' package: pip install cma"
            )

        self.D = pca_components
        self.n_samples = n_samples
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self._rng = np.random.default_rng(seed)

        # Initial parameter vector: [mean, log_std]
        self._init_params = np.concatenate([
            init_mean.astype(np.float64),
            np.clip(init_log_std.astype(np.float64), LOG_STD_MIN, LOG_STD_MAX),
        ])

        # CMA-ES options
        opts = self._cma.CMAOptions()
        opts["verbose"] = -9       # suppress output
        opts["seed"] = int(seed)
        if popsize is not None:
            opts["popsize"] = int(popsize)
        for k, v in (cmaes_opts or {}).items():
            opts[k] = v

        self._sigma0 = float(sigma0)
        self._opts = opts

        # State
        self._es = None
        self._candidates = []      # list of np.ndarray each shape (2D,)
        self._candidate_idx = 0
        self._rewards: dict = {}   # candidate_idx -> scalar reward

        # Derived after first es.ask()
        self.popsize: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, phase_label: str = ""):
        """Create a fresh CMA-ES instance. Call once per phase."""
        self._es = self._cma.CMAEvolutionStrategy(
            self._init_params.copy(),
            self._sigma0,
            self._opts,
        )
        self._candidates = self._es.ask()
        self.popsize = len(self._candidates)
        self._candidate_idx = 0
        self._rewards = {}
        print(
            f"[CMAESAgent] reset phase='{phase_label}' | "
            f"D={self.D} param_dim={2*self.D} popsize={self.popsize} "
            f"sigma0={self._sigma0:.3f}"
        )

    # ------------------------------------------------------------------
    # Per-episode interface
    # ------------------------------------------------------------------

    def ask(self) -> np.ndarray:
        """Return the parameter vector for the current candidate."""
        if self._es is None:
            raise RuntimeError("CMAESAgent.reset() must be called before ask()")
        return self._candidates[self._candidate_idx]

    def sample_synthetic(
        self,
        candidate_params: np.ndarray,
        target_class: int,
    ) -> tuple:
        """
        Decode params and draw n_samples i.i.d. from the Gaussian.
        Returns (x_syn, y_syn) as float32/long torch tensors on self.device.
        """
        mean    = candidate_params[:self.D]
        log_std = np.clip(candidate_params[self.D:], LOG_STD_MIN, LOG_STD_MAX)
        std     = np.exp(log_std)

        # i.i.d. Gaussian samples  shape (n_samples, D)
        noise   = self._rng.standard_normal((self.n_samples, self.D))
        samples = mean + std * noise   # broadcast

        x_syn = torch.tensor(samples, dtype=torch.float32, device=self.device)
        y_syn = torch.full((self.n_samples,), target_class, dtype=torch.long, device=self.device)
        return x_syn, y_syn

    def tell(self, candidate_idx: int, reward: float):
        """Record the reward for a candidate within the current generation."""
        self._rewards[candidate_idx] = float(reward)

    def advance(self):
        """Move to the next candidate. Call after tell()."""
        self._candidate_idx += 1

    # ------------------------------------------------------------------
    # End-of-generation update
    # ------------------------------------------------------------------

    def step_generation(self) -> dict:
        """
        Push evaluated candidates to CMA-ES and fetch the next generation.
        Returns es.stop() dict (empty = keep running, non-empty = converged).
        """
        n_eval = len(self._rewards)
        if n_eval == 0:
            return {}

        # Build parallel lists in candidate order (partial generation allowed)
        evaluated_idx = sorted(self._rewards.keys())
        solutions  = [self._candidates[i] for i in evaluated_idx]
        # CMA-ES minimises — negate rewards
        fitnesses  = [-self._rewards[i] for i in evaluated_idx]

        self._es.tell(solutions, fitnesses)

        stop = self._es.stop()
        if not stop:
            self._candidates = self._es.ask()
            self.popsize = len(self._candidates)

        self._candidate_idx = 0
        self._rewards = {}
        return stop

    # ------------------------------------------------------------------
    # Best result access
    # ------------------------------------------------------------------

    @property
    def sigma(self) -> float:
        """Current CMA-ES step size."""
        return float(self._es.sigma) if self._es is not None else self._sigma0

    def best_params(self) -> np.ndarray:
        """Best parameter vector seen so far."""
        if self._es is None:
            return self._init_params.copy()
        return np.array(self._es.result.xbest)

    def best_sample_synthetic(self, target_class: int) -> tuple:
        """Draw n_samples from the best-seen distribution (fixed seed for reproducibility)."""
        rng_det = np.random.default_rng(0)
        params  = self.best_params()
        mean    = params[:self.D]
        log_std = np.clip(params[self.D:], LOG_STD_MIN, LOG_STD_MAX)
        std     = np.exp(log_std)
        noise   = rng_det.standard_normal((self.n_samples, self.D))
        samples = mean + std * noise
        x_syn = torch.tensor(samples, dtype=torch.float32, device=self.device)
        y_syn = torch.full((self.n_samples,), target_class, dtype=torch.long, device=self.device)
        return x_syn, y_syn
