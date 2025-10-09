# testsuite.py
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split  # kept for backward-compatibility
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # kept for type hints/back-compat
from sklearn.decomposition import PCA  # kept for type hints/back-compat

# --- CTGAN / SDV ---
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    _HAS_SDV = True
except Exception:
    _HAS_SDV = False


class TestSuite:
    def __init__(self, seed_dir, experiment_dir, seed, run_id,
                 beta_factory=None, best_beta_path=None, alpha_factory=None, dataset=None):
        self.seed_dir = seed_dir
        self.experiment_dir = experiment_dir
        self.seed = seed
        self.run_id = run_id
        self.beta_factory = beta_factory
        self.best_beta_path = best_beta_path
        self.alpha_factory = alpha_factory or beta_factory
        self.dataset = dataset  # NEW: required for multi-dataset support

    # ---------------- Core helpers ----------------
    def _make_generator(self, device, seed):
        if seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen

    def _to_binary_rope_vs_rest(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map labels to rope-vs-rest: {0,2} -> 0, {1} -> 1, preserving dtype/device.
        Works for both 0/1 and 0/1/2 inputs.
        """
        return (y == 1).to(dtype=torch.long, device=y.device)


    def _p1_from_agent(self, agent, x):
        agent.model.eval()
        with torch.no_grad():
            logits = agent.model(x)
            probs = torch.softmax(logits, -1)
            return probs[..., 1]

    def _all_f1_from_probs(self, y_true, p1, threshold=0.5):
        """
        Compute rope-vs-rest F1s from P(y=1). We *only* binarize for metrics.
        Training labels can remain 0/1/2.
        """
        eps = 1e-8
        y_bin = self._to_binary_rope_vs_rest(y_true).to(p1.device)
        y_pred_pos = (p1 >= threshold).long()

        # Positive (rope=1) class
        tp1 = ((y_pred_pos == 1) & (y_bin == 1)).sum().float()
        fp1 = ((y_pred_pos == 1) & (y_bin == 0)).sum().float()
        fn1 = ((y_pred_pos == 0) & (y_bin == 1)).sum().float()
        prec1 = tp1 / (tp1 + fp1 + eps)
        rec1  = tp1 / (tp1 + fn1 + eps)
        f1_1  = (2 * prec1 * rec1) / (prec1 + rec1 + eps)

        # Negative (non-rope) class
        tp0 = ((y_pred_pos == 0) & (y_bin == 0)).sum().float()
        fp0 = ((y_pred_pos == 0) & (y_bin == 1)).sum().float()
        fn0 = ((y_pred_pos == 1) & (y_bin == 0)).sum().float()
        prec0 = tp0 / (tp0 + fn0 + eps)
        rec0  = tp0 / (tp0 + fp0 + eps)
        f1_0  = (2 * prec0 * rec0) / (prec0 + rec0 + eps)

        n1 = (y_bin == 1).sum().float()
        n0 = (y_bin == 0).sum().float()
        n  = n0 + n1 + eps

        f1_macro    = 0.5 * (f1_1 + f1_0)
        f1_weighted = (f1_1 * (n1 / n)) + (f1_0 * (n0 / n))
        return float(f1_1), float(f1_0), float(f1_weighted), float(f1_macro)


    def _brier_mean(self, y_true, p1):
        """
        Rope-vs-rest Brier: binarize then MSE on P(y=1).
        """
        y_bin = self._to_binary_rope_vs_rest(y_true).to(p1.device).float()
        return float(((p1 - y_bin) ** 2).mean())


    def _sample_real_minority(self, x_pool, y_pool, add_n: int, seed: int | None):
        add_n = int(max(0, add_n))
        if add_n == 0:
            return None, None

        with torch.no_grad():
            y1_mask = (y_pool == 1)
            if y1_mask.sum().item() == 0:
                return None, None
            idx_all = torch.nonzero(y1_mask, as_tuple=False).squeeze(1)
            gen = self._make_generator(x_pool.device, seed)
            # with replacement for availability
            rand_idx = torch.randint(0, idx_all.shape[0], (add_n,), device=idx_all.device, generator=gen)
            sel = idx_all[rand_idx]
            Xr = x_pool[sel]
            yr = torch.ones(add_n, dtype=y_pool.dtype, device=y_pool.device)
            return Xr, yr

    # ---------------- Dataset-facing wrappers (new) ----------------
    def _require_dataset(self):
        if self.dataset is None:
            raise ValueError(
                "TestSuite.dataset is required. Pass a Dataset instance "
                "with the desired dataset_name to TestSuite(..., dataset=Dataset(...))."
            )

    def _unbiased_pool_theta(self, *, bias_pct, val_frac, test_frac, train_size, pca_components, device):
        self._require_dataset()
        return self.dataset.rebuild_original_train_pool_theta(
            bias_pct=bias_pct,
            val_frac=val_frac,
            test_frac=test_frac,
            train_size=train_size,
            pca_components=pca_components,
            device=device,
        )

    def _ctgan_view(self, *, bias_pct, val_frac, test_frac, train_size, pca_components, device):
        self._require_dataset()
        return self.dataset.ctgan_training_view(
            bias_pct=bias_pct,
            val_frac=val_frac,
            test_frac=test_frac,
            train_size=train_size,
            pca_components=pca_components,
            device=device,
        )

    # ---------------- Existing baselines (β) ----------------
    def run_oversample_baseline(self, x_train, y_train, x_test, y_test,
                                f1_thresh: float = 0.5, batch_size: int = 32,
                                seed: int | None = None):
        if (x_train is None) or (y_train is None) or (self.beta_factory is None):
            print("[TestSuite] Oversample baseline not run (missing x_train/y_train or beta_factory).")
            return {
                "oversample_added": 0,
                "oversample_f1_minority": float('nan'),
                "oversample_f1_majority": float('nan'),
                "oversample_f1_weighted": float('nan'),
                "oversample_f1_macro": float('nan'),
                "oversample_brier": float('nan'),
            }

        # Majority is NON-rope (y != 1). Keep original 0/1/2 labels for training.
        y_is_min = (y_train == 1)
        y_is_maj = (y_train != 1)

        n1 = int(y_is_min.sum().item())
        nM = int(y_is_maj.sum().item())
        if n1 == 0 or nM == 0:
            print("[TestSuite] Oversample baseline skipped: one side missing in training.")
            return {
                "oversample_added": 0,
                "oversample_f1_minority": float('nan'),
                "oversample_f1_majority": float('nan'),
                "oversample_f1_weighted": float('nan'),
                "oversample_f1_macro": float('nan'),
                "oversample_brier": float('nan'),
            }

        n_add = max(0, nM - n1)
        if n_add > 0:
            idx_min_all = torch.nonzero(y_is_min, as_tuple=False).squeeze(1)
            gen = self._make_generator(idx_min_all.device, seed)
            rand_idx = torch.randint(0, idx_min_all.shape[0], (n_add,), device=idx_min_all.device, generator=gen)
            add_idx = idx_min_all[rand_idx]
            X_add = x_train[add_idx]
            y_add = y_train[add_idx]  # keep label==1
            x_os = torch.cat([x_train, X_add], dim=0)
            y_os = torch.cat([y_train, y_add], dim=0)
        else:
            x_os, y_os = x_train, y_train

        beta = self.beta_factory()
        if hasattr(beta.model, "to"): beta.model.to(x_train.device)
        dl = DataLoader(TensorDataset(x_os, y_os), batch_size=batch_size, shuffle=True)
        beta.train(dl)

        with torch.no_grad():
            p1 = self._p1_from_agent(beta, x_test)  # P(y=1|x), valid for 2 or 3 logits
            f1_min, f1_maj, f1_w, f1_macro = self._all_f1_from_probs(y_test, p1, f1_thresh)
            brier = self._brier_mean(y_test, p1)

        print(f"[TEST] OverSampleβ -> F1(rope)={f1_min:.4f} | F1(non-rope)={f1_maj:.4f} | "
            f"F1(weighted)={f1_w:.4f} | F1(macro)={f1_macro:.4f} | Brier={brier:.4f}")

        return {
            "oversample_added": int(n_add),
            "oversample_f1_minority": f1_min,
            "oversample_f1_majority": f1_maj,
            "oversample_f1_weighted": f1_w,
            "oversample_f1_macro": f1_macro,
            "oversample_brier": brier,
        }


    def run_undersample_baseline(self, x_train, y_train, x_test, y_test,
                                f1_thresh: float = 0.5, batch_size: int = 32,
                                seed: int | None = None):
        if (x_train is None) or (y_train is None) or (self.beta_factory is None):
            print("[TestSuite] Undersample baseline not run (missing x_train/y_train or beta_factory).")
            return {
                "undersample_dropped": 0,
                "undersample_f1_minority": float('nan'),
                "undersample_f1_majority": float('nan'),
                "undersample_f1_weighted": float('nan'),
                "undersample_f1_macro": float('nan'),
                "undersample_brier": float('nan'),
            }

        y_is_min = (y_train == 1)
        y_is_maj = (y_train != 1)

        n1 = int(y_is_min.sum().item())
        nM = int(y_is_maj.sum().item())
        if n1 == 0 or nM == 0:
            print("[TestSuite] Undersample baseline skipped: one side missing in training.")
            return {
                "undersample_dropped": 0,
                "undersample_f1_minority": float('nan'),
                "undersample_f1_majority": float('nan'),
                "undersample_f1_weighted": float('nan'),
                "undersample_f1_macro": float('nan'),
                "undersample_brier": float('nan'),
            }

        if nM > n1:
            idx_maj_all = torch.nonzero(y_is_maj, as_tuple=False).squeeze(1)
            gen = self._make_generator(idx_maj_all.device, seed)
            perm = torch.randperm(nM, device=idx_maj_all.device, generator=gen)[:n1]
            keep_maj_idx = idx_maj_all[perm]
            X_keep = x_train[keep_maj_idx]
            y_keep = y_train[keep_maj_idx]  # can be 0 or 2
            X_min  = x_train[y_is_min]
            y_min  = y_train[y_is_min]      # == 1
            x_us = torch.cat([X_keep, X_min], dim=0)
            y_us = torch.cat([y_keep, y_min], dim=0)
            n_drop = nM - n1
        else:
            x_us, y_us = x_train, y_train
            n_drop = 0

        beta = self.beta_factory()
        if hasattr(beta.model, "to"): beta.model.to(x_train.device)
        dl = DataLoader(TensorDataset(x_us, y_us), batch_size=batch_size, shuffle=True)
        beta.train(dl)

        with torch.no_grad():
            p1 = self._p1_from_agent(beta, x_test)
            f1_min, f1_maj, f1_w, f1_macro = self._all_f1_from_probs(y_test, p1, f1_thresh)
            brier = self._brier_mean(y_test, p1)

        print(f"[TEST] UnderSampleβ -> dropped={n_drop} | F1(rope)={f1_min:.4f} | F1(non-rope)={f1_maj:.4f} | "
            f"F1(weighted)={f1_w:.4f} | F1(macro)={f1_macro:.4f} | Brier={brier:.4f}")

        return {
            "undersample_dropped": int(max(0, n_drop)),
            "undersample_f1_minority": f1_min,
            "undersample_f1_majority": f1_maj,
            "undersample_f1_weighted": f1_w,
            "undersample_f1_macro": f1_macro,
            "undersample_brier": brier,
        }


    def run_jitter_baseline(self, x_train, y_train, x_test, y_test,
                            f1_thresh: float = 0.5,
                            jitter_n=None, jitter_scale: float = 0.20,
                            seed: int | None = None):
        if (x_train is None) or (y_train is None) or (self.beta_factory is None):
            print("[TestSuite] Jitter baseline not run (missing x_train/y_train or beta_factory).")
            return {
                "jitter_n": 0,
                "jitter_scale": float(jitter_scale),
                "jitter_f1_minority": float('nan'),
                "jitter_f1_majority": float('nan'),
                "jitter_f1_weighted": float('nan'),
                "jitter_f1_macro": float('nan'),
                "jitter_brier": float('nan'),
            }

        with torch.no_grad():
            Xmin = x_train[y_train == 1]
            if Xmin.numel() == 0:
                print("[TestSuite] Jitter baseline skipped: no minority samples in x_train.")
                return {
                    "jitter_n": 0,
                    "jitter_scale": float(jitter_scale),
                    "jitter_f1_minority": float('nan'),
                    "jitter_f1_majority": float('nan'),
                    "jitter_f1_weighted": float('nan'),
                    "jitter_f1_macro": float('nan'),
                    "jitter_brier": float('nan'),
                }

            if jitter_n is None:
                jitter_n = Xmin.shape[0]
            gen = self._make_generator(Xmin.device, seed)
            idx = torch.randint(0, Xmin.shape[0], (int(jitter_n),), device=Xmin.device, generator=gen)
            base = Xmin[idx]
            std = x_train.std(dim=0, unbiased=False)
            noise = torch.randn(base.shape, device=base.device, generator=gen) * (jitter_scale * (std + 1e-6))
            Xj = base + noise
            yj = torch.ones(int(jitter_n), dtype=y_train.dtype, device=y_train.device)

        jitter_beta = self.beta_factory()
        if hasattr(jitter_beta.model, "to"): jitter_beta.model.to(x_train.device)

        x_hybrid = torch.cat([x_train, Xj], dim=0)
        y_hybrid = torch.cat([y_train, yj], dim=0)
        dl = DataLoader(TensorDataset(x_hybrid, y_hybrid), batch_size=32, shuffle=True)
        jitter_beta.train(dl)

        with torch.no_grad():
            p1_j = self._p1_from_agent(jitter_beta, x_test)
            j_f1_min, j_f1_maj, j_f1_w, j_f1_macro = self._all_f1_from_probs(y_test, p1_j, f1_thresh)
            j_brier = self._brier_mean(y_test, p1_j)

        print(f"[TEST] Jitterβ -> F1(min)={j_f1_min:.4f} | F1(maj)={j_f1_maj:.4f} | "
              f"F1(weighted)={j_f1_w:.4f} | F1(macro)={j_f1_macro:.4f} | Brier={j_brier:.4f}")

        return {
            "jitter_n": int(jitter_n),
            "jitter_scale": float(jitter_scale),
            "jitter_f1_minority": j_f1_min,
            "jitter_f1_majority": j_f1_maj,
            "jitter_f1_weighted": j_f1_w,
            "jitter_f1_macro": j_f1_macro,
            "jitter_brier": j_brier,
        }

    # ---------------- New ALPHA baselines ----------------
    def run_alpha_raw_original_train(
        self,
        x_test, y_test,
        *,
        data_path: str = "census+income/adult.data",  # kept for drop-in compatibility (unused)
        bias_pct: float = 0.20,
        val_frac: float = 0.20,
        test_frac: float = 0.20,
        train_size: int | None = None,
        pca_components: int | None = None,  # if None, infer from x_test
        batch_size: int = 64,
        f1_thresh: float = 0.5,
    ):
        """
        Train alpha on the ORIGINAL TRAIN split (unbiased), preprocessed using
        enc/scale/PCA fitted on TRAIN-biased (to match experiment’s feature space).
        Uses Dataset.rebuild_original_train_pool_theta (dataset-agnostic).
        """
        if self.alpha_factory is None:
            print("[TestSuite] α_raw_original not run (alpha_factory missing).")
            return {
                "alpha_raworig_N": 0,
                "alpha_raworig_f1_minority": float('nan'),
                "alpha_raworig_f1_majority": float('nan'),
                "alpha_raworig_f1_weighted": float('nan'),
                "alpha_raworig_f1_macro": float('nan'),
                "alpha_raworig_brier": float('nan'),
            }

        device = x_test.device
        if pca_components is None:
            pca_components = int(x_test.shape[1])

        X_pool, y_pool = self._unbiased_pool_theta(
            bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac,
            train_size=train_size, pca_components=pca_components, device=device
        )

        model = self.alpha_factory()
        if hasattr(model.model, "to"): model.model.to(device)
        dl = DataLoader(TensorDataset(X_pool, y_pool), batch_size=batch_size, shuffle=True)
        model.train(dl)

        with torch.no_grad():
            p1 = self._p1_from_agent(model, x_test)
            f1_min, f1_maj, f1_w, f1_macro = self._all_f1_from_probs(y_test, p1, f1_thresh)
            brier = self._brier_mean(y_test, p1)

        print(f"[TEST] α_raw_original -> N={len(y_pool)} | F1(min)={f1_min:.4f} | F1(maj)={f1_maj:.4f} | "
              f"F1(w)={f1_w:.4f} | F1(macro)={f1_macro:.4f} | Brier={brier:.4f}")

        return {
            "alpha_raworig_N": int(len(y_pool)),
            "alpha_raworig_f1_minority": f1_min,
            "alpha_raworig_f1_majority": f1_maj,
            "alpha_raworig_f1_weighted": f1_w,
            "alpha_raworig_f1_macro": f1_macro,
            "alpha_raworig_brier": brier,
        }

    def run_alpha_same_train_plus_real_minority(
        self,
        x_train_used, y_train_used, x_test, y_test,
        *,
        real_add_n: int = 2000,
        data_path: str = "census+income/adult.data",  # kept for drop-in compatibility (unused)
        bias_pct: float = 0.20,
        val_frac: float = 0.20,
        test_frac: float = 0.20,
        train_size: int | None = None,
        pca_components: int | None = None,  # if None, infer from x_train_used
        batch_size: int = 64,
        f1_thresh: float = 0.5,
        seed: int | None = None,
    ):
        """
        Train alpha on the SAME train set you pass in, plus `real_add_n` real minority samples
        sampled from the ORIGINAL TRAIN split (unbiased), transformed into the same feature space.
        Uses Dataset.rebuild_original_train_pool_theta (dataset-agnostic).
        """
        if self.alpha_factory is None:
            print("[TestSuite] α_same+real not run (alpha_factory missing).")
            return {
                "alpha_plusreal_added": 0,
                "alpha_plusreal_final_N": int(x_train_used.shape[0] if x_train_used is not None else 0),
                "alpha_plusreal_pos_frac": float('nan'),
                "alpha_plusreal_f1_minority": float('nan'),
                "alpha_plusreal_f1_majority": float('nan'),
                "alpha_plusreal_f1_weighted": float('nan'),
                "alpha_plusreal_f1_macro": float('nan'),
                "alpha_plusreal_brier": float('nan'),
            }

        device = x_train_used.device
        if pca_components is None:
            pca_components = int(x_train_used.shape[1])

        # Rebuild original TRAIN pool (unbiased) in same space via Dataset
        X_pool, y_pool = self._unbiased_pool_theta(
            bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac, train_size=train_size,
            pca_components=pca_components, device=device
        )

        # Sample real minority from unbiased pool
        Xr, yr = self._sample_real_minority(X_pool, y_pool, int(real_add_n), seed)
        if Xr is not None:
            x_aug = torch.cat([x_train_used, Xr], dim=0)
            y_aug = torch.cat([y_train_used, yr], dim=0)
            added = int(real_add_n)
        else:
            x_aug, y_aug = x_train_used, y_train_used
            added = 0

        final_pos_frac = float((y_aug == 1).float().mean().item())

        model = self.alpha_factory()
        if hasattr(model.model, "to"): model.model.to(device)
        dl = DataLoader(TensorDataset(x_aug, y_aug), batch_size=batch_size, shuffle=True)
        model.train(dl)

        with torch.no_grad():
            p1 = self._p1_from_agent(model, x_test)
            f1_min, f1_maj, f1_w, f1_macro = self._all_f1_from_probs(y_test, p1, f1_thresh)
            brier = self._brier_mean(y_test, p1)

        print(f"[TEST] α_same+REAL -> added={added} | final_N={len(y_aug)} | pos_frac={final_pos_frac:.3f} | "
              f"F1(min)={f1_min:.4f} | F1(maj)={f1_maj:.4f} | F1(w)={f1_w:.4f} | F1(macro)={f1_macro:.4f} | "
              f"Brier={brier:.4f}")

        return {
            "alpha_plusreal_added": int(added),
            "alpha_plusreal_final_N": int(len(y_aug)),
            "alpha_plusreal_pos_frac": final_pos_frac,
            "alpha_plusreal_f1_minority": f1_min,
            "alpha_plusreal_f1_majority": f1_maj,
            "alpha_plusreal_f1_weighted": f1_w,
            "alpha_plusreal_f1_macro": f1_macro,
            "alpha_plusreal_brier": brier,
        }

    def run_ctgan_baseline(
        self,
        x_train_used: torch.Tensor, y_train_used: torch.Tensor,
        x_test: torch.Tensor, y_test: torch.Tensor,
        *,
        data_path: str = "census+income/adult.data",  # kept for drop-in compatibility (unused)
        bias_pct: float = 0.20,
        val_frac: float = 0.20,
        test_frac: float = 0.20,
        train_size: int | None = None,
        pca_components: int | None = None,  # if None, infer from x_train_used
        n_synth: int = 2000,
        ctgan_epochs: int = 300,
        cap_ctgan_train: int | None = None,
        batch_size: int = 64,
        f1_thresh: float = 0.5,
        seed: int | None = None,
    ):
        """
        Alpha + CTGAN: add synthetic minority (conditioned on label) mapped to θ-space.
        Uses Dataset.ctgan_training_view(). Skips cleanly if not supported (e.g., PAMAP2).
        """
        if not _HAS_SDV:
            print("[TestSuite] CTGAN baseline skipped: SDV/CTGAN not installed.")
            return {
                "alphaCT_added": 0, "alphaCT_final_N": int(x_train_used.shape[0]),
                "alphaCT_pos_frac": float((y_train_used == 1).float().mean().item()),
                "alphaCT_f1_minority": float('nan'), "alphaCT_f1_majority": float('nan'),
                "alphaCT_f1_weighted": float('nan'), "alphaCT_f1_macro": float('nan'),
                "alphaCT_brier": float('nan'),
            }

        device = x_train_used.device
        if pca_components is None:
            pca_components = int(x_train_used.shape[1])

        # Get raw unbiased TRAIN and θ-mapping transforms from Dataset
        view = self._ctgan_view(
            bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac,
            train_size=train_size, pca_components=pca_components, device=device
        )

        if not view.get("supported", False):
            print("[TestSuite] CTGAN baseline skipped: dataset does not support CTGAN view.")
            return {
                "alphaCT_added": 0, "alphaCT_final_N": int(x_train_used.shape[0]),
                "alphaCT_pos_frac": float((y_train_used == 1).float().mean().item()),
                "alphaCT_f1_minority": float('nan'), "alphaCT_f1_majority": float('nan'),
                "alphaCT_f1_weighted": float('nan'), "alphaCT_f1_macro": float('nan'),
                "alphaCT_brier": float('nan'),
            }

        X_train_unbiased_df = view["X_train_unbiased_df"]
        y_train_unbiased    = view["y_train_unbiased"]
        encoder = view["encoder"]
        scaler  = view["scaler"]
        pca     = view["pca"]
        cat_cols= view.get("cat_cols", [])
        num_cols= view.get("num_cols", [])

        # Train CTGAN (raw space) on ORIGINAL TRAIN (unbiased)
        synth = self._ctgan_train_conditional(
            X_train_unbiased_df, y_train_unbiased,
            label_col="__y__", ctgan_epochs=int(ctgan_epochs), seed=seed,
            cap_ctgan_train=cap_ctgan_train
        )

        # Sample conditional minority, clean, and map to θ-space
        syn_df = self._ctgan_sample_minority(synth, n_samples=int(n_synth), label_col="__y__", label_value=1)
        added_syn = int(len(syn_df))
        if added_syn > 0:
            syn_df_no_y = syn_df.drop(columns=["__y__"], errors="ignore")
            syn_cat = encoder.transform(syn_df_no_y[cat_cols]) if len(cat_cols) else np.empty((len(syn_df_no_y), 0))
            syn_num = scaler.transform(syn_df_no_y[num_cols])  if len(num_cols) else np.empty((len(syn_df_no_y), 0))
            syn_all = np.hstack([syn_num, syn_cat])
            syn_theta = pca.transform(syn_all)
            X_aug = torch.cat([x_train_used, torch.tensor(syn_theta, dtype=torch.float32, device=device)], dim=0)
            y_aug = torch.cat([y_train_used, torch.ones(added_syn, dtype=y_train_used.dtype, device=device)], dim=0)
        else:
            X_aug, y_aug = x_train_used, y_train_used

        pos_frac = float((y_aug == 1).float().mean().item())

        # Train/eval alpha on augmented set
        if self.alpha_factory is None:
            print("[TestSuite] CTGAN baseline skipped: alpha_factory missing.")
            return {
                "alphaCT_added": added_syn, "alphaCT_final_N": int(len(y_aug)),
                "alphaCT_pos_frac": pos_frac,
                "alphaCT_f1_minority": float('nan'), "alphaCT_f1_majority": float('nan'),
                "alphaCT_f1_weighted": float('nan'), "alphaCT_f1_macro": float('nan'),
                "alphaCT_brier": float('nan'),
            }

        model = self.alpha_factory()
        if hasattr(model.model, "to"):
            model.model.to(device)
        dl = DataLoader(TensorDataset(X_aug, y_aug), batch_size=int(batch_size), shuffle=True)
        model.train(dl)

        with torch.no_grad():
            p1 = self._p1_from_agent(model, x_test)
            f1_min, f1_maj, f1_w, f1_macro = self._all_f1_from_probs(y_test, p1, f1_thresh)
            brier = self._brier_mean(y_test, p1)

        print(f"[TEST] Alpha+CTGAN -> added={added_syn} | final_N={len(y_aug)} | pos_frac={pos_frac:.3f} | "
              f"F1(min)={f1_min:.4f} | F1(maj)={f1_maj:.4f} | F1(w)={f1_w:.4f} | F1(macro)={f1_macro:.4f} | "
              f"Brier={brier:.4f}")

        return {
            "alphaCT_added": int(added_syn),
            "alphaCT_final_N": int(len(y_aug)),
            "alphaCT_pos_frac": pos_frac,
            "alphaCT_f1_minority": f1_min,
            "alphaCT_f1_majority": f1_maj,
            "alphaCT_f1_weighted": f1_w,
            "alphaCT_f1_macro": f1_macro,
            "alphaCT_brier": brier,
        }

    # ---------------- Final test orchestrator ----------------
    def log_final_test(self, alpha_model, x_test, y_test, f1_thresh: float = 0.5,
                       prefer_best_beta: bool = True, beta_model=None,
                       x_train=None, y_train=None,
                       # existing jitter baseline params
                       jitter_n=None, jitter_scale: float = 0.20,
                       # NEW alpha toggles/params
                       run_alpha_raw_original: bool = True,
                       run_alpha_plus_real: bool = True,
                       alpha_plus_real_n: int = 2000,
                       # NEW CTGAN baseline toggles/params
                       run_alpha_plus_ctgan: bool = True,
                       alpha_plus_ctgan_n: int = 2000,
                       ctgan_epochs: int = 300,
                       cap_ctgan_train: int | None = None,
                       # dataset settings used to rebuild original pool
                       data_path: str = "census+income/adult.data",  # kept for drop-in compatibility (unused)
                       bias_pct: float = 0.20,
                       val_frac: float = 0.20,
                       test_frac: float = 0.20,
                       train_size: int | None = None,
                       seed: int | None = None):

        """
        End-of-run θ_test evaluation.
        Also (optionally) trains/evaluates:
          A) α_raw_original on original TRAIN (unbiased),
          B) α_same+REAL on the passed train set + real minority points from original TRAIN,
          C) α+CTGAN when supported by the dataset (skips otherwise).
        """
        beta_for_eval = None
        if prefer_best_beta and self.beta_factory is not None and self.best_beta_path and self.best_beta_path.exists():
            beta_for_eval = self.beta_factory()
            state = torch.load(self.best_beta_path, map_location=x_test.device)
            beta_for_eval.model.load_state_dict(state)
            print(f"[TestSuite] Loaded best β weights from: {self.best_beta_path}")
        elif beta_model is not None:
            beta_for_eval = beta_model
            print("[TestSuite] Using provided β (no best checkpoint found or factory missing).")
        else:
            print("[TestSuite] No β available for TEST evaluation (skipping β metrics).")

        with torch.no_grad():
            p1_alpha = self._p1_from_agent(alpha_model, x_test)
            a_f1_min, a_f1_maj, a_f1_w, a_f1_macro = self._all_f1_from_probs(y_test, p1_alpha, f1_thresh)
            a_brier = self._brier_mean(y_test, p1_alpha)

            if beta_for_eval is not None:
                p1_beta = self._p1_from_agent(beta_for_eval, x_test)
                b_f1_min, b_f1_maj, b_f1_w, b_f1_macro = self._all_f1_from_probs(y_test, p1_beta, f1_thresh)
                b_brier = self._brier_mean(y_test, p1_beta)
            else:
                b_f1_min = b_f1_maj = b_f1_w = b_f1_macro = b_brier = float('nan')

        # ----- built-in β baselines -----
        jitter_metrics = self.run_jitter_baseline(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            f1_thresh=f1_thresh, jitter_n=jitter_n, jitter_scale=jitter_scale, seed=seed
        ) if (x_train is not None and y_train is not None) else {}

        oversample_metrics = self.run_oversample_baseline(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            f1_thresh=f1_thresh, batch_size=32, seed=seed
        ) if (x_train is not None and y_train is not None) else {}

        undersample_metrics = self.run_undersample_baseline(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            f1_thresh=f1_thresh, batch_size=32, seed=seed
        ) if (x_train is not None and y_train is not None) else {}

        # ----- NEW α baselines (dataset-driven) -----
        alpha_raworig_metrics = self.run_alpha_raw_original_train(
            x_test=x_test, y_test=y_test,
            data_path=data_path,  # ignored, for back-compat
            bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac,
            train_size=train_size, pca_components=int(x_test.shape[1]), batch_size=64, f1_thresh=f1_thresh
        ) if run_alpha_raw_original else {}

        alpha_plusreal_metrics = self.run_alpha_same_train_plus_real_minority(
            x_train_used=x_train, y_train_used=y_train, x_test=x_test, y_test=y_test,
            real_add_n=int(alpha_plus_real_n),
            data_path=data_path,  # ignored, for back-compat
            bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac, train_size=train_size,
            pca_components=int(x_train.shape[1]) if x_train is not None else int(x_test.shape[1]),
            batch_size=64, f1_thresh=f1_thresh, seed=seed
        ) if (run_alpha_plus_real and x_train is not None and y_train is not None) else {}

        alpha_ctgan_metrics = self.run_ctgan_baseline(
            x_train_used=x_train, y_train_used=y_train, x_test=x_test, y_test=y_test,
            data_path=data_path,  # ignored, for back-compat
            bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac,
            train_size=train_size, pca_components=int(x_train.shape[1]) if x_train is not None else int(x_test.shape[1]),
            n_synth=int(alpha_plus_ctgan_n), ctgan_epochs=int(ctgan_epochs),
            cap_ctgan_train=cap_ctgan_train, batch_size=64, f1_thresh=f1_thresh, seed=seed
        ) if (run_alpha_plus_ctgan and x_train is not None and y_train is not None) else {}

        # ----- Console summary -----
        print("\n[TEST] Alpha -> F1(min)=%.4f | F1(maj)=%.4f | F1(weighted)=%.4f | F1(macro)=%.4f | Brier=%.4f"
              % (a_f1_min, a_f1_maj, a_f1_w, a_f1_macro, a_brier))
        if not np.isnan(b_f1_min):
            print("[TEST] Beta  -> F1(min)=%.4f | F1(maj)=%.4f | F1(weighted)=%.4f | F1(macro)=%.4f | Brier=%.4f"
                  % (b_f1_min, b_f1_maj, b_f1_w, b_f1_macro, b_brier))
        else:
            print("[TestSuite] Beta  -> (no checkpoint/factory)")

        # ----- Persist -----
        seed_csv = self.seed_dir / "final_test_metrics.csv"
        base_row = {
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "seed": str(self.seed),
            "run_id": self.run_id,
            "threshold": float(f1_thresh),

            "alpha_f1_minority": a_f1_min,
            "alpha_f1_majority": a_f1_maj,
            "alpha_f1_weighted": a_f1_w,
            "alpha_f1_macro": a_f1_macro,
            "alpha_brier": a_brier,

            "beta_f1_minority": b_f1_min,
            "beta_f1_majority": b_f1_maj,
            "beta_f1_weighted": b_f1_w,
            "beta_f1_macro": b_f1_macro,
            "beta_brier": b_brier,
        }

        row = dict(base_row)
        for metrics_var in [
            "jitter_metrics",
            "oversample_metrics",
            "undersample_metrics",
            "alpha_raworig_metrics",
            "alpha_plusreal_metrics",
            "alpha_ctgan_metrics",
        ]:
            if metrics_var in locals() and locals()[metrics_var] is not None:
                row.update(locals()[metrics_var])

        pd.DataFrame([row]).to_csv(seed_csv, mode="a", header=not seed_csv.exists(), index=False)
        print(f"[TestSuite] Final test metrics appended to (seed): {seed_csv}")

        exp_csv = self.experiment_dir / "final_test_metrics.csv"
        pd.DataFrame([row]).to_csv(exp_csv, mode="a", header=not exp_csv.exists(), index=False)
        print(f"[TestSuite] Final test metrics appended to (experiment): {exp_csv}")
        return row