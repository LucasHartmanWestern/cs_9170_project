# testsuite.py
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split  # kept for backward-compatibility
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # kept for type hints/back-compat
from sklearn.decomposition import PCA  # kept for type hints/back-compat
import json, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

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
    # --- ADD THESE HELPERS INSIDE TestSuite (near your other metric helpers) ---

    def _safe_rate(self, num: torch.Tensor, den: torch.Tensor) -> float:
        """Return num/den as float, NaN if den==0."""
        den_v = float(den.item()) if torch.is_tensor(den) else float(den)
        if den_v <= 0:
            return float("nan")
        return float((num.float() / den.float()).item())

    def _fairness_from_probs(self, y_true: torch.Tensor, p1: torch.Tensor, a: torch.Tensor, threshold: float = 0.5):
        """
        Fairness metrics on rope-vs-rest (binary):
        - DP diff: |P(ŷ=1|a=0) - P(ŷ=1|a=1)|
        - EO (Equal Opportunity) TPR diff: |TPR(a=0) - TPR(a=1)|
        - EOd (Equalized Odds) max diff: max(|TPR diff|, |FPR diff|)
        Also returns groupwise rates for debugging.

        Notes:
        - Assumes binary protected attribute a in {0,1}. Anything else -> coerced to {0,1} by (a!=0).
        - If a group has no samples / no positives / no negatives -> corresponding rate is NaN.
        """
        device = p1.device
        y_bin = self._to_binary_rope_vs_rest(y_true).to(device=device).long()
        y_hat = (p1 >= float(threshold)).to(device=device).long()

        a = torch.as_tensor(a, device=device).long()
        # coerce to {0,1} robustly
        a01 = (a != 0).long()

        out = {}

        def mask_g(g: int):
            return (a01 == int(g))

        # --- DP: P(y_hat=1 | a=g) ---
        dp_rates = {}
        for g in (0, 1):
            m = mask_g(g)
            den = m.sum()
            num = ((y_hat == 1) & m).sum()
            dp_rates[g] = self._safe_rate(num, den)

        # --- TPR/FPR per group ---
        tpr = {}
        fpr = {}
        for g in (0, 1):
            m = mask_g(g)

            pos = m & (y_bin == 1)
            neg = m & (y_bin == 0)

            tp = ((y_hat == 1) & pos).sum()
            fn = ((y_hat == 0) & pos).sum()
            fp = ((y_hat == 1) & neg).sum()
            tn = ((y_hat == 0) & neg).sum()

            tpr[g] = self._safe_rate(tp, tp + fn)
            fpr[g] = self._safe_rate(fp, fp + tn)

        # diffs (absolute)
        def abs_diff(x, y):
            if np.isnan(x) or np.isnan(y):
                return float("nan")
            return float(abs(x - y))

        dp_diff = abs_diff(dp_rates[0], dp_rates[1])
        tpr_diff = abs_diff(tpr[0], tpr[1])
        fpr_diff = abs_diff(fpr[0], fpr[1])

        eod_max = float("nan")
        if not np.isnan(tpr_diff) and not np.isnan(fpr_diff):
            eod_max = float(max(tpr_diff, fpr_diff))

        # pack
        out["dp_rate_g0"] = dp_rates[0]
        out["dp_rate_g1"] = dp_rates[1]
        out["dp_diff"] = dp_diff

        out["tpr_g0"] = tpr[0]
        out["tpr_g1"] = tpr[1]
        out["eo_tpr_diff"] = tpr_diff

        out["fpr_g0"] = fpr[0]
        out["fpr_g1"] = fpr[1]
        out["eod_fpr_diff"] = fpr_diff
        out["eod_max_diff"] = eod_max

        # (optional) also provide "avg" equalized-odds gap
        if not np.isnan(tpr_diff) and not np.isnan(fpr_diff):
            out["eod_avg_diff"] = float(0.5 * (tpr_diff + fpr_diff))
        else:
            out["eod_avg_diff"] = float("nan")

        return out


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
    def _ctgan_train_conditional(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        *,
        label_col: str = "__y__",
        ctgan_epochs: int = 300,
        seed: int | None = None,
        cap_ctgan_train: int | None = None,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not _HAS_SDV:
            raise RuntimeError("SDV/CTGAN not installed")

        df = X_df.copy().reset_index(drop=True)
        df[label_col] = np.asarray(y, dtype=int)

        if cap_ctgan_train is not None and cap_ctgan_train < len(df):
            df = df.sample(n=int(cap_ctgan_train), random_state=seed).reset_index(drop=True)

        # Build metadata and force label to categorical
        md = SingleTableMetadata()
        md.detect_from_dataframe(df)
        try:
            md.update_column(label_col, sdtype="categorical")
        except Exception:
            pass

        synth = CTGANSynthesizer(metadata=md, epochs=int(ctgan_epochs))
        synth.fit(df)
        return synth


    def _ctgan_sample_minority(
        self,
        synth,
        n_samples: int,
        *,
        label_col: str = "__y__",
        label_value: int = 1,
    ) -> pd.DataFrame:
        n_samples = int(n_samples)

        # SDV >= 1.0: sample(num_rows=..., conditions=DataFrame)
        cond_df = pd.DataFrame({label_col: [label_value] * n_samples})
        try:
            return synth.sample(num_rows=n_samples, conditions=cond_df)
        except Exception:
            pass

        # Older SDV: sample_conditions([{label_col: value, "num_rows": n}])
        try:
            out = synth.sample_conditions([{label_col: label_value, "num_rows": n_samples}])
            if isinstance(out, list):
                out = pd.concat(out, ignore_index=True)
            return out
        except Exception:
            pass

        # Fallback: unconditional sample then filter
        tmp = synth.sample(num_rows=max(n_samples * 2, 1000))
        if label_col in tmp.columns:
            tmp = tmp[tmp[label_col] == label_value]
            if len(tmp) >= n_samples:
                return tmp.sample(n=n_samples, replace=False).reset_index(drop=True)
            # pad if short
            if len(tmp) > 0:
                need = n_samples - len(tmp)
                return pd.concat([tmp, tmp.sample(n=need, replace=True)], ignore_index=True)
        # Replace the final return with:
        return pd.DataFrame()  # empty; caller checks len() and skips cleanly
 # last-resort empty frame

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
        bias_pct = None,
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
        bias_pct = None,
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
        bias_pct = None,
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
        print(f"[TestSuite] CTGAN Running...")
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



    def run_ctabgan_baseline(
        self,
        x_train_used: torch.Tensor, y_train_used: torch.Tensor,
        x_test: torch.Tensor, y_test: torch.Tensor,
        *,
        data_path: str = "census+income/adult.data",  # kept for drop-in compatibility (unused)
        bias_pct = None,
        val_frac: float = 0.20,
        test_frac: float = 0.20,
        train_size: int | None = None,
        pca_components: int | None = None,  # if None, infer from x_train_used
        n_synth: int = 2000,
        batch_size: int = 64,
        f1_thresh: float = 0.5,
        seed: int | None = None,
        ctab_python: str = "/home/epigou/envs/ctabgan/bin/python",
        ctab_repo: str = "/home/epigou/CTAB-GAN-Plus-DP",
        ctab_runner: str = "",
    ):
        """
        Alpha + CTAB-GAN: add synthetic minority (conditioned on label) mapped to θ-space.
        Mirrors run_ctgan_baseline() training/eval path for fair comparison.
        """
        print(f"[TestSuite] CTABGAN Running...")

        device = x_train_used.device
        if pca_components is None:
            pca_components = int(x_train_used.shape[1])

        # ---- Get raw unbiased TRAIN and θ-mapping transforms (same as CTGAN baseline) ----
        view = self._ctgan_view(
            bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac,
            train_size=train_size, pca_components=pca_components, device=device
        )

        if not view.get("supported", False):
            print("[TestSuite] CTABGAN baseline skipped: dataset does not support CTGAN view.")
            return {
                "alphaCTAB_added": 0, "alphaCTAB_final_N": int(x_train_used.shape[0]),
                "alphaCTAB_pos_frac": float((y_train_used == 1).float().mean().item()),
                "alphaCTAB_f1_minority": float('nan'), "alphaCTAB_f1_majority": float('nan'),
                "alphaCTAB_f1_weighted": float('nan'), "alphaCTAB_f1_macro": float('nan'),
                "alphaCTAB_brier": float('nan'),
            }

        X_train_unbiased_df = view["X_train_unbiased_df"]
        y_train_unbiased    = view["y_train_unbiased"].astype(int)
        encoder = view["encoder"]
        scaler  = view["scaler"]
        pca     = view["pca"]
        cat_cols= list(view.get("cat_cols", []))
        num_cols= list(view.get("num_cols", []))

        # ---- Write CTAB input CSV (raw + label) ----
        label_col = "__y__"
        out_dir = Path(self.seed_dir) / "ctabgan"
        out_dir.mkdir(parents=True, exist_ok=True)

        real_csv = out_dir / "ctab_real_train.csv"
        df_real = X_train_unbiased_df.copy()
        df_real[label_col] = y_train_unbiased
        df_real.to_csv(real_csv, index=False)

        out_csv = out_dir / f"ctab_synth_seed{int(seed) if seed is not None else 0}.csv"

        # ---- Column metadata for CTAB ----
        categorical_cols = cat_cols + [label_col]
        mixed_cols = {}
        general_cols = []
        integer_cols = []

        # ---- Call CTAB runner: minority-only samples ----
        cmd = [
            ctab_python, ctab_runner,
            "--ctab_repo", ctab_repo,
            "--real_csv", str(real_csv),
            "--out_csv", str(out_csv),
            "--label_col", label_col,
            "--n_rows", str(int(n_synth)),
            "--seed", str(int(seed) if seed is not None else 0),
            "--categorical_cols_json", json.dumps(categorical_cols),
            "--mixed_cols_json", json.dumps(mixed_cols),
            "--general_cols_json", json.dumps(general_cols),
            "--integer_cols_json", json.dumps(integer_cols),
            "--filter_to_label", "1",
            "--oversample_factor", "5",
        ]
        subprocess.check_call(cmd)

        # ---- Load synthetic and map to θ-space (same mapping logic as CTGAN baseline) ----
        syn_df = pd.read_csv(out_csv)
        added_syn = int(len(syn_df))

        if added_syn > 0:
            syn_df_no_y = syn_df.drop(columns=[label_col], errors="ignore")
            syn_df_no_y = syn_df_no_y[X_train_unbiased_df.columns]  # enforce order

            syn_cat = encoder.transform(syn_df_no_y[cat_cols]) if len(cat_cols) else np.empty((len(syn_df_no_y), 0))
            syn_num = scaler.transform(syn_df_no_y[num_cols])  if len(num_cols) else np.empty((len(syn_df_no_y), 0))

            syn_all = np.hstack([syn_num, syn_cat])
            syn_theta = pca.transform(syn_all)

            X_aug = torch.cat([x_train_used, torch.tensor(syn_theta, dtype=torch.float32, device=device)], dim=0)
            y_aug = torch.cat([y_train_used, torch.ones(added_syn, dtype=y_train_used.dtype, device=device)], dim=0)
        else:
            X_aug, y_aug = x_train_used, y_train_used

        pos_frac = float((y_aug == 1).float().mean().item())

        # ---- Train/eval alpha on augmented set (EXACTLY like CTGAN baseline) ----
        if self.alpha_factory is None:
            print("[TestSuite] CTABGAN baseline skipped: alpha_factory missing.")
            return {
                "alphaCTAB_added": added_syn, "alphaCTAB_final_N": int(x_train_used.shape[0]),
                "alphaCTAB_pos_frac": pos_frac,
                "alphaCTAB_f1_minority": float('nan'), "alphaCTAB_f1_majority": float('nan'),
                "alphaCTAB_f1_weighted": float('nan'), "alphaCTAB_f1_macro": float('nan'),
                "alphaCTAB_brier": float('nan'),
            }

        model = self.alpha_factory()
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)

        dl = DataLoader(TensorDataset(X_aug, y_aug), batch_size=int(batch_size), shuffle=True)
        model.train(dl)

        with torch.no_grad():
            p1 = self._p1_from_agent(model, x_test)
            f1_min, f1_maj, f1_w, f1_macro = self._all_f1_from_probs(y_test, p1, f1_thresh)
            brier = self._brier_mean(y_test, p1)

        print(f"[TEST] Alpha+CTABGAN -> added={added_syn} | final_N={len(y_aug)} | pos_frac={pos_frac:.3f} | "
            f"F1(min)={f1_min:.4f} | F1(maj)={f1_maj:.4f} | F1(w)={f1_w:.4f} | F1(macro)={f1_macro:.4f} | "
            f"Brier={brier:.4f}")

        return {
            "alphaCTAB_added": int(added_syn),
            "alphaCTAB_final_N": int(len(y_aug)),
            "alphaCTAB_pos_frac": pos_frac,
            "alphaCTAB_f1_minority": f1_min,
            "alphaCTAB_f1_majority": f1_maj,
            "alphaCTAB_f1_weighted": f1_w,
            "alphaCTAB_f1_macro": f1_macro,
            "alphaCTAB_brier": brier,
        }



    # ---------------- Final test orchestrator ----------------
    def log_final_test(
        self, alpha_model, x_test, y_test, f1_thresh: float = 0.5,
        prefer_best_beta: bool = True, beta_model=None,
        x_train=None, y_train=None,
        jitter_n=None, jitter_scale: float = 0.20,
        run_alpha_raw_original: bool = True,
        run_alpha_plus_real: bool = True,
        alpha_plus_real_n: int = 2000,
        run_alpha_plus_ctgan: bool = False,
        alpha_plus_ctgan_n: int = 2000,
        ctgan_epochs: int = 300,
        cap_ctgan_train: int | None = None,
        run_ctabgan: bool = False,
        alpha_plus_ctabgan_n: int = 2000,
        # CTABGAN subprocess wiring (default None)
        ctab_python: str | None = None,
        ctab_repo: str | None = None,
        ctab_runner: str | None = None,
        data_path: str = "census+income/adult.data",
        bias_pct = None,
        val_frac: float = 0.20,
        test_frac: float = 0.20,
        train_size: int | None = None,
        batch_size: int = 64,
        pca_components: int | None = None,
        seed: int | None = None,
        a_test: torch.Tensor | None = None,   # <--- NEW
    ):
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
        # --- NEW: fairness metrics (DP / EO / EOd) for alpha and beta ---
        if a_test is None:
            print("[TestSuite] a_test (protected attribute) not provided; fairness metrics will be NaN in final_test_metrics.csv")
            alpha_fair = {
                "dp_diff": float("nan"),
                "eo_tpr_diff": float("nan"),
                "eod_max_diff": float("nan"),
                "eod_avg_diff": float("nan"),
                "dp_rate_g0": float("nan"), "dp_rate_g1": float("nan"),
                "tpr_g0": float("nan"), "tpr_g1": float("nan"),
                "fpr_g0": float("nan"), "fpr_g1": float("nan"),
                "eod_fpr_diff": float("nan"),
            }
            beta_fair = dict(alpha_fair)
        else:
            alpha_fair = self._fairness_from_probs(y_test, p1_alpha, a_test, threshold=f1_thresh)
            if p1_beta is not None:
                beta_fair = self._fairness_from_probs(y_test, p1_beta, a_test, threshold=f1_thresh)
            else:
                beta_fair = {k: float("nan") for k in alpha_fair.keys()}
        # ----- built-in β baselines -----
        # jitter_metrics = self.run_jitter_baseline(
        #     x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        #     f1_thresh=f1_thresh, jitter_n=jitter_n, jitter_scale=jitter_scale, seed=seed
        # ) if (x_train is not None and y_train is not None) else {}

        # oversample_metrics = self.run_oversample_baseline(
        #     x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        #     f1_thresh=f1_thresh, batch_size=32, seed=seed
        # ) if (x_train is not None and y_train is not None) else {}

        # undersample_metrics = self.run_undersample_baseline(
        #     x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        #     f1_thresh=f1_thresh, batch_size=32, seed=seed
        # ) if (x_train is not None and y_train is not None) else {}

        # # ----- NEW α baselines (dataset-driven) -----
        # alpha_raworig_metrics = self.run_alpha_raw_original_train(
        #     x_test=x_test, y_test=y_test,
        #     data_path=data_path,  # ignored, for back-compat
        #     bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac,
        #     train_size=train_size, pca_components=int(x_test.shape[1]), batch_size=64, f1_thresh=f1_thresh
        # ) if run_alpha_raw_original else {}

        # alpha_plusreal_metrics = self.run_alpha_same_train_plus_real_minority(
        #     x_train_used=x_train, y_train_used=y_train, x_test=x_test, y_test=y_test,
        #     real_add_n=int(alpha_plus_real_n),
        #     data_path=data_path,  # ignored, for back-compat
        #     bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac, train_size=train_size,
        #     pca_components=int(x_train.shape[1]) if x_train is not None else int(x_test.shape[1]),
        #     batch_size=64, f1_thresh=f1_thresh, seed=seed
        # ) if (run_alpha_plus_real and x_train is not None and y_train is not None) else {}

        alpha_ctgan_metrics = self.run_ctgan_baseline(
            x_train_used=x_train, y_train_used=y_train, x_test=x_test, y_test=y_test,
            data_path=data_path,
            bias_pct=bias_pct, val_frac=val_frac, test_frac=test_frac,
            train_size=train_size,
            pca_components=int(x_train.shape[1]) if x_train is not None else int(x_test.shape[1]),
            n_synth=int(alpha_plus_ctgan_n), ctgan_epochs=int(ctgan_epochs),
            cap_ctgan_train=cap_ctgan_train,
            batch_size=int(batch_size),         # <-- CHANGED
            f1_thresh=f1_thresh, seed=seed
        ) if (run_alpha_plus_ctgan and x_train is not None and y_train is not None) else {}


        ctabgan_metrics = {}
        if run_ctabgan and x_train is not None and y_train is not None:
            # If user didn't pass a runner path, you can require it (recommended).
            if ctab_runner is None:
                print("[TestSuite] CTABGAN requested but ctab_runner is None. Skipping CTABGAN baseline.")
            else:
                ctabgan_metrics = self.run_ctabgan_baseline(
                    x_train_used=x_train,
                    y_train_used=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    data_path=data_path,
                    bias_pct=bias_pct,
                    val_frac=val_frac,
                    test_frac=test_frac,
                    train_size=train_size,
                    pca_components=int(x_train.shape[1]) if x_train is not None else int(x_test.shape[1]),
                    n_synth=int(alpha_plus_ctabgan_n),
                    batch_size=int(batch_size),      # <-- CHANGED
                    f1_thresh=f1_thresh,
                    seed=seed,
                    ctab_python=ctab_python,         # may be None; runner should use sys.executable
                    ctab_repo=ctab_repo,             # may be None; runner can require it
                    ctab_runner=ctab_runner,
                )




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
            # --- NEW fairness (alpha) ---
            "alpha_dp_diff": alpha_fair["dp_diff"],
            "alpha_eo_tpr_diff": alpha_fair["eo_tpr_diff"],
            "alpha_eod_max_diff": alpha_fair["eod_max_diff"],
            "alpha_eod_avg_diff": alpha_fair["eod_avg_diff"],
            "alpha_dp_rate_g0": alpha_fair["dp_rate_g0"],
            "alpha_dp_rate_g1": alpha_fair["dp_rate_g1"],
            "alpha_tpr_g0": alpha_fair["tpr_g0"],
            "alpha_tpr_g1": alpha_fair["tpr_g1"],
            "alpha_fpr_g0": alpha_fair["fpr_g0"],
            "alpha_fpr_g1": alpha_fair["fpr_g1"],

            # --- NEW fairness (beta) ---
            "beta_dp_diff": beta_fair["dp_diff"],
            "beta_eo_tpr_diff": beta_fair["eo_tpr_diff"],
            "beta_eod_max_diff": beta_fair["eod_max_diff"],
            "beta_eod_avg_diff": beta_fair["eod_avg_diff"],
            "beta_dp_rate_g0": beta_fair["dp_rate_g0"],
            "beta_dp_rate_g1": beta_fair["dp_rate_g1"],
            "beta_tpr_g0": beta_fair["tpr_g0"],
            "beta_tpr_g1": beta_fair["tpr_g1"],
            "beta_fpr_g0": beta_fair["fpr_g0"],
            "beta_fpr_g1": beta_fair["fpr_g1"],
            }

        row = dict(base_row)
        for metrics_var in [
            "jitter_metrics",
            "oversample_metrics",
            "undersample_metrics",
            "alpha_raworig_metrics",
            "alpha_plusreal_metrics",
            "alpha_ctgan_metrics",
            "ctabgan_metrics"
        ]:
            if metrics_var in locals() and locals()[metrics_var] is not None:
                row.update(locals()[metrics_var])

        pd.DataFrame([row]).to_csv(seed_csv, mode="a", header=not seed_csv.exists(), index=False)
        print(f"[TestSuite] Final test metrics appended to (seed): {seed_csv}")

        exp_csv = self.experiment_dir / "final_test_metrics.csv"
        pd.DataFrame([row]).to_csv(exp_csv, mode="a", header=not exp_csv.exists(), index=False)
        print(f"[TestSuite] Final test metrics appended to (experiment): {exp_csv}")
        return row