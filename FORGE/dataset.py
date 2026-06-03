import ast
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

import torch

class Dataset:
    DATASET_REGISTRY = {
        "census_income": {
            "data_path": "datasets/census+income/adult.data",
            "protected_attributes": ["sex", "race", "age", "native-country"]
        },
        "capture24": {
            "data_path": "datasets/capture24",
            "protected_attributes": ["sex"]
        },
    }
    def __init__(self, dataset_name, minority_id, majority_id, pca_components=2, seed=42, device="cpu", use_pca: bool = False):
        self.dataset_name = dataset_name
        self.MINORITY_ID = minority_id
        self.MAJORITY_ID = majority_id
        self.pca_components = pca_components
        self.seed = seed
        self.device = device
        self.data_path = self.DATASET_REGISTRY[dataset_name]["data_path"]
        self.protected_attributes = self.DATASET_REGISTRY[self.dataset_name].get("protected_attributes", [])
        self.use_pca = bool(use_pca)

    def _to_theta(
        self,
        X_train_all: np.ndarray,
        X_val_all: np.ndarray,
        X_test_all: np.ndarray,
        *,
        pca_components: int,
    ):
        """
        Map engineered features -> theta space.
        If use_pca=False: identity mapping (raw engineered features).
        If use_pca=True: PCA fitted on TRAIN only.

        Returns:
            X_train_theta, X_val_theta, X_test_theta, pca_or_none
        """
        if not getattr(self, "use_pca", False):
            return X_train_all, X_val_all, X_test_all, None
        d = X_train_all.shape[1]
        k = min(int(pca_components), int(d))

        pca = self._make_pca(k)
        X_train_theta = pca.fit_transform(X_train_all)
        X_val_theta   = pca.transform(X_val_all)
        X_test_theta  = pca.transform(X_test_all)
        self.pca_transform = pca  # stored for post-hoc analysis (check_run.py)
        return X_train_theta, X_val_theta, X_test_theta, pca

    def _make_pca(self, n_components: int) -> PCA:
        return PCA(
            n_components=n_components,
            svd_solver="full",
            random_state=self.seed,
        )

    def split_census_income(
            self,
            train_size=None,
            da_pct=None,
            val_frac=0.20,
            test_frac=0.20,
            pca_components=2,
            drop_protected: bool = False,
            protected_cols=None,
            dp_protected_col: str = "sex",
        ):
        """
        PIPELINE:
        1) Load raw data.
        2) (Optional) Drop protected attributes entirely.
        3) Split raw data into θ_train / θ_val / θ_test (stratified by original labels).
        4) Within each split, apply the *same bias* by downsampling the minority class.
        5) Fit OneHotEncoder + StandardScaler on θ_train only; transform θ_val and θ_test (no leakage).
        6) Fit PCA on θ_train only; transform θ_val and θ_test (no leakage).
        7) Optionally subsample θ_train to a fixed size after biasing.
        8) Return torch tensors for train/val/test on self.device.
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1, \
            "val_frac and test_frac must be in (0,1) and sum to < 1."

        if protected_cols is None:
            protected_cols = ["sex", "race", "age", "native-country"]

        # 1) Load raw data
        data_path = self.data_path
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        X_df_raw = pd.read_csv(
            data_path,
            header=None,
            names=column_names,
            na_values="?",
            skipinitialspace=True
        )

        X_df_raw = X_df_raw.dropna()
        y_raw = np.where(X_df_raw["income"].isin([">50K", ">50K."]), 1, 0).astype(int)
        X_df_raw = X_df_raw.drop(columns=["income"])

        # Extract protected attribute for fairness metrics (before any dropping)
        A_df_raw = X_df_raw[[dp_protected_col]].copy()

        # 2) OPTIONAL: Drop protected attributes entirely (before splits/encoding/PCA)
        if drop_protected:
            drop_cols = [c for c in protected_cols if c in X_df_raw.columns]
            if len(drop_cols) > 0:
                X_df_raw = X_df_raw.drop(columns=drop_cols)

        # Identify column types (after optional drop)
        cat_cols = [c for c in X_df_raw.columns if X_df_raw[c].dtype.name in ["category", "object", "bool"]]
        num_cols = [c for c in X_df_raw.columns if np.issubdtype(X_df_raw[c].dtype, np.number)]

        # 3) Split raw into θ_train / θ_temp, then θ_val / θ_test
        X_train_df, X_temp_df, y_train, y_temp, A_train_df, A_temp_df = train_test_split(
            X_df_raw, y_raw, A_df_raw, test_size=(val_frac + test_frac),
            random_state=self.seed, stratify=y_raw
        )
        rel_test = test_frac / (val_frac + test_frac)
        X_val_df, X_test_df, y_val, y_test, A_val_df, A_test_df = train_test_split(
            X_temp_df, y_temp, A_temp_df, test_size=rel_test,
            random_state=self.seed, stratify=y_temp
        )

        # Helper: map protected attribute to 0/1
        def _map_protected_census(a_series):
            return (a_series.str.strip().str.lower() == "male").astype(np.int64).to_numpy()

        # Helper: group-specific DA+ targeting.
        # Targets exactly round(da_pct * train_size) disadvantaged-group (female, a=0)
        # positives. Keeps all advantaged-group positives and all negatives. Fills the
        # remainder of train_size from the non-disadvantaged-positive pool.
        def apply_da_pct_bias(df_split, y_split, a_split_df, da_pct, n_total):
            df = df_split.copy()
            df["__y__"] = y_split
            df["__a__"] = _map_protected_census(a_split_df[dp_protected_col])

            target_da_plus = max(1, round(da_pct * n_total))

            disadv_pos_mask = (df["__a__"] == 0) & (df["__y__"] == 1)  # female positive
            disadv_pos = df[disadv_pos_mask]
            rest       = df[~disadv_pos_mask]  # adv positives + all negatives

            keep_n = min(len(disadv_pos), target_da_plus)
            if keep_n < target_da_plus:
                print(f"  [da_pct WARNING] Only {len(disadv_pos)} disadvantaged positives "
                      f"available; target was {target_da_plus}.")
            disadv_pos_kept = disadv_pos.sample(n=keep_n, random_state=self.seed, replace=False)

            n_rest_needed = n_total - keep_n
            if len(rest) >= n_rest_needed:
                rest_kept = rest.sample(n=n_rest_needed, random_state=self.seed, replace=False)
            else:
                rest_kept = rest

            result = (pd.concat([disadv_pos_kept, rest_kept], axis=0)
                      .sample(frac=1.0, random_state=self.seed)
                      .reset_index(drop=True))

            y_out = result["__y__"].to_numpy(dtype=int)
            a_out = result["__a__"].to_numpy()
            X_out = result.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        # 4) Apply DA+ bias to train; val/test unbiased.
        if da_pct is not None:
            n_total = train_size if train_size is not None else len(X_train_df)
            X_train_biased_df, y_train_biased, a_train = apply_da_pct_bias(
                X_train_df, y_train, A_train_df, da_pct, n_total)
        else:
            X_train_biased_df, y_train_biased = X_train_df.copy().reset_index(drop=True), y_train.copy()
            a_train = _map_protected_census(A_train_df[dp_protected_col])

        X_val_biased_df, y_val_biased = X_val_df.copy().reset_index(drop=True), y_val.copy()
        a_val = _map_protected_census(A_val_df[dp_protected_col])

        X_test_biased_df, y_test_biased = X_test_df.copy().reset_index(drop=True), y_test.copy()
        a_test = _map_protected_census(A_test_df[dp_protected_col])

        if da_pct is None and train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _, a_train, _ = train_test_split(
                X_train_biased_df, y_train_biased, a_train,
                train_size=train_size, random_state=self.seed, stratify=y_train_biased
            )

        # 5) Fit encoder + scaler on θ_train only; transform val/test (no leakage)
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

        scaler = StandardScaler()

        X_train_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_biased_df), 0))
        X_train_num = scaler.fit_transform(X_train_biased_df[num_cols])   if len(num_cols) else np.empty((len(X_train_biased_df), 0))
        X_train_all = np.hstack([X_train_num, X_train_cat])

        X_val_cat  = encoder.transform(X_val_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_val_biased_df), 0))
        X_val_num  = scaler.transform(X_val_biased_df[num_cols])  if len(num_cols) else np.empty((len(X_val_biased_df), 0))
        X_val_all  = np.hstack([X_val_num, X_val_cat])

        X_test_cat = encoder.transform(X_test_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_test_biased_df), 0))
        X_test_num = scaler.transform(X_test_biased_df[num_cols])  if len(num_cols) else np.empty((len(X_test_biased_df), 0))
        X_test_all = np.hstack([X_test_num, X_test_cat])

        # 6) Map to θ-space (PCA if enabled; otherwise identity)
        X_train_theta_np, X_val_theta_np, X_test_theta_np, pca = self._to_theta(
            X_train_all, X_val_all, X_test_all,
            pca_components=pca_components
        )

        # 7) Convert to torch tensors on device
        X_train_theta = torch.tensor(X_train_theta_np, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_val_theta_np,   dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_theta_np,  dtype=torch.float32, device=self.device)


        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_biased,   dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_biased,  dtype=torch.long, device=self.device)

        # Store protected attributes for fairness metrics
        self.a_train = torch.tensor(a_train, dtype=torch.long, device=self.device)
        self.a_val   = torch.tensor(a_val,   dtype=torch.long, device=self.device)
        self.a_test  = torch.tensor(a_test,  dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col

        # ---- Sanity check logging ----
        def log_distribution(name, y_split):
            n_total = len(y_split)
            n_min = np.sum(y_split == 1)
            pct_min = 100.0 * n_min / n_total if n_total > 0 else 0.0
            print(f"[{name}] size={n_total}, minority={n_min} ({pct_min:.2f}%)")

        log_distribution("TRAIN", y_train_biased)
        log_distribution("VAL",   y_val_biased)
        log_distribution("TEST",  y_test_biased)

        # Cache GAN view for apples-to-apples CTGAN/CTAB baselines
        try:
            self._gan_view_cache = {
                "supported": True,
                "X_train_unbiased_df": X_train_df.copy(),
                "y_train_unbiased": y_train.astype(int).copy(),
                "encoder": encoder,
                "scaler": scaler,
                "pca": pca,                  # None if use_pca=False
                "use_pca": self.use_pca,
                "pca_components": int(pca_components),
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
                "drop_protected": bool(drop_protected),
                "protected_cols": list(protected_cols),
            }
        except Exception:
            pass

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


    def split_capture24(
        self,
        train_size=None,
        val_size=None,
        da_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=10,
        dp_protected_col: str = "sex",
    ):
        """
        CAPTURE-24 wrist accelerometer dataset: MVPA (y=1) vs non-MVPA (y=0).
        Protected attribute: sex (0=male, 1=female).

        Requires the feature cache built by:
            python scripts/download_capture24.py --data-dir datasets/capture24

        The cache contains per-window features (32-dim: mean/std/min/max/rms/p25/p75/iqr
        for each of x, y, z, vector-magnitude axes over 5 s windows at 100 Hz).

        Split is subject-level (no window leakage between train/val/test).
        """
        data_dir   = Path(self.data_path)
        cache_path = data_dir / "capture24_features_cache.npz"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"CAPTURE-24 feature cache not found at {cache_path}.\n"
                "Run: python scripts/download_capture24.py --data-dir datasets/capture24"
            )

        cache   = np.load(cache_path, allow_pickle=True)
        X_all   = cache["X"]                        # (N, 32)
        y_all   = cache["y"].astype(int)
        a_all   = cache["a"].astype(int)            # 0=male, 1=female
        sid_all = cache["subject_ids"].astype(int)  # 0..n_subjects-1

        n_subjects = int(sid_all.max()) + 1
        print(f"[capture24] Cache: {len(X_all):,} windows, {n_subjects} participants")
        print(f"[capture24] MVPA={int((y_all==1).sum()):,}  "
              f"({100.*(y_all==1).mean():.1f}%),  "
              f"female={int((a_all==1).sum()):,}  "
              f"({100.*(a_all==1).mean():.1f}%)")

        # Feature names (32 accelerometer stats + sex as input feature)
        feature_names = [
            f"{ax}_{stat}"
            for ax in ("x", "y", "z", "vm")
            for stat in ("mean", "std", "min", "max", "rms", "p25", "p75", "iqr")
        ] + ["sex"]

        # ---- Subject-level train / val / test split -------------------------
        # Stratify by sex so each split contains both groups.
        rng = np.random.default_rng(self.seed)

        subject_sex = np.array([
            int(np.bincount(a_all[sid_all == s]).argmax())
            for s in range(n_subjects)
        ])
        female_subs = np.where(subject_sex == 1)[0]
        male_subs   = np.where(subject_sex == 0)[0]
        rng.shuffle(female_subs)
        rng.shuffle(male_subs)

        def _split_subs(subs):
            n     = len(subs)
            n_te  = max(1, round(n * test_frac))
            n_va  = max(1, round(n * val_frac))
            n_tr  = n - n_te - n_va
            return subs[:n_tr], subs[n_tr:n_tr + n_va], subs[n_tr + n_va:]

        f_tr, f_va, f_te = _split_subs(female_subs)
        m_tr, m_va, m_te = _split_subs(male_subs)

        train_subs = np.concatenate([f_tr, m_tr])
        val_subs   = np.concatenate([f_va, m_va])
        test_subs  = np.concatenate([f_te, m_te])

        tr_mask = np.isin(sid_all, train_subs)
        va_mask = np.isin(sid_all, val_subs)
        te_mask = np.isin(sid_all, test_subs)

        X_train_df = pd.DataFrame(
            np.column_stack([X_all[tr_mask], a_all[tr_mask]]), columns=feature_names)
        X_val_df   = pd.DataFrame(
            np.column_stack([X_all[va_mask], a_all[va_mask]]), columns=feature_names)
        X_test_df  = pd.DataFrame(
            np.column_stack([X_all[te_mask], a_all[te_mask]]), columns=feature_names)

        y_train = y_all[tr_mask]
        y_val   = y_all[va_mask]
        y_test  = y_all[te_mask]

        A_train = pd.Series(a_all[tr_mask], name="sex")
        A_val   = pd.Series(a_all[va_mask], name="sex")
        A_test  = pd.Series(a_all[te_mask], name="sex")

        # ---- Bias injection (DA+ targeting) ---------------------------------
        # Targets exactly round(da_pct * n_total) female (a=1) positives.
        # Keeps all male positives and all negatives. Fills remainder from non-disadv-pos pool.
        def apply_da_pct_bias_c24(df_split, y_split, a_split, da_pct, n_total):
            df = df_split.copy()
            df["__y__"] = y_split
            df["__a__"] = a_split.to_numpy(dtype=np.int64) if hasattr(a_split, "to_numpy") else np.array(a_split, dtype=np.int64)

            target_da_plus = max(1, round(da_pct * n_total))

            disadv_pos_mask = (df["__a__"] == 1) & (df["__y__"] == 1)  # female MVPA
            disadv_pos = df[disadv_pos_mask]
            rest       = df[~disadv_pos_mask]

            keep_n = min(len(disadv_pos), target_da_plus)
            if keep_n < target_da_plus:
                print(f"  [da_pct WARNING] Only {len(disadv_pos)} disadvantaged positives "
                      f"available; target was {target_da_plus}.")
            disadv_pos_kept = disadv_pos.sample(n=keep_n, random_state=self.seed, replace=False)

            n_rest_needed = n_total - keep_n
            if len(rest) >= n_rest_needed:
                rest_kept = rest.sample(n=n_rest_needed, random_state=self.seed, replace=False)
            else:
                rest_kept = rest

            result = (pd.concat([disadv_pos_kept, rest_kept], axis=0)
                      .sample(frac=1.0, random_state=self.seed)
                      .reset_index(drop=True))

            y_out = result["__y__"].to_numpy(dtype=int)
            a_out = result["__a__"].to_numpy(dtype=np.int64)
            X_out = result.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        if da_pct is not None:
            n_total = train_size if train_size is not None else len(X_train_df)
            X_train_b, y_train_b, a_train = apply_da_pct_bias_c24(
                X_train_df, y_train, A_train, da_pct, n_total)
        else:
            X_train_b = X_train_df.copy().reset_index(drop=True)
            y_train_b = y_train.copy()
            a_train   = A_train.to_numpy(dtype=np.int64)

        X_val_b = X_val_df.copy().reset_index(drop=True)
        y_val_b = y_val.copy()
        a_val   = A_val.to_numpy(dtype=np.int64)

        X_test_b = X_test_df.copy().reset_index(drop=True)
        y_test_b = y_test.copy()
        a_test   = A_test.to_numpy(dtype=np.int64)

        if da_pct is None and train_size is not None and train_size < len(X_train_b):
            X_train_b, _, y_train_b, _, a_train, _ = train_test_split(
                X_train_b, y_train_b, a_train,
                train_size=train_size, random_state=self.seed, stratify=y_train_b,
            )

        # ---- Optional: subsample val (keeps val set comparable to train) ----
        # If val_size not given but train_size is, auto-cap val at train_size
        # so the reward signal is not dominated by millions of unbiased windows.
        _effective_val_size = val_size or (train_size if train_size is not None else None)
        if _effective_val_size is not None and _effective_val_size < len(X_val_b):
            X_val_b, _, y_val_b, _, a_val, _ = train_test_split(
                X_val_b, y_val_b, a_val,
                train_size=_effective_val_size, random_state=self.seed, stratify=y_val_b,
            )

        # ---- Scaler (fit on train only) -------------------------------------
        scaler = StandardScaler()
        Xtr_z  = scaler.fit_transform(X_train_b.values)
        Xva_z  = scaler.transform(X_val_b.values)
        Xte_z  = scaler.transform(X_test_b.values)

        # ---- PCA (optional) -------------------------------------------------
        X_tr_th, X_va_th, X_te_th, pca = self._to_theta(
            Xtr_z, Xva_z, Xte_z, pca_components=pca_components
        )

        # ---- Tensors --------------------------------------------------------
        X_train_theta = torch.tensor(X_tr_th, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_va_th, dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_te_th, dtype=torch.float32, device=self.device)

        y_train_theta = torch.tensor(y_train_b, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_b,   dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_b,  dtype=torch.long, device=self.device)

        self.a_train        = torch.tensor(a_train, dtype=torch.long, device=self.device)
        self.a_val          = torch.tensor(a_val,   dtype=torch.long, device=self.device)
        self.a_test         = torch.tensor(a_test,  dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col

        # ---- Logging --------------------------------------------------------
        def _log(tag, ys, as_):
            n = len(ys)
            n1 = int((ys == 1).sum())
            nf = int((as_ == 1).sum())
            nfp = int(((ys == 1) & (as_ == 1)).sum())
            nmp = int(((ys == 1) & (as_ == 0)).sum())
            print(f"[{tag}] size={n:,}, MVPA={n1:,} ({100.*n1/n:.1f}%), "
                  f"female={nf:,}, female_MVPA={nfp:,}, male_MVPA={nmp:,}")

        _log("TRAIN", y_train_b, a_train)
        _log("VAL",   y_val_b,   a_val)
        _log("TEST",  y_test_b,  a_test)
        print(f"use_pca={self.use_pca} | theta_dim={X_train_theta.shape[1]}")

        # ---- GAN view (for CTGAN/CTAB-GAN baselines) ------------------------
        try:
            self._gan_view_cache = {
                "supported":           True,
                "X_train_unbiased_df": X_train_df.copy(),
                "y_train_unbiased":    y_train.astype(int).copy(),
                "encoder":             None,
                "scaler":              scaler,
                "pca":                 pca,
                "use_pca":             self.use_pca,
                "pca_components":      int(pca_components),
                "cat_cols":            [],
                "num_cols":            feature_names,
            }
        except Exception:
            pass

        return (X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta)

    # ------------------------------------------------------------------
    # Subject-level k-fold helpers for capture24
    # ------------------------------------------------------------------

    @staticmethod
    def _c24_kfold_assignments(cache, k: int, fold_rng_seed=None):
        """
        Build a stratified subject-level fold assignment.

        Subjects are split into female and male pools.

        fold_rng_seed=None (default): sort each pool ascending by MVPA rate,
        then round-robin into folds (the original deterministic assignment).

        fold_rng_seed=<int>: shuffle each pool with that seed, then
        round-robin. Use to find assignments where all folds have a viable
        α-EO signal and WGL correctly targets the female group.

        The assignment is reproducible given the same seed so every method —
        FORGE, alpha, and all baselines — sees the identical partitioning.
        """
        y_all   = cache["y"].astype(int)
        a_all   = cache["a"].astype(int)
        sid_all = cache["subject_ids"].astype(int)
        n_subjects = int(sid_all.max()) + 1

        subject_sex  = np.array([int(np.bincount(a_all[sid_all == s]).argmax())
                                  for s in range(n_subjects)])
        subject_mvpa = np.array([(y_all[sid_all == s] == 1).mean()
                                  for s in range(n_subjects)])

        female_subs = np.where(subject_sex == 1)[0]
        male_subs   = np.where(subject_sex == 0)[0]

        if fold_rng_seed is None:
            # Original: sort ascending by MVPA rate
            f_ordered = female_subs[np.argsort(subject_mvpa[female_subs])]
            m_ordered = male_subs[np.argsort(subject_mvpa[male_subs])]
        else:
            rng = np.random.RandomState(fold_rng_seed)
            f_ordered = female_subs[rng.permutation(len(female_subs))]
            m_ordered = male_subs[rng.permutation(len(male_subs))]

        folds = [[] for _ in range(k)]
        for i, s in enumerate(f_ordered):
            folds[i % k].append(int(s))
        for i, s in enumerate(m_ordered):
            folds[i % k].append(int(s))

        return folds  # list of k lists

    def split_capture24_kfold(
        self,
        fold_idx: int,
        n_folds: int = 5,
        train_size=None,
        da_pct=None,
        pca_components: int = 10,
        dp_protected_col: str = "sex",
        kfold_val_frac: float = 0.4,
        fold_rng_seed=None,
    ):
        """
        Subject-level k-fold split for capture24.

        Fold assignment is deterministic (stratified by per-subject female-MVPA
        rate, round-robin). For each rotation:
          test   = last (1 - kfold_val_frac) of held-out subjects' windows
          val    = first kfold_val_frac of the same held-out subjects' windows
          train  = all remaining subjects

        Val and test share the same subjects (stratified-random split within each
        held-out subject, stratified by y×a). This ensures val and test have the
        same female MVPA rate, so val α-EO ≈ test α-EO and the RL reward signal
        targets the same fairness problem as the final evaluation. Stratified
        random splitting also preserves the full temporal distribution in both
        splits, avoiding day-of-study drift that a temporal split would introduce.

        Scarcity injection (da_pct) is applied to training windows only.
        Val is capped at train_size windows so the reward signal is not
        dominated by large unbiased windows.
        Test is NOT subsampled — full subject windows for stable EO estimation.
        """
        data_dir   = Path(self.data_path)
        cache_path = data_dir / "capture24_features_cache.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"CAPTURE-24 feature cache not found at {cache_path}.\n"
                "Run: python scripts/download_capture24.py --data-dir datasets/capture24"
            )

        cache   = np.load(cache_path, allow_pickle=True)
        X_all   = cache["X"]
        y_all   = cache["y"].astype(int)
        a_all   = cache["a"].astype(int)
        sid_all = cache["subject_ids"].astype(int)

        n_subjects = int(sid_all.max()) + 1
        print(f"[capture24-kfold] Cache: {len(X_all):,} windows, {n_subjects} subjects, "
              f"fold {fold_idx}/{n_folds}")

        folds = Dataset._c24_kfold_assignments(cache, n_folds, fold_rng_seed=fold_rng_seed)

        held_out_subs = np.array(folds[fold_idx])
        train_subs    = np.concatenate([folds[f] for f in range(n_folds) if f != fold_idx])

        # Val and test come from the same held-out subjects, split by stratified
        # random sampling (stratified by y×a). This ensures val and test have
        # the same female MVPA rate, so val α-EO ≈ test α-EO, making the RL
        # reward signal and final evaluation target the same fairness problem.
        # Stratified random sampling also preserves the full temporal distribution
        # in both splits (no day-of-study drift).
        va_mask = np.zeros(len(sid_all), dtype=bool)
        te_mask = np.zeros(len(sid_all), dtype=bool)
        for sub in held_out_subs:
            sub_idx = np.where(sid_all == sub)[0]
            y_sub = y_all[sub_idx]
            a_sub = a_all[sub_idx]
            strat = y_sub * 2 + a_sub  # 4 strata: (y=0/1) × (a=0/1)
            n_val = max(1, int(len(sub_idx) * kfold_val_frac))
            try:
                idx_val, idx_te = train_test_split(
                    sub_idx, train_size=n_val,
                    stratify=strat, random_state=self.seed,
                )
            except ValueError:
                # Fallback for subjects with too few samples per stratum
                idx_val = sub_idx[:n_val]
                idx_te  = sub_idx[n_val:]
            va_mask[idx_val] = True
            te_mask[idx_te]  = True

        tr_mask = np.isin(sid_all, train_subs)

        feature_names = [
            f"{ax}_{stat}"
            for ax in ("x", "y", "z", "vm")
            for stat in ("mean", "std", "min", "max", "rms", "p25", "p75", "iqr")
        ] + ["sex"]

        X_train_df = pd.DataFrame(
            np.column_stack([X_all[tr_mask], a_all[tr_mask]]), columns=feature_names)
        X_val_df   = pd.DataFrame(
            np.column_stack([X_all[va_mask], a_all[va_mask]]), columns=feature_names)
        X_test_df  = pd.DataFrame(
            np.column_stack([X_all[te_mask], a_all[te_mask]]), columns=feature_names)

        y_train = y_all[tr_mask]
        y_val   = y_all[va_mask]
        y_test  = y_all[te_mask]

        A_train = pd.Series(a_all[tr_mask], name="sex")
        A_val   = pd.Series(a_all[va_mask], name="sex")
        A_test  = pd.Series(a_all[te_mask], name="sex")

        # Bias injection: training only
        def apply_da_pct_bias_c24(df_split, y_split, a_split, da_pct, n_total):
            df = df_split.copy()
            df["__y__"] = y_split
            df["__a__"] = (a_split.to_numpy(dtype=np.int64)
                           if hasattr(a_split, "to_numpy") else np.array(a_split, dtype=np.int64))
            target_da_plus = max(1, round(da_pct * n_total))
            disadv_pos_mask = (df["__a__"] == 1) & (df["__y__"] == 1)
            disadv_pos = df[disadv_pos_mask]
            rest       = df[~disadv_pos_mask]
            keep_n = min(len(disadv_pos), target_da_plus)
            if keep_n < target_da_plus:
                print(f"  [da_pct WARNING] Only {len(disadv_pos)} disadvantaged positives "
                      f"available; target was {target_da_plus}.")
            disadv_pos_kept = disadv_pos.sample(n=keep_n, random_state=self.seed, replace=False)
            n_rest_needed = n_total - keep_n
            rest_kept = (rest.sample(n=n_rest_needed, random_state=self.seed, replace=False)
                         if len(rest) >= n_rest_needed else rest)
            result = (pd.concat([disadv_pos_kept, rest_kept])
                      .sample(frac=1.0, random_state=self.seed)
                      .reset_index(drop=True))
            y_out = result["__y__"].to_numpy(dtype=int)
            a_out = result["__a__"].to_numpy(dtype=np.int64)
            return result.drop(columns=["__y__", "__a__"]), y_out, a_out

        if da_pct is not None:
            n_total = train_size if train_size is not None else len(X_train_df)
            X_train_b, y_train_b, a_train = apply_da_pct_bias_c24(
                X_train_df, y_train, A_train, da_pct, n_total)
        else:
            X_train_b = X_train_df.copy().reset_index(drop=True)
            y_train_b = y_train.copy()
            a_train   = A_train.to_numpy(dtype=np.int64)

        # Val: unbiased, capped at train_size for stable reward
        X_val_b  = X_val_df.copy().reset_index(drop=True)
        y_val_b  = y_val.copy()
        a_val    = A_val.to_numpy(dtype=np.int64)
        _val_cap = train_size if train_size is not None else None
        if _val_cap is not None and _val_cap < len(X_val_b):
            strat_va = y_val_b * 2 + a_val  # preserve female MVPA rate
            X_val_b, _, y_val_b, _, a_val, _ = train_test_split(
                X_val_b, y_val_b, a_val,
                train_size=_val_cap, random_state=self.seed, stratify=strat_va,
            )

        # Test: full natural distribution (no subsampling)
        X_test_b = X_test_df.copy().reset_index(drop=True)
        y_test_b = y_test.copy()
        a_test   = A_test.to_numpy(dtype=np.int64)

        # Scaler fit on train only
        scaler = StandardScaler()
        Xtr_z  = scaler.fit_transform(X_train_b.values)
        Xva_z  = scaler.transform(X_val_b.values)
        Xte_z  = scaler.transform(X_test_b.values)

        X_tr_th, X_va_th, X_te_th, pca = self._to_theta(
            Xtr_z, Xva_z, Xte_z, pca_components=pca_components
        )

        X_train_theta = torch.tensor(X_tr_th, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_va_th, dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_te_th, dtype=torch.float32, device=self.device)
        y_train_theta = torch.tensor(y_train_b, dtype=torch.long,  device=self.device)
        y_val_theta   = torch.tensor(y_val_b,   dtype=torch.long,  device=self.device)
        y_test_theta  = torch.tensor(y_test_b,  dtype=torch.long,  device=self.device)

        self.a_train          = torch.tensor(a_train, dtype=torch.long, device=self.device)
        self.a_val            = torch.tensor(a_val,   dtype=torch.long, device=self.device)
        self.a_test           = torch.tensor(a_test,  dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col
        self.pca_transform    = pca

        def _log(tag, ys, as_):
            n = len(ys); n1 = int((ys == 1).sum())
            nf = int((as_ == 1).sum())
            nfp = int(((ys == 1) & (as_ == 1)).sum())
            nmp = int(((ys == 1) & (as_ == 0)).sum())
            print(f"[{tag}] size={n:,}, MVPA={n1:,} ({100.*n1/n:.1f}%), "
                  f"female={nf:,}, female_MVPA={nfp:,}, male_MVPA={nmp:,}")

        _log("TRAIN", y_train_b, a_train)
        _log("VAL",   y_val_b,   a_val)
        _log("TEST",  y_test_b,  a_test)
        print(f"fold_idx={fold_idx} | held_out_subs={list(held_out_subs)} "
              f"| theta_dim={X_train_theta.shape[1]}")

        # GAN view for CTGAN/FairTabDDPM baselines
        try:
            self._gan_view_cache = {
                "supported":           True,
                "X_train_unbiased_df": X_train_df.copy(),
                "y_train_unbiased":    y_train.astype(int).copy(),
                "encoder":             None,
                "scaler":              scaler,
                "pca":                 pca,
                "use_pca":             self.use_pca,
                "pca_components":      int(pca_components),
                "cat_cols":            [],
                "num_cols":            feature_names,
            }
        except Exception:
            pass

        return (X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta)

    def get_data_splits(self, **kwargs):
        _tabular_drop = ("win_seconds", "step_seconds")
        if self.dataset_name == "census_income":
            kw = {k: v for k, v in kwargs.items() if k not in _tabular_drop}
            return self.split_census_income(**kw)
        elif self.dataset_name == "capture24":
            c24_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ("drop_protected", "protected_cols",
                                       "win_seconds", "step_seconds")}
            if "fold_idx" in c24_kwargs:
                kfold_keys = ("fold_idx", "n_folds", "train_size", "da_pct",
                              "pca_components", "dp_protected_col", "kfold_val_frac",
                              "fold_rng_seed")
                kfold_kwargs = {k: v for k, v in c24_kwargs.items() if k in kfold_keys}
                return self.split_capture24_kfold(**kfold_kwargs)
            return self.split_capture24(**c24_kwargs)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

