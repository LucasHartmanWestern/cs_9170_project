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
            "protected_attributes": ["sex", "race", "age", "native-country"],
        },
        "capture24": {
            "data_path": "datasets/capture24",
            "protected_attributes": ["sex"],
        },
    }

    _C24_FEATURE_NAMES = [
        f"{ax}_{stat}"
        for ax in ("x", "y", "z", "vm")
        for stat in ("mean", "std", "min", "max", "rms", "p25", "p75", "iqr")
    ] + ["sex"]

    def __init__(self, dataset_name, minority_id, majority_id,
                 pca_components=2, seed=42, device="cpu", use_pca: bool = False):
        self.dataset_name = dataset_name
        self.minority_id = minority_id
        self.majority_id = majority_id
        self.pca_components = pca_components
        self.seed = seed
        self.device = device
        self.data_path = self.DATASET_REGISTRY[dataset_name]["data_path"]
        self.protected_attributes = self.DATASET_REGISTRY[dataset_name].get("protected_attributes", [])
        self.use_pca = bool(use_pca)

    # ------------------------------------------------------------------ helpers

    def _make_pca(self, n_components: int) -> PCA:
        return PCA(n_components=n_components, svd_solver="full", random_state=self.seed)

    def _apply_pca(self, X_tr, X_va, X_te, pca_components: int):
        """Fit PCA on train, transform all splits. No-op if use_pca=False."""
        if not self.use_pca:
            return X_tr, X_va, X_te, None
        k = min(int(pca_components), X_tr.shape[1])
        pca = self._make_pca(k)
        X_tr_pca = pca.fit_transform(X_tr)
        X_va_pca = pca.transform(X_va)
        X_te_pca = pca.transform(X_te)
        self.pca_transform = pca
        return X_tr_pca, X_va_pca, X_te_pca, pca

    def _to_tensors(self, X_tr, X_va, X_te, y_tr, y_va, y_te, a_tr, a_va, a_te):
        """Convert numpy arrays to torch tensors and store protected attributes."""
        self.a_train = torch.tensor(a_tr, dtype=torch.long, device=self.device)
        self.a_val   = torch.tensor(a_va, dtype=torch.long, device=self.device)
        self.a_test  = torch.tensor(a_te, dtype=torch.long, device=self.device)
        return (
            torch.tensor(X_tr, dtype=torch.float32, device=self.device),
            torch.tensor(X_va, dtype=torch.float32, device=self.device),
            torch.tensor(X_te, dtype=torch.float32, device=self.device),
            torch.tensor(y_tr, dtype=torch.long, device=self.device),
            torch.tensor(y_va, dtype=torch.long, device=self.device),
            torch.tensor(y_te, dtype=torch.long, device=self.device),
        )

    @staticmethod
    def _inject_da_pct_bias(df, y, a, disadv_id, da_pct, n_total, seed):
        """
        Subsample disadvantaged-group positives to exactly round(da_pct * n_total).
        All other rows (advantaged positives + all negatives) are kept and filled
        up to n_total. Val/test are never passed here.
        """
        df = df.copy()
        df["__y__"] = y
        df["__a__"] = np.asarray(a, dtype=np.int64)

        target = max(1, round(da_pct * n_total))
        disadv_mask = (df["__a__"] == disadv_id) & (df["__y__"] == 1)
        disadv = df[disadv_mask]
        rest   = df[~disadv_mask]

        keep = min(len(disadv), target)
        if keep < target:
            print(f"  [da_pct WARNING] Only {len(disadv)} disadvantaged positives available; "
                  f"target was {target}.")
        kept = disadv.sample(n=keep, random_state=seed, replace=False)

        n_rest = n_total - keep
        rest_kept = (rest.sample(n=n_rest, random_state=seed, replace=False)
                     if len(rest) >= n_rest else rest)

        result = (pd.concat([kept, rest_kept])
                  .sample(frac=1.0, random_state=seed)
                  .reset_index(drop=True))

        y_out = result["__y__"].to_numpy(dtype=int)
        a_out = result["__a__"].to_numpy(dtype=np.int64)
        return result.drop(columns=["__y__", "__a__"]), y_out, a_out

    @staticmethod
    def _log_split(tag, y, a):
        n = len(y); n1 = int((y == 1).sum())
        nf = int((a == 1).sum())
        print(f"[{tag}] n={n:,}  pos={n1:,} ({100.*n1/n:.1f}%)  "
              f"female={nf:,}  f_pos={int(((y==1)&(a==1)).sum()):,}  "
              f"m_pos={int(((y==1)&(a==0)).sum()):,}")

    # ---------------------------------------------------------- census_income

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
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1

        if protected_cols is None:
            protected_cols = self.protected_attributes

        # Load
        col_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income",
        ]
        df = pd.read_csv(self.data_path, header=None, names=col_names,
                         na_values="?", skipinitialspace=True).dropna()
        y = np.where(df["income"].isin([">50K", ">50K."]), 1, 0).astype(int)
        df = df.drop(columns=["income"])

        A = df[[dp_protected_col]].copy()

        if drop_protected:
            df = df.drop(columns=[c for c in protected_cols if c in df.columns])

        cat_cols = [c for c in df.columns if df[c].dtype.name in ("category", "object", "bool")]
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

        # Stratified train / val / test split
        X_tr, X_tmp, y_tr, y_tmp, A_tr, A_tmp = train_test_split(
            df, y, A, test_size=(val_frac + test_frac), random_state=self.seed, stratify=y)
        X_va, X_te, y_va, y_te, A_va, A_te = train_test_split(
            X_tmp, y_tmp, A_tmp, test_size=test_frac / (val_frac + test_frac),
            random_state=self.seed, stratify=y_tmp)

        def _map_sex(s):  # "Male"/"Female" → 1/0
            return (s.str.strip().str.lower() == "male").astype(np.int64).to_numpy()

        a_tr = _map_sex(A_tr[dp_protected_col])
        a_va = _map_sex(A_va[dp_protected_col])
        a_te = _map_sex(A_te[dp_protected_col])

        # DA+ scarcity injection on train only (census: female = disadv_id=0)
        if da_pct is not None:
            n_total = train_size if train_size is not None else len(X_tr)
            X_tr, y_tr, a_tr = self._inject_da_pct_bias(X_tr, y_tr, a_tr, 0, da_pct, n_total, self.seed)
        elif train_size is not None and train_size < len(X_tr):
            X_tr, _, y_tr, _, a_tr, _ = train_test_split(
                X_tr, y_tr, a_tr, train_size=train_size, random_state=self.seed, stratify=y_tr)

        # Encode + scale (fit on train only)
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        scaler = StandardScaler()

        def _transform(X, fit=False):
            cat = enc.fit_transform(X[cat_cols]) if fit else enc.transform(X[cat_cols])
            num = scaler.fit_transform(X[num_cols]) if fit else scaler.transform(X[num_cols])
            cat = cat if len(cat_cols) else np.empty((len(X), 0))
            num = num if len(num_cols) else np.empty((len(X), 0))
            return np.hstack([num, cat])

        Xtr_enc = _transform(X_tr, fit=True)
        Xva_enc = _transform(X_va)
        Xte_enc = _transform(X_te)

        Xtr_f, Xva_f, Xte_f, _ = self._apply_pca(Xtr_enc, Xva_enc, Xte_enc, pca_components)

        self.dp_protected_col = dp_protected_col
        for tag, yi, ai in [("TRAIN", y_tr, a_tr), ("VAL", y_va, a_va), ("TEST", y_te, a_te)]:
            self._log_split(tag, yi, ai)

        return self._to_tensors(Xtr_f, Xva_f, Xte_f, y_tr, y_va, y_te, a_tr, a_va, a_te)

    # ---------------------------------------------------------- capture24

    def _load_c24_cache(self):
        cache_path = Path(self.data_path) / "capture24_features_cache.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"CAPTURE-24 feature cache not found at {cache_path}.\n"
                "Run: python scripts/download_capture24.py --data-dir datasets/capture24"
            )
        return np.load(cache_path, allow_pickle=True)

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
        cache   = self._load_c24_cache()
        X_all   = cache["X"]
        y_all   = cache["y"].astype(int)
        a_all   = cache["a"].astype(int)
        sid_all = cache["subject_ids"].astype(int)

        n_subjects = int(sid_all.max()) + 1
        print(f"[capture24] {len(X_all):,} windows, {n_subjects} subjects  "
              f"MVPA={int((y_all==1).sum()):,} ({100.*(y_all==1).mean():.1f}%)  "
              f"female={int((a_all==1).sum()):,} ({100.*(a_all==1).mean():.1f}%)")

        # Subject-level split stratified by sex
        rng = np.random.default_rng(self.seed)
        subject_sex = np.array([int(np.bincount(a_all[sid_all == s]).argmax())
                                 for s in range(n_subjects)])
        f_subs = np.where(subject_sex == 1)[0]; rng.shuffle(f_subs)
        m_subs = np.where(subject_sex == 0)[0]; rng.shuffle(m_subs)

        def _split(subs):
            n_te = max(1, round(len(subs) * test_frac))
            n_va = max(1, round(len(subs) * val_frac))
            return subs[:len(subs)-n_te-n_va], subs[len(subs)-n_te-n_va:len(subs)-n_te], subs[len(subs)-n_te:]

        f_tr, f_va, f_te = _split(f_subs)
        m_tr, m_va, m_te = _split(m_subs)
        tr_mask = np.isin(sid_all, np.concatenate([f_tr, m_tr]))
        va_mask = np.isin(sid_all, np.concatenate([f_va, m_va]))
        te_mask = np.isin(sid_all, np.concatenate([f_te, m_te]))

        fn = self._C24_FEATURE_NAMES

        def _make_df(mask):
            return pd.DataFrame(np.column_stack([X_all[mask], a_all[mask]]), columns=fn)

        X_tr_df = _make_df(tr_mask); y_tr = y_all[tr_mask]; a_tr_s = pd.Series(a_all[tr_mask])
        X_va_df = _make_df(va_mask); y_va = y_all[va_mask]; a_va_s = pd.Series(a_all[va_mask])
        X_te_df = _make_df(te_mask); y_te = y_all[te_mask]; a_te   = a_all[te_mask].astype(np.int64)

        # DA+ injection on train only (capture24: female = disadv_id=1)
        if da_pct is not None:
            n_total = train_size if train_size is not None else len(X_tr_df)
            X_tr_df, y_tr, a_tr = self._inject_da_pct_bias(
                X_tr_df, y_tr, a_tr_s.to_numpy(dtype=np.int64), 1, da_pct, n_total, self.seed)
        else:
            a_tr = a_tr_s.to_numpy(dtype=np.int64)
            if train_size is not None and train_size < len(X_tr_df):
                X_tr_df, _, y_tr, _, a_tr, _ = train_test_split(
                    X_tr_df, y_tr, a_tr, train_size=train_size,
                    random_state=self.seed, stratify=y_tr)

        a_va = a_va_s.to_numpy(dtype=np.int64)

        # Cap val at train_size (keeps reward signal comparable to training set)
        val_cap = val_size or train_size
        if val_cap is not None and val_cap < len(X_va_df):
            X_va_df, _, y_va, _, a_va, _ = train_test_split(
                X_va_df, y_va, a_va, train_size=val_cap,
                random_state=self.seed, stratify=y_va)

        scaler = StandardScaler()
        Xtr_z = scaler.fit_transform(X_tr_df.values)
        Xva_z = scaler.transform(X_va_df.values)
        Xte_z = scaler.transform(X_te_df.values)

        Xtr_f, Xva_f, Xte_f, _ = self._apply_pca(Xtr_z, Xva_z, Xte_z, pca_components)

        self.dp_protected_col = dp_protected_col
        for tag, yi, ai in [("TRAIN", y_tr, a_tr), ("VAL", y_va, a_va), ("TEST", y_te, a_te)]:
            self._log_split(tag, yi, ai)

        return self._to_tensors(Xtr_f, Xva_f, Xte_f, y_tr, y_va, y_te, a_tr, a_va, a_te)

    # -------------------------------------------------- capture24 k-fold

    @staticmethod
    def _c24_kfold_assignments(cache, k: int, fold_rng_seed=None):
        """
        Stratified subject-level fold assignment.

        fold_rng_seed=None: sort each sex pool by MVPA rate then round-robin.
        fold_rng_seed=<int>: shuffle each pool with that seed then round-robin.
        Reproducible given the same seed so all methods see identical partitioning.
        """
        y_all   = cache["y"].astype(int)
        a_all   = cache["a"].astype(int)
        sid_all = cache["subject_ids"].astype(int)
        n_subjects = int(sid_all.max()) + 1

        subject_sex  = np.array([int(np.bincount(a_all[sid_all == s]).argmax())
                                  for s in range(n_subjects)])
        subject_mvpa = np.array([(y_all[sid_all == s] == 1).mean()
                                  for s in range(n_subjects)])

        f_subs = np.where(subject_sex == 1)[0]
        m_subs = np.where(subject_sex == 0)[0]

        if fold_rng_seed is None:
            f_ord = f_subs[np.argsort(subject_mvpa[f_subs])]
            m_ord = m_subs[np.argsort(subject_mvpa[m_subs])]
        else:
            rng   = np.random.RandomState(fold_rng_seed)
            f_ord = f_subs[rng.permutation(len(f_subs))]
            m_ord = m_subs[rng.permutation(len(m_subs))]

        folds = [[] for _ in range(k)]
        for i, s in enumerate(f_ord): folds[i % k].append(int(s))
        for i, s in enumerate(m_ord): folds[i % k].append(int(s))
        return folds

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
        Subject-level k-fold for capture24.

        Held-out subjects are split into val (kfold_val_frac) and test
        (1 - kfold_val_frac) by stratified random sampling within each subject
        (strata = y×a), so val and test share the same female MVPA rate and
        the reward signal targets the same fairness problem as final evaluation.

        Val is capped at train_size; test is never subsampled.
        """
        cache   = self._load_c24_cache()
        X_all   = cache["X"]
        y_all   = cache["y"].astype(int)
        a_all   = cache["a"].astype(int)
        sid_all = cache["subject_ids"].astype(int)

        print(f"[capture24-kfold] {len(X_all):,} windows, fold {fold_idx}/{n_folds}")

        folds         = Dataset._c24_kfold_assignments(cache, n_folds, fold_rng_seed=fold_rng_seed)
        held_out_subs = np.array(folds[fold_idx])
        train_subs    = np.concatenate([folds[f] for f in range(n_folds) if f != fold_idx])

        va_mask = np.zeros(len(sid_all), dtype=bool)
        te_mask = np.zeros(len(sid_all), dtype=bool)
        for sub in held_out_subs:
            idx   = np.where(sid_all == sub)[0]
            strat = y_all[idx] * 2 + a_all[idx]
            n_val = max(1, int(len(idx) * kfold_val_frac))
            try:
                idx_va, idx_te = train_test_split(
                    idx, train_size=n_val, stratify=strat, random_state=self.seed)
            except ValueError:
                idx_va, idx_te = idx[:n_val], idx[n_val:]
            va_mask[idx_va] = True
            te_mask[idx_te] = True

        tr_mask = np.isin(sid_all, train_subs)
        fn      = self._C24_FEATURE_NAMES

        def _make_df(mask):
            return pd.DataFrame(np.column_stack([X_all[mask], a_all[mask]]), columns=fn)

        X_tr_df = _make_df(tr_mask); y_tr = y_all[tr_mask]; a_tr_s = pd.Series(a_all[tr_mask])
        X_va_df = _make_df(va_mask); y_va = y_all[va_mask]; a_va_s = pd.Series(a_all[va_mask])
        X_te_df = _make_df(te_mask); y_te = y_all[te_mask]; a_te   = a_all[te_mask].astype(np.int64)

        # DA+ injection on train (capture24: female = disadv_id=1)
        if da_pct is not None:
            n_total = train_size if train_size is not None else len(X_tr_df)
            X_tr_df, y_tr, a_tr = self._inject_da_pct_bias(
                X_tr_df, y_tr, a_tr_s.to_numpy(dtype=np.int64), 1, da_pct, n_total, self.seed)
        else:
            a_tr = a_tr_s.to_numpy(dtype=np.int64)

        a_va = a_va_s.to_numpy(dtype=np.int64)

        # Cap val at train_size for stable reward signal
        if train_size is not None and train_size < len(X_va_df):
            strat_va = y_va * 2 + a_va
            X_va_df, _, y_va, _, a_va, _ = train_test_split(
                X_va_df, y_va, a_va, train_size=train_size,
                random_state=self.seed, stratify=strat_va)

        scaler = StandardScaler()
        Xtr_z = scaler.fit_transform(X_tr_df.values)
        Xva_z = scaler.transform(X_va_df.values)
        Xte_z = scaler.transform(X_te_df.values)

        Xtr_f, Xva_f, Xte_f, _ = self._apply_pca(Xtr_z, Xva_z, Xte_z, pca_components)

        self.dp_protected_col = dp_protected_col
        for tag, yi, ai in [("TRAIN", y_tr, a_tr), ("VAL", y_va, a_va), ("TEST", y_te, a_te)]:
            self._log_split(tag, yi, ai)
        print(f"fold {fold_idx} | held_out={list(held_out_subs)} | dim={Xtr_f.shape[1]}")

        return self._to_tensors(Xtr_f, Xva_f, Xte_f, y_tr, y_va, y_te, a_tr, a_va, a_te)

    # ----------------------------------------------------------------- router

    def get_data_splits(self, **kwargs):
        if self.dataset_name == "census_income":
            return self.split_census_income(**kwargs)
        if self.dataset_name == "capture24":
            kw = {k: v for k, v in kwargs.items()
                  if k not in ("drop_protected", "protected_cols")}
            if "fold_idx" in kw:
                valid = {"fold_idx", "n_folds", "train_size", "da_pct",
                         "pca_components", "dp_protected_col", "kfold_val_frac", "fold_rng_seed"}
                return self.split_capture24_kfold(**{k: v for k, v in kw.items() if k in valid})
            return self.split_capture24(**kw)
        raise ValueError(f"Unknown dataset: {self.dataset_name}")
