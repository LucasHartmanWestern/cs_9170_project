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
        "pamap2": {
            "data_path": "datasets/PAMAP2_Dataset",
            "protected_attributes": []
        },
        "credit_card": {
            "data_path":"datasets/default+of+credit+card+clients/default of credit card clients.xls",
            "protected_attributes": ["SEX", "AGE"]
        },
        "ptb_xl": {
            "data_path": "datasets/ptb-xl",
            "protected_attributes": ["age", "sex"]
        },
        "capture24": {
            "data_path": "datasets/capture24",
            "protected_attributes": ["sex"]
        },
        "compas": {
            "data_path": "datasets/compas",
            "protected_attributes": ["race", "sex"]
        },
        "meps": {
            "data_path": "datasets/meps",
            "protected_attributes": ["race", "sex"]
        },
        "acs_income": {
            "data_path": "datasets/acs_income",
            "protected_attributes": ["sex", "race"]
        },
        "fairjob": {
            "data_path": "datasets/fairjob",
            "protected_attributes": ["protected_attribute"]
        }
    }
    def __init__(self, dataset_name, multiclass, minority_id, majority_id, third_id, pca_components=2, seed=42, device="cpu", use_pca: bool = False, whiten_pca: bool = False):
        self.dataset_name = dataset_name
        self.MINORITY_ID = minority_id
        self.MAJORITY_ID = majority_id
        self.third_id=third_id
        self.pca_components = pca_components
        self.seed = seed
        self.device = device
        self.data_path = self.DATASET_REGISTRY[dataset_name]["data_path"]
        self.protected_attributes = self.DATASET_REGISTRY[self.dataset_name].get("protected_attributes", [])
        self.multiclass=multiclass
        self.use_pca = bool(use_pca)
        self.whiten_pca = bool(whiten_pca)

    @dataclass
    class GanView:
        supported: bool
        X_train_unbiased_df: pd.DataFrame
        y_train_unbiased: np.ndarray
        encoder: OneHotEncoder
        scaler: StandardScaler
        pca: PCA
        cat_cols: list[str]
        num_cols: list[str]

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
        
        return X_train_theta, X_val_theta, X_test_theta, pca

    def _make_pca(self, n_components: int) -> PCA:
        return PCA(
            n_components=n_components,
            svd_solver="full",          # deterministic basis for same data
            random_state=self.seed,     # deterministic for randomized solvers
            whiten=getattr(self, "whiten_pca", False),
        )

    def split_census_income(
            self,
            train_size=None,
            bias_pct=None,
            val_frac=0.20,
            test_frac=0.20,
            pca_components=2,
            drop_protected: bool = False,
            protected_cols=None,
            return_test_df: bool = False,
            return_raw: bool = False,
            bias_val: bool = True,
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

        # Helper: apply the SAME bias inside a split (keep all majority; keep fraction of minority)
        def apply_bias(df_split, y_split, a_split_df, target_minority_pct):
            df = df_split.copy()
            df["__y__"] = y_split
            df["__a__"] = a_split_df[dp_protected_col].to_numpy()
            df_major = df[df["__y__"] == 0]
            df_minor = df[df["__y__"] == 1]

            n_major = len(df_major)
            n_minor = len(df_minor)

            if n_minor == 0 or n_major == 0:
                df_biased = df
            else:
                keep_minority = int(np.floor((target_minority_pct * n_major) / (1 - target_minority_pct)))
                keep_minority = min(n_minor, max(1, keep_minority))
                df_minor_biased = df_minor.sample(n=keep_minority, random_state=self.seed, replace=False)
                df_biased = (
                    pd.concat([df_major, df_minor_biased], axis=0)
                    .sample(frac=1.0, random_state=self.seed)
                    .reset_index(drop=True)
                )

            y_out = df_biased["__y__"].to_numpy(dtype=int)
            a_out = _map_protected_census(df_biased["__a__"])
            X_out = df_biased.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        # 4) Apply bias: always train (if bias_pct set), conditionally val, never test
        if bias_pct is not None:
            target_minority_pct = bias_pct
            X_train_biased_df, y_train_biased, a_train = apply_bias(X_train_df, y_train, A_train_df, target_minority_pct)
            if bias_val:
                X_val_biased_df, y_val_biased, a_val = apply_bias(X_val_df, y_val, A_val_df, target_minority_pct)
            else:
                X_val_biased_df, y_val_biased = X_val_df.copy().reset_index(drop=True), y_val.copy()
                a_val = _map_protected_census(A_val_df[dp_protected_col])
        else:
            X_train_biased_df, y_train_biased = X_train_df.copy().reset_index(drop=True), y_train.copy()
            a_train = _map_protected_census(A_train_df[dp_protected_col])
            X_val_biased_df, y_val_biased = X_val_df.copy().reset_index(drop=True), y_val.copy()
            a_val = _map_protected_census(A_val_df[dp_protected_col])
        X_test_biased_df, y_test_biased = X_test_df.copy().reset_index(drop=True), y_test.copy()
        a_test = _map_protected_census(A_test_df[dp_protected_col])

        # Optional: subsample θ_train to fixed size after biasing
        if train_size is not None and train_size < len(X_train_biased_df):
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

        if return_test_df:
            return (
                X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta,
                X_test_biased_df.copy()    # <--- raw df for group membership
            )
        if return_raw:
            return (
                X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta,
                X_test_biased_df.copy(), y_test_biased.copy()
            )

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


    def split_pamap2(
        self,
        train_size=None,
        bias_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=6,
        win_seconds=5.0,
        step_seconds=2.5,
        drop_magnetometers=True,
        use_vector_norms=True,
        stats=("std", "rms"),
        bias_val: bool = True,
    ):
        """
        PAMAP2
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1, \
            "val_frac and test_frac must be in (0,1) and sum to < 1."

        # Cache for rebuild_original_train_pool_theta consistency
        self._win_seconds = win_seconds
        self._step_seconds = step_seconds

        rng = np.random.RandomState(self.seed)

        # constants / helpers 
        third_activity_id = self.third_id
        FS = 100
        WIN = int(win_seconds * FS)
        STEP = int(step_seconds * FS)
        data_dir = self.data_path
        imu_pos = ["hand", "chest", "ankle"]
        ACC16 = ["acc16g_x", "acc16g_y", "acc16g_z"]
        GYR = ["gyro_x", "gyro_y", "gyro_z"]
        MAG = ["mag_x", "mag_y", "mag_z"]

        def colnames():
            cols = ["timestamp", "activity_id", "heart_rate"]
            sub = ["temp","acc16g_x","acc16g_y","acc16g_z","acc6g_x","acc6g_y","acc6g_z",
                "gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z",
                "orient_w","orient_x","orient_y","orient_z"]
            for p in imu_pos:
                cols += [f"{p}_{s}" for s in sub]
            return cols  # 54 total

        def parse_sid(path):
            m = re.search(r"subject(\d+)", Path(path).stem.lower())
            return m.group(1) if m else "unknown"

        #load all .dat
        data_dir = Path(data_dir)
        files = []
        for sub in ["Protocol", "Optional", "protocol", "optional"]:
            d = data_dir / sub
            if d.exists():
                files += list(d.glob("subject*.dat"))
        if not files:
            raise FileNotFoundError(f"No .dat files found under {data_dir}")

        dfs = []
        for f in sorted(files):
            df = pd.read_csv(
                f, sep=r"\s+", header=None, names=colnames(),
                engine="python", na_values=["NaN", "nan"]
            )
            df["subject_id"] = parse_sid(f)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        #choose activities & map labels
        include_third_label = self.multiclass
        keep_ids = (
            [self.MAJORITY_ID, self.MINORITY_ID]
            if not include_third_label
            else [self.MAJORITY_ID, self.MINORITY_ID, third_activity_id]
        )
        df = df[df["activity_id"].isin(keep_ids)].copy()

        mapping = {self.MAJORITY_ID: 0, self.MINORITY_ID: 1}
        if include_third_label:
            mapping[third_activity_id] = 2
        df["y"] = df["activity_id"].map(mapping)

        #per-subject interpolation
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        num = [c for c in num if c not in ("activity_id", "y")]
        df = df.sort_values(["subject_id", "timestamp"]).groupby("subject_id", group_keys=False).apply(
            lambda g: g.assign(**{c: g[c].interpolate(limit_direction="both") for c in num})
        )

        #base channels (heart + acc16g + gyro +/- mag)
        keep_triplets = [("acc", ACC16), ("gyr", GYR)]
        if not drop_magnetometers:
            keep_triplets.append(("mag", MAG))

        base_cols = []
        if use_vector_norms:
            for p in imu_pos:
                for name, axes in keep_triplets:
                    cols = [f"{p}_{a}" for a in axes]
                    df[f"{p}_{name}_norm"] = np.sqrt((df[cols].values ** 2).sum(axis=1))
                    base_cols.append(f"{p}_{name}_norm")
        else:
            for p in imu_pos:
                for _, axes in keep_triplets:
                    for a in axes:
                        base_cols.append(f"{p}_{a}")

        base_cols = ["heart_rate"] + base_cols

        #windowing → features
        rows, labels, subjects = [], [], []
        df = df.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

        compute_mean = ("mean" in stats)
        compute_std  = ("std"  in stats)
        compute_rms  = ("rms"  in stats)

        for sid, g in df.groupby("subject_id", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            for start in range(0, max(0, n - WIN + 1), STEP):
                w = g.iloc[start:start + WIN]
                if len(w) < WIN:
                    continue

                y_win = int(np.bincount(w["y"].astype(int)).argmax())
                W = w[base_cols]

                feats = {}
                if compute_mean:
                    feats.update(W.mean().add_suffix("__mean").to_dict())
                if compute_std:
                    feats.update(W.std(ddof=1).add_suffix("__std").to_dict())
                if compute_rms:
                    feats.update((np.sqrt((W**2).mean())).add_suffix("__rms").to_dict())

                feats["subject_id"] = sid
                rows.append(feats)
                labels.append(y_win)
                subjects.append(sid)

        feat_df = pd.DataFrame(rows).fillna(0.0)
        y_all = np.asarray(labels, dtype=int)
        subj_all = np.asarray(subjects)

        # Temporal within-(subject, activity) split: for each subject × activity
        # block, the windows are split temporally so every activity from every
        # subject appears in all three splits.  This is critical when a single
        # subject holds the only female examples — it guarantees female runners
        # appear in train, val, and test regardless of when that subject ran
        # during the recording session.
        train_frac = 1.0 - val_frac - test_frac
        m_train = np.zeros(len(feat_df), dtype=bool)
        m_val   = np.zeros(len(feat_df), dtype=bool)
        m_test  = np.zeros(len(feat_df), dtype=bool)

        y_all_int = y_all.astype(int)
        for sid in np.unique(subj_all):
            for act in np.unique(y_all_int[subj_all == sid]):
                idx = np.where((subj_all == sid) & (y_all_int == act))[0]
                n = len(idx)
                if n < 3:   # too few windows to split — put in train
                    m_train[idx] = True
                    continue
                n_tr   = max(1, int(np.floor(train_frac * n)))
                n_val_ = max(1, int(np.floor(val_frac   * n)))
                n_tr   = min(n_tr,   n - 2)
                n_val_ = min(n_val_, n - n_tr - 1)
                m_train[idx[:n_tr]]                = True
                m_val  [idx[n_tr:n_tr + n_val_]]  = True
                m_test [idx[n_tr + n_val_:]]       = True

        X_train_df = feat_df[m_train].copy()
        X_val_df   = feat_df[m_val].copy()
        X_test_df  = feat_df[m_test].copy()
        y_train    = y_all[m_train]
        y_val      = y_all[m_val]
        y_test     = y_all[m_test]

        #bias (class 1) inside each split
        def apply_bias_rope(df_split, y_split, target_minority_pct, seed_local):
            dfb = df_split.copy()
            dfb["__y__"] = y_split

            mask_rope = (dfb["__y__"] == 1)
            df_rope = dfb[mask_rope]
            df_nonrope = dfb[~mask_rope]  # walking (+ third if present)

            n_nonrope = len(df_nonrope)
            n_rope = len(df_rope)

            if n_nonrope == 0 or n_rope == 0:
                out = dfb
            else:
                n_rope_keep = int(np.floor((target_minority_pct * n_nonrope) / (1 - target_minority_pct)))
                n_rope_keep = min(n_rope, max(1, n_rope_keep))
                df_rope_b = df_rope.sample(n=n_rope_keep, random_state=seed_local, replace=False)
                out = (
                    pd.concat([df_nonrope, df_rope_b], axis=0)
                    .sample(frac=1.0, random_state=seed_local)
                    .reset_index(drop=True)
                )

            y_out = out["__y__"].to_numpy(dtype=int)
            X_out = out.drop(columns=["__y__"])
            return X_out, y_out

        # Deterministic seeds
        seed_local = int(self.seed)

        # Bias: always train (if bias_pct set), conditionally val, never test
        if bias_pct is not None:
            target_minority_pct = float(bias_pct)
            X_train_biased_df, y_train_biased = apply_bias_rope(X_train_df, y_train, target_minority_pct, seed_local)
            if bias_val:
                X_val_biased_df, y_val_biased = apply_bias_rope(X_val_df, y_val, target_minority_pct, seed_local)
            else:
                X_val_biased_df, y_val_biased = X_val_df.copy().reset_index(drop=True), y_val.copy()
        else:
            X_train_biased_df, y_train_biased = X_train_df.copy().reset_index(drop=True), y_train.copy()
            X_val_biased_df, y_val_biased = X_val_df.copy().reset_index(drop=True), y_val.copy()
        X_test_biased_df, y_test_biased = X_test_df.copy().reset_index(drop=True), y_test.copy()


        # Map subject_id → sex (0=female, 1=male) using PAMAP2 demographics
        # Only subject 102 is female; all others (101, 103-109) are male
        FEMALE_SUBJECTS = {"102"}
        def _subject_to_sex(df_split):
            return np.where(
                df_split["subject_id"].isin(FEMALE_SUBJECTS), 0, 1
            ).astype(np.int64)

        a_train = _subject_to_sex(X_train_biased_df)
        a_val   = _subject_to_sex(X_val_biased_df)
        a_test  = _subject_to_sex(X_test_biased_df)

        #subsample TRAIN after biasing (deterministic)
        if train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _, a_train, _ = train_test_split(
                X_train_biased_df,
                y_train_biased,
                a_train,
                train_size=train_size,
                random_state=seed_local,
                stratify=y_train_biased,
            )

        #scaler + PCA on TRAIN only
        meta_cols = ["subject_id"]
        feature_cols = [c for c in X_train_biased_df.columns if c not in meta_cols]

        scaler = StandardScaler()
        Xtr_z = scaler.fit_transform(X_train_biased_df[feature_cols].values)
        Xval_z = scaler.transform(X_val_biased_df[feature_cols].values)
        Xtest_z = scaler.transform(X_test_biased_df[feature_cols].values)

        # Map to θ-space (PCA if enabled; otherwise identity)
        X_train_theta_np, X_val_theta_np, X_test_theta_np, pca = self._to_theta(
            Xtr_z, Xval_z, Xtest_z,
            pca_components=pca_components
        )

        # -------- tensors on device --------
        device = self.device
        X_train_theta = torch.tensor(X_train_theta_np, dtype=torch.float32, device=device)
        X_val_theta   = torch.tensor(X_val_theta_np,   dtype=torch.float32, device=device)
        X_test_theta  = torch.tensor(X_test_theta_np,  dtype=torch.float32, device=device)


        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=device)
        y_val_theta   = torch.tensor(y_val_biased, dtype=torch.long, device=device)
        y_test_theta  = torch.tensor(y_test_biased, dtype=torch.long, device=device)

        # Store protected attributes for fairness metrics
        self.a_train = torch.tensor(a_train, dtype=torch.long, device=device)
        self.a_val   = torch.tensor(a_val,   dtype=torch.long, device=device)
        self.a_test  = torch.tensor(a_test,  dtype=torch.long, device=device)
        self.dp_protected_col = "sex"

        # Cache GAN view for CTGAN/CTAB baselines
        try:
            self._gan_view_cache = {
                "supported": True,
                "X_train_unbiased_df": X_train_df.copy(),
                "y_train_unbiased": y_train.astype(int).copy(),
                "scaler": scaler,
                "pca": pca,
                "cat_cols": [],
                "num_cols": feature_cols,  # treat all as numeric
                "encoder": None,
            }
        except Exception:
            pass

        # ---- Sanity logging ----
        def log_counts(name, ysplit):
            vals, cnts = np.unique(ysplit, return_counts=True)
            parts = ", ".join([f"class{int(v)}={int(c)}" for v, c in zip(vals, cnts)])
            if 1 in vals:
                n_total = len(ysplit)
                n_rope = cnts[list(vals).index(1)]
                pct_rope = 100.0 * n_rope / n_total if n_total else 0.0
                print(f"[{name}] size={n_total}, rope={n_rope} ({pct_rope:.2f}%), {parts}")
            else:
                print(f"[{name}] size={len(ysplit)}, {parts}")

        log_counts("TRAIN", y_train_biased)
        log_counts("VAL",   y_val_biased)
        log_counts("TEST",  y_test_biased)
        print(
            f"use_pca={self.use_pca} | theta_dim={X_train_theta.shape[1]} | "
            f"features_in={len(feature_cols)} | classes={sorted(np.unique(y_train_biased))}"
        )


        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta

    def split_credit_card(
        self,
        train_size=None,
        bias_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=2,
        drop_protected: bool = False,
        protected_cols = ["SEX", "AGE"],
        return_test_df: bool = False,
        return_raw: bool = False,
        bias_val: bool = True,

        # --- NEW: which protected attr to use for DP reward ---
        dp_protected_col: str = "SEX",
        # UCI credit card: SEX is often coded as 1/2. We map to 0/1 via (== sex_positive_value).
        sex_positive_value: int = 2,
    ):
        """
        Excel (.xls) format used by UCI "Default of Credit Card Clients":
        - Row 1: X1..X23,Y headers
        - Row 2: real feature names (LIMIT_BAL, SEX, ...)
        - Data rows follow
        Pipeline matches split_census_income: split -> bias per split -> fit transforms on train only -> PCA -> tensors.
        Y = target, minority=1=did default
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1, \
            "val_frac and test_frac must be in (0,1) and sum to < 1."
        if self.multiclass:
            raise ValueError("credit_card dataset is binary only; set multiclass=False.")

        if protected_cols is None:
            protected_cols = ["SEX", "AGE"]

        # -----------------------------
        # 1) Load raw Excel + fix header
        # -----------------------------
        data_path = self.data_path

        # Read with first row as column headers: X1..X23,Y
        df = pd.read_excel(data_path, header=0)

        # Second row contains the true names
        # Example row: [ID, LIMIT_BAL, SEX, ..., PAY_AMT6, default payment next month]
        true_names = df.iloc[0].tolist()

        # Replace column names with true names
        df.columns = [str(x).strip() for x in true_names]

        # Drop the row we used for names
        df = df.iloc[1:].reset_index(drop=True)

        # Coerce to numeric where possible
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")

        # Label column: usually "default payment next month"
        label_col = None
        for cand in ["default payment next month", "default payment next month ", "Y"]:
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            # fallback: find a column that looks like the target from your screenshot
            matches = [c for c in df.columns if "default" in c.lower() and "next" in c.lower()]
            if matches:
                label_col = matches[0]
            else:
                raise ValueError(f"Could not find label column. Columns: {list(df.columns)}")

        # Drop ID if present
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        y_raw = df[label_col].astype(int).to_numpy()
        # sanity check label distribution (default should be ~20% in full dataset)
        vals, cnts = np.unique(y_raw, return_counts=True)
        print("Label distribution:", dict(zip(vals.tolist(), cnts.tolist())))
        if set(vals.tolist()) != {0, 1}:
            raise ValueError("Label parsing failed; y_raw not binary {0,1}")

        if dp_protected_col not in df.columns:
            raise ValueError(
                f"dp_protected_col={dp_protected_col!r} not found in columns. "
                f"Available: {list(df.columns)}"
            )

        A_df_raw = df[[dp_protected_col]].copy()
        X_df_raw = df.drop(columns=[label_col])

        # -----------------------------
        # 2) Optional: drop protected
        # -----------------------------
        if drop_protected:
            drop_cols = [c for c in protected_cols if c in X_df_raw.columns]
            if drop_cols:
                X_df_raw = X_df_raw.drop(columns=drop_cols)

        # -----------------------------
        # Column typing
        # -----------------------------
        # These are categorical by meaning (even if stored as ints)
        cat_cols = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X_df_raw.columns]
        # Everything else numeric
        num_cols = [c for c in X_df_raw.columns if c not in cat_cols]

        # -----------------------------
        # 3) Split raw -> train/val/test
        # -----------------------------
        X_train_df, X_temp_df, y_train, y_temp, A_train_df, A_temp_df = train_test_split(
            X_df_raw, y_raw, A_df_raw,
            test_size=(val_frac + test_frac),
            random_state=self.seed,
            stratify=y_raw
        )

        rel_test = test_frac / (val_frac + test_frac)
        X_val_df, X_test_df, y_val, y_test, A_val_df, A_test_df = train_test_split(
            X_temp_df, y_temp, A_temp_df,
            test_size=rel_test,
            random_state=self.seed,
            stratify=y_temp
        )

        # -----------------------------
        # 4) Apply same bias in each split
        # -----------------------------
        def _map_protected(a_raw):
            """Map raw protected attribute values to 0/1."""
            if dp_protected_col == "SEX":
                return (a_raw.astype(int) == int(sex_positive_value)).astype(np.int64)
            else:
                # AGE or other continuous: binary by median split
                med = np.median(a_raw.astype(float))
                return (a_raw.astype(float) >= med).astype(np.int64)

        def apply_bias(df_split, y_split, a_split_df, target_minority_pct):
            dfb = df_split.copy()
            dfb["__y__"] = y_split
            dfb["__a__"] = a_split_df[dp_protected_col].to_numpy()

            maj = dfb[dfb["__y__"] == 0]
            mino = dfb[dfb["__y__"] == 1]

            if len(maj) == 0 or len(mino) == 0:
                out = dfb
            else:
                keep_min = int(np.floor((target_minority_pct * len(maj)) / (1 - target_minority_pct)))
                keep_min = min(len(mino), max(1, keep_min))
                mino_b = mino.sample(n=keep_min, random_state=self.seed, replace=False)
                out = pd.concat([maj, mino_b], axis=0).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

            y_out = out["__y__"].to_numpy(dtype=int)
            a_out = _map_protected(out["__a__"].to_numpy())
            X_out = out.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        if bias_pct is not None:
            target_minority_pct = float(bias_pct)
            X_train_biased_df, y_train_biased, a_train = apply_bias(X_train_df, y_train, A_train_df, target_minority_pct)
            if bias_val:
                X_val_biased_df, y_val_biased, a_val = apply_bias(X_val_df, y_val, A_val_df, target_minority_pct)
            else:
                X_val_biased_df = X_val_df.copy().reset_index(drop=True)
                y_val_biased = y_val.copy()
                a_val = _map_protected(A_val_df[dp_protected_col].to_numpy())
        else:
            X_train_biased_df = X_train_df.copy().reset_index(drop=True)
            y_train_biased = y_train.copy()
            a_train = _map_protected(A_train_df[dp_protected_col].to_numpy())
            X_val_biased_df = X_val_df.copy().reset_index(drop=True)
            y_val_biased = y_val.copy()
            a_val = _map_protected(A_val_df[dp_protected_col].to_numpy())
        X_test_biased_df = X_test_df.copy().reset_index(drop=True)
        y_test_biased = y_test.copy()
        a_test = _map_protected(A_test_df[dp_protected_col].to_numpy())


        # Optional subsample biased train
        if train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _, a_train, _ = train_test_split(
                X_train_biased_df, y_train_biased, a_train,
                train_size=train_size,
                random_state=self.seed,
                stratify=y_train_biased
            )


        # -----------------------------
        # 5) Fit encoder + scaler on train only
        # -----------------------------
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

        scaler = StandardScaler()

        Xtr_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if cat_cols else np.empty((len(X_train_biased_df), 0))
        Xtr_num = scaler.fit_transform(X_train_biased_df[num_cols])  if num_cols else np.empty((len(X_train_biased_df), 0))
        Xtr_all = np.hstack([Xtr_num, Xtr_cat])

        Xva_cat = encoder.transform(X_val_biased_df[cat_cols]) if cat_cols else np.empty((len(X_val_biased_df), 0))
        Xva_num = scaler.transform(X_val_biased_df[num_cols])  if num_cols else np.empty((len(X_val_biased_df), 0))
        Xva_all = np.hstack([Xva_num, Xva_cat])

        Xte_cat = encoder.transform(X_test_biased_df[cat_cols]) if cat_cols else np.empty((len(X_test_biased_df), 0))
        Xte_num = scaler.transform(X_test_biased_df[num_cols])  if num_cols else np.empty((len(X_test_biased_df), 0))
        Xte_all = np.hstack([Xte_num, Xte_cat])

        # -----------------------------
        # 6) Map to θ-space (PCA if enabled; otherwise identity)
        # -----------------------------
        X_train_theta_np, X_val_theta_np, X_test_theta_np, pca = self._to_theta(
            Xtr_all, Xva_all, Xte_all,
            pca_components=pca_components
        )

        # -----------------------------
        # 7) Torch tensors
        # -----------------------------
        X_train_theta = torch.tensor(X_train_theta_np, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_val_theta_np,   dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_theta_np,  dtype=torch.float32, device=self.device)


        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_biased, dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_biased, dtype=torch.long, device=self.device)

        self.a_train = torch.tensor(a_train, dtype=torch.long, device=self.device)
        self.a_val   = torch.tensor(a_val,   dtype=torch.long, device=self.device)
        self.a_test  = torch.tensor(a_test,  dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col

        # ---- Logging ----
        def log_dist(name, ysplit):
            n = len(ysplit)
            n1 = int(np.sum(ysplit == 1))
            p1 = 100.0 * n1 / n if n else 0.0
            print(f"[{name}] size={n}, minority={n1} ({p1:.2f}%)")
        log_dist("TRAIN", y_train_biased)
        log_dist("VAL",   y_val_biased)
        log_dist("TEST",  y_test_biased)

        # Cache GAN view (unbiased train + transforms)
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
                "label_col": label_col,
                "dp_protected_col": dp_protected_col,
            }
        except Exception:
            pass

        if return_test_df:
            return (
                X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta,
                X_test_biased_df.copy()
            )
        if return_raw:
            return (
                X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta,
                X_test_biased_df.copy(), y_test_biased.copy()
            )

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


    def split_compas(
        self,
        train_size=None,
        bias_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=2,
        drop_protected: bool = False,
        protected_cols=None,
        return_test_df: bool = False,
        return_raw: bool = False,
        bias_val: bool = True,
        dp_protected_col: str = "race",
    ):
        """
        COMPAS Recidivism dataset (ProPublica, 2016).
        CSV: datasets/compas/compas-scores-two-years.csv

        Applies standard ProPublica filters then restricts to African-American
        and Caucasian defendants (the two largest groups). Race is encoded as
        binary: 0=Caucasian, 1=African-American.

        Label: two_year_recid (0=no recidivism, 1=recidivism).
        Protected attr: race (binary) or sex (binary).
        Disadvantaged group: Caucasian (minority_id=0) — naturally fewer
        positive examples, so positive-class scarcity is most pronounced.

        Pipeline mirrors split_census_income: split → bias per split →
        fit encoder/scaler on train only → PCA → tensors.
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1

        if protected_cols is None:
            protected_cols = ["race", "sex"]

        # ---- 1) Load and filter ----
        df = pd.read_csv(Path(self.data_path) / "compas-scores-two-years.csv")

        # Standard ProPublica filters
        df = df[df["days_b_screening_arrest"] <= 30]
        df = df[df["days_b_screening_arrest"] >= -30]
        df = df[df["is_recid"] != -1]
        df = df[df["c_charge_degree"] != "O"]
        df = df[df["score_text"] != "N/A"]

        if dp_protected_col == "race":
            df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
        df = df.reset_index(drop=True)

        label_col = "two_year_recid"
        y_raw = df[label_col].astype(int).to_numpy()

        vals, cnts = np.unique(y_raw, return_counts=True)
        print("[compas] Label distribution:", dict(zip(vals.tolist(), cnts.tolist())))
        if set(vals.tolist()) != {0, 1}:
            raise ValueError("Label parsing failed; expected binary {0,1}")

        A_df_raw = df[[dp_protected_col]].copy()

        # Feature columns (ProPublica canonical set)
        num_cols_all = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count",
                        "priors_count", "days_b_screening_arrest", "c_days_from_compas"]
        # Include the protected attribute as a feature (same convention as sex
        # experiments where sex is both feature and protected attr).
        # drop_protected=True will remove it later if needed.
        if dp_protected_col == "race":
            cat_cols_all = ["race", "sex", "age_cat", "c_charge_degree"]
        else:
            cat_cols_all = ["sex", "age_cat", "c_charge_degree"]

        feature_cols = num_cols_all + cat_cols_all
        X_df_raw = df[feature_cols].copy()

        # ---- 2) Optional: drop protected ----
        if drop_protected:
            drop_c = [c for c in protected_cols if c in X_df_raw.columns]
            if drop_c:
                X_df_raw = X_df_raw.drop(columns=drop_c)

        cat_cols = [c for c in cat_cols_all if c in X_df_raw.columns]
        num_cols = [c for c in num_cols_all if c in X_df_raw.columns]

        # ---- 3) Split ----
        X_train_df, X_temp_df, y_train, y_temp, A_train_df, A_temp_df = train_test_split(
            X_df_raw, y_raw, A_df_raw,
            test_size=(val_frac + test_frac),
            random_state=self.seed,
            stratify=y_raw,
        )
        rel_test = test_frac / (val_frac + test_frac)
        X_val_df, X_test_df, y_val, y_test, A_val_df, A_test_df = train_test_split(
            X_temp_df, y_temp, A_temp_df,
            test_size=rel_test,
            random_state=self.seed,
            stratify=y_temp,
        )

        # ---- Protected attribute mapping ----
        def _map_protected(a_series):
            if dp_protected_col == "race":
                # 0 = Caucasian (disadvantaged/minority), 1 = African-American
                return (a_series != "Caucasian").astype(np.int64).to_numpy()
            elif dp_protected_col == "sex":
                # 0 = Female (minority), 1 = Male
                return (a_series.str.lower() == "male").astype(np.int64).to_numpy()
            else:
                raise ValueError(f"Unsupported dp_protected_col: {dp_protected_col!r}")

        # ---- 4) Bias injection ----
        def apply_bias(df_split, y_split, a_split_df, target_minority_pct):
            dfb = df_split.copy()
            dfb["__y__"] = y_split
            dfb["__a__"] = a_split_df[dp_protected_col].to_numpy()

            if dp_protected_col == "race":
                # Group-specific bias: reduce only Caucasian (a=0) positives.
                # African-American records are kept at their natural rate so the
                # classifier can learn the positive class from the advantaged group.
                # target_minority_pct = target positive rate *within* Caucasian subgroup.
                is_cauc     = dfb["__a__"] == "Caucasian"
                df_cauc     = dfb[is_cauc]
                df_aa       = dfb[~is_cauc]
                df_cauc_neg = df_cauc[df_cauc["__y__"] == 0]
                df_cauc_pos = df_cauc[df_cauc["__y__"] == 1]
                if len(df_cauc_neg) == 0 or len(df_cauc_pos) == 0:
                    out = dfb
                else:
                    keep_min = int(np.floor(
                        (target_minority_pct * len(df_cauc_neg)) / (1 - target_minority_pct)
                    ))
                    keep_min = min(len(df_cauc_pos), max(1, keep_min))
                    df_cauc_pos_b = df_cauc_pos.sample(
                        n=keep_min, random_state=self.seed, replace=False
                    )
                    out = pd.concat(
                        [df_aa, df_cauc_neg, df_cauc_pos_b], axis=0
                    ).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
            else:
                df_maj = dfb[dfb["__y__"] == 0]
                df_min = dfb[dfb["__y__"] == 1]
                if len(df_maj) == 0 or len(df_min) == 0:
                    out = dfb
                else:
                    keep_min = int(np.floor((target_minority_pct * len(df_maj)) / (1 - target_minority_pct)))
                    keep_min = min(len(df_min), max(1, keep_min))
                    df_min_b = df_min.sample(n=keep_min, random_state=self.seed, replace=False)
                    out = pd.concat([df_maj, df_min_b], axis=0).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

            y_out = out["__y__"].to_numpy(dtype=int)
            a_out = _map_protected(out["__a__"])
            X_out = out.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        if bias_pct is not None:
            X_train_biased_df, y_train_biased, a_train = apply_bias(X_train_df, y_train, A_train_df, float(bias_pct))
            if bias_val:
                X_val_biased_df, y_val_biased, a_val = apply_bias(X_val_df, y_val, A_val_df, float(bias_pct))
            else:
                X_val_biased_df = X_val_df.copy().reset_index(drop=True)
                y_val_biased = y_val.copy()
                a_val = _map_protected(A_val_df[dp_protected_col])
        else:
            X_train_biased_df = X_train_df.copy().reset_index(drop=True)
            y_train_biased = y_train.copy()
            a_train = _map_protected(A_train_df[dp_protected_col])
            X_val_biased_df = X_val_df.copy().reset_index(drop=True)
            y_val_biased = y_val.copy()
            a_val = _map_protected(A_val_df[dp_protected_col])

        X_test_biased_df = X_test_df.copy().reset_index(drop=True)
        y_test_biased = y_test.copy()
        a_test = _map_protected(A_test_df[dp_protected_col])

        # ---- Optional train subsample ----
        if train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _, a_train, _ = train_test_split(
                X_train_biased_df, y_train_biased, a_train,
                train_size=train_size,
                random_state=self.seed,
                stratify=y_train_biased,
            )

        # ---- 5) Fit encoder + scaler on train only ----
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        scaler = StandardScaler()

        Xtr_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if cat_cols else np.empty((len(X_train_biased_df), 0))
        Xtr_num = scaler.fit_transform(X_train_biased_df[num_cols])  if num_cols else np.empty((len(X_train_biased_df), 0))
        Xtr_all = np.hstack([Xtr_num, Xtr_cat])

        Xva_cat = encoder.transform(X_val_biased_df[cat_cols]) if cat_cols else np.empty((len(X_val_biased_df), 0))
        Xva_num = scaler.transform(X_val_biased_df[num_cols])  if num_cols else np.empty((len(X_val_biased_df), 0))
        Xva_all = np.hstack([Xva_num, Xva_cat])

        Xte_cat = encoder.transform(X_test_biased_df[cat_cols]) if cat_cols else np.empty((len(X_test_biased_df), 0))
        Xte_num = scaler.transform(X_test_biased_df[num_cols])  if num_cols else np.empty((len(X_test_biased_df), 0))
        Xte_all = np.hstack([Xte_num, Xte_cat])

        # ---- 6) PCA ----
        X_train_theta_np, X_val_theta_np, X_test_theta_np, pca = self._to_theta(
            Xtr_all, Xva_all, Xte_all, pca_components=pca_components
        )

        # ---- 7) Tensors ----
        X_train_theta = torch.tensor(X_train_theta_np, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_val_theta_np,   dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_theta_np,  dtype=torch.float32, device=self.device)

        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_biased,   dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_biased,  dtype=torch.long, device=self.device)

        self.a_train = torch.tensor(a_train, dtype=torch.long, device=self.device)
        self.a_val   = torch.tensor(a_val,   dtype=torch.long, device=self.device)
        self.a_test  = torch.tensor(a_test,  dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col

        def log_dist(name, y_split):
            n = len(y_split); n1 = int(np.sum(y_split == 1))
            print(f"[compas/{name}] size={n}, minority={n1} ({100.*n1/n if n else 0:.2f}%)")
        log_dist("TRAIN", y_train_biased)
        log_dist("VAL",   y_val_biased)
        log_dist("TEST",  y_test_biased)

        try:
            self._gan_view_cache = {
                "supported": True,
                "X_train_unbiased_df": X_train_df.copy(),
                "y_train_unbiased": y_train.astype(int).copy(),
                "encoder": encoder,
                "scaler": scaler,
                "pca": pca,
                "use_pca": self.use_pca,
                "pca_components": int(pca_components),
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
                "drop_protected": bool(drop_protected),
                "protected_cols": list(protected_cols),
                "label_col": label_col,
                "dp_protected_col": dp_protected_col,
            }
        except Exception:
            pass

        if return_test_df:
            return (X_train_theta, X_val_theta, X_test_theta,
                    y_train_theta, y_val_theta, y_test_theta,
                    X_test_biased_df.copy())
        if return_raw:
            return (X_train_theta, X_val_theta, X_test_theta,
                    y_train_theta, y_val_theta, y_test_theta,
                    X_test_biased_df.copy(), y_test_biased.copy())

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


    def split_meps(
        self,
        train_size=None,
        bias_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=2,
        drop_protected: bool = False,
        protected_cols=None,
        return_test_df: bool = False,
        return_raw: bool = False,
        bias_val: bool = True,
        dp_protected_col: str = "race",
    ):
        """
        MEPS Full Year Consolidated 2022 (HC-243).

        Target: received any dental visit in 2022 (DVTOT22 >= 1, y=1).
        Protected attr: race (0=Black non-Hispanic/disadvantaged, 1=White non-Hispanic).
        Bias: group-specific -- only Black Americans' positives are reduced, mirroring
              split_compas. White Americans' dental-visit rate is preserved at its
              natural level so the classifier can learn the positive class.

        Natural positive-class rates (pre-bias): ~30% Black, ~49% White.
        At real_data_size=3000 with bias injection, DA+≈43 is achievable for Black group.

        Expected file: datasets/meps/h243.csv  (pre-converted from SAS transport)
        Fallback:      datasets/meps/h243.ssp  (SAS transport, read via pandas.read_sas)
        Download from: https://meps.ahrq.gov/data_stats/download_data_files.jsp (HC-243)
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1

        if protected_cols is None:
            protected_cols = ["ethnicity"] if dp_protected_col == "ethnicity" else ["race"]

        data_dir = Path(self.data_path)

        # ---- 1) Load ----
        csv_path = data_dir / "h243.csv"
        sas_path = data_dir / "h243.ssp"
        if csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
        elif sas_path.exists():
            df = pd.read_sas(str(sas_path), encoding="iso-8859-1")
            df.to_csv(csv_path, index=False)
        else:
            raise FileNotFoundError(
                f"MEPS HC-243 not found. Expected:\n  {csv_path}\n  {sas_path}\n"
                "Download HC-243 from https://meps.ahrq.gov/data_stats/download_data_files.jsp"
            )

        df.columns = [c.upper() for c in df.columns]

        # Stata categoricals arrive as "2 NON-HISPANIC WHITE ONLY" etc.
        # Extract the leading integer code for all relevant columns.
        def _meps_code(series):
            return pd.to_numeric(series.astype(str).str.split().str[0], errors="coerce")

        # RACETHX: 1=Hispanic, 2=White non-Hisp, 3=Black non-Hisp, 4=Asian, 5=Other non-Hisp
        df["RACETHX"] = _meps_code(df["RACETHX"])

        if dp_protected_col == "ethnicity":
            # ---- 2) Keep all race groups (Hispanic vs everyone else) ----
            df = df[df["RACETHX"].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

            # ---- 3) Target: any outpatient facility visit ----
            # OPTOTV22: total outpatient visits 2022. Negative codes = inapplicable/missing → 0.
            df["OPTOTV22"] = _meps_code(df["OPTOTV22"]).fillna(0).clip(lower=0)
            y_raw = (df["OPTOTV22"] >= 1).astype(int).to_numpy()

            vals, cnts = np.unique(y_raw, return_counts=True)
            print("[meps/ethnicity] Label distribution (pre-bias):", dict(zip(vals.tolist(), cnts.tolist())))
            for group_code, group_name in [(1, "Hispanic"), (2, "White non-Hisp"), (3, "Black non-Hisp"), (4, "Asian non-Hisp"), (5, "Other non-Hisp")]:
                mask = df["RACETHX"] == group_code
                n = int(mask.sum())
                if n == 0:
                    continue
                n1 = int((y_raw[mask.to_numpy()] == 1).sum())
                print(f"  [{group_name}] n={n}, outpatient+= {n1} ({100.*n1/n:.1f}%)")

            A_df_raw = df[["RACETHX"]].rename(columns={"RACETHX": "ethnicity"}).copy()
        else:
            # ---- 2) Filter to Black non-Hispanic (RACETHX=3) and White non-Hispanic (RACETHX=2) ----
            df = df[df["RACETHX"].isin([2, 3])].copy().reset_index(drop=True)

            # ---- 3) Target: any dental visit ----
            # DVTOT22: total dental visits 2022. Negative codes treated as 0.
            df["DVTOT22"] = _meps_code(df["DVTOT22"]).fillna(0).clip(lower=0)
            y_raw = (df["DVTOT22"] >= 1).astype(int).to_numpy()

            vals, cnts = np.unique(y_raw, return_counts=True)
            print("[meps/race] Label distribution (pre-bias):", dict(zip(vals.tolist(), cnts.tolist())))
            for group_code, group_name in [(3, "Black non-Hisp"), (2, "White non-Hisp")]:
                mask = df["RACETHX"] == group_code
                n = mask.sum(); n1 = int((y_raw[mask.to_numpy()] == 1).sum())
                print(f"  [{group_name}] n={n}, dental+= {n1} ({100.*n1/n if n else 0:.1f}%)")

            A_df_raw = df[["RACETHX"]].rename(columns={"RACETHX": "race"}).copy()

        # ---- 4) Feature columns ----
        num_cols_all = ["AGE22X", "TTLP22X"]
        cat_cols_all = ["SEX", "RACETHX", "POVCAT22", "INSCOV22", "REGION22", "MARRY22X"]

        # Convert Stata-label strings to integer codes; replace negatives (special codes) with NaN
        for col in cat_cols_all:
            if col in df.columns:
                df[col] = _meps_code(df[col])
                df.loc[df[col] < 0, col] = np.nan
        for col in num_cols_all:
            if col in df.columns:
                df[col] = _meps_code(df[col])

        feature_cols = [c for c in num_cols_all + cat_cols_all if c in df.columns]
        X_df_raw = df[feature_cols].copy()

        # Categoricals as strings so OneHotEncoder treats them as nominal
        for col in cat_cols_all:
            if col in X_df_raw.columns:
                X_df_raw[col] = X_df_raw[col].astype("str")

        # ---- 5) Optional: drop protected ----
        if drop_protected:
            drop_c = [c for c in protected_cols + ["RACETHX"] if c in X_df_raw.columns]
            if drop_c:
                X_df_raw = X_df_raw.drop(columns=drop_c)

        cat_cols = [c for c in X_df_raw.columns if X_df_raw[c].dtype == object]
        num_cols = [c for c in X_df_raw.columns if np.issubdtype(X_df_raw[c].dtype, np.number)]

        # ---- 6) Split ----
        X_train_df, X_temp_df, y_train, y_temp, A_train_df, A_temp_df = train_test_split(
            X_df_raw, y_raw, A_df_raw,
            test_size=(val_frac + test_frac),
            random_state=self.seed,
            stratify=y_raw,
        )
        rel_test = test_frac / (val_frac + test_frac)
        X_val_df, X_test_df, y_val, y_test, A_val_df, A_test_df = train_test_split(
            X_temp_df, y_temp, A_temp_df,
            test_size=rel_test,
            random_state=self.seed,
            stratify=y_temp,
        )

        # ---- Protected attribute mapping ----
        if dp_protected_col == "ethnicity":
            # 0 = Hispanic (disadvantaged, RACETHX=1)
            # 1 = non-Hispanic (advantaged, RACETHX in {2,3,4,5})
            def _map_protected(a_df):
                return (a_df["ethnicity"] != 1).astype(np.int64).to_numpy()
            _a_col = "ethnicity"
            _disadv_code = 1  # Hispanic
        else:
            # 0 = Black non-Hispanic (disadvantaged, RACETHX=3)
            # 1 = White non-Hispanic (advantaged, RACETHX=2)
            def _map_protected(a_df):
                return (a_df["race"] != 3).astype(np.int64).to_numpy()
            _a_col = "race"
            _disadv_code = 3  # Black non-Hisp

        # ---- 7) Bias injection (group-specific: reduce only disadvantaged-group positives) ----
        def apply_bias(df_split, y_split, a_split_df, target_minority_pct):
            dfb = df_split.copy()
            dfb["__y__"] = y_split
            dfb["__a__"] = a_split_df[_a_col].to_numpy()

            is_disadv    = dfb["__a__"] == _disadv_code
            df_disadv    = dfb[is_disadv]
            df_adv       = dfb[~is_disadv]
            df_disadv_neg = df_disadv[df_disadv["__y__"] == 0]
            df_disadv_pos = df_disadv[df_disadv["__y__"] == 1]

            if len(df_disadv_neg) == 0 or len(df_disadv_pos) == 0:
                out = dfb
            else:
                keep_min = int(np.floor(
                    (target_minority_pct * len(df_disadv_neg)) / (1 - target_minority_pct)
                ))
                keep_min = min(len(df_disadv_pos), max(1, keep_min))
                df_disadv_pos_b = df_disadv_pos.sample(n=keep_min, random_state=self.seed, replace=False)
                out = pd.concat(
                    [df_adv, df_disadv_neg, df_disadv_pos_b], axis=0
                ).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

            y_out = out["__y__"].to_numpy(dtype=int)
            a_out = _map_protected(out[["__a__"]].rename(columns={"__a__": _a_col}))
            X_out = out.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        if bias_pct is not None:
            X_train_biased_df, y_train_biased, a_train = apply_bias(
                X_train_df, y_train, A_train_df, float(bias_pct))
            if bias_val:
                X_val_biased_df, y_val_biased, a_val = apply_bias(
                    X_val_df, y_val, A_val_df, float(bias_pct))
            else:
                X_val_biased_df = X_val_df.copy().reset_index(drop=True)
                y_val_biased = y_val.copy()
                a_val = _map_protected(A_val_df)
        else:
            X_train_biased_df = X_train_df.copy().reset_index(drop=True)
            y_train_biased = y_train.copy()
            a_train = _map_protected(A_train_df)
            X_val_biased_df = X_val_df.copy().reset_index(drop=True)
            y_val_biased = y_val.copy()
            a_val = _map_protected(A_val_df)

        X_test_biased_df = X_test_df.copy().reset_index(drop=True)
        y_test_biased = y_test.copy()
        a_test = _map_protected(A_test_df)

        # ---- Optional train subsample ----
        if train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _, a_train, _ = train_test_split(
                X_train_biased_df, y_train_biased, a_train,
                train_size=train_size,
                random_state=self.seed,
                stratify=y_train_biased,
            )

        # DA+ diagnostic
        da_plus = int(np.sum((y_train_biased == 1) & (a_train == 0)))
        n_disadv = int(np.sum(a_train == 0))
        if dp_protected_col == "ethnicity":
            print(f"[meps/TRAIN] DA+ (Hispanic, outpatient=1): {da_plus} of {n_disadv} Hispanic train examples")
        else:
            print(f"[meps/TRAIN] DA+ (Black, dental=1): {da_plus} of {n_disadv} Black train examples")

        # ---- 8) Fit encoder + scaler on train only ----
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        scaler = StandardScaler()

        Xtr_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if cat_cols else np.empty((len(X_train_biased_df), 0))
        Xtr_num = scaler.fit_transform(X_train_biased_df[num_cols].fillna(0)) if num_cols else np.empty((len(X_train_biased_df), 0))
        Xtr_all = np.hstack([Xtr_num, Xtr_cat])

        Xva_cat = encoder.transform(X_val_biased_df[cat_cols]) if cat_cols else np.empty((len(X_val_biased_df), 0))
        Xva_num = scaler.transform(X_val_biased_df[num_cols].fillna(0)) if num_cols else np.empty((len(X_val_biased_df), 0))
        Xva_all = np.hstack([Xva_num, Xva_cat])

        Xte_cat = encoder.transform(X_test_biased_df[cat_cols]) if cat_cols else np.empty((len(X_test_biased_df), 0))
        Xte_num = scaler.transform(X_test_biased_df[num_cols].fillna(0)) if num_cols else np.empty((len(X_test_biased_df), 0))
        Xte_all = np.hstack([Xte_num, Xte_cat])

        # ---- 9) PCA ----
        X_train_theta_np, X_val_theta_np, X_test_theta_np, pca = self._to_theta(
            Xtr_all, Xva_all, Xte_all, pca_components=pca_components
        )

        # ---- 10) Tensors ----
        X_train_theta = torch.tensor(X_train_theta_np, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_val_theta_np,   dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_theta_np,  dtype=torch.float32, device=self.device)

        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_biased,   dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_biased,  dtype=torch.long, device=self.device)

        self.a_train = torch.tensor(a_train, dtype=torch.long, device=self.device)
        self.a_val   = torch.tensor(a_val,   dtype=torch.long, device=self.device)
        self.a_test  = torch.tensor(a_test,  dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col

        outcome_label = "outpatient+" if dp_protected_col == "ethnicity" else "dental+"
        def log_dist(name, y_split):
            n = len(y_split); n1 = int(np.sum(y_split == 1))
            print(f"[meps/{name}] size={n}, {outcome_label} {n1} ({100.*n1/n if n else 0:.2f}%)")
        log_dist("TRAIN", y_train_biased)
        log_dist("VAL",   y_val_biased)
        log_dist("TEST",  y_test_biased)

        label_col = "OPTOTV22_binary" if dp_protected_col == "ethnicity" else "DVTOT22_binary"
        try:
            self._gan_view_cache = {
                "supported": True,
                "X_train_unbiased_df": X_train_df.copy(),
                "y_train_unbiased": y_train.astype(int).copy(),
                "encoder": encoder,
                "scaler": scaler,
                "pca": pca,
                "use_pca": self.use_pca,
                "pca_components": int(pca_components),
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
                "drop_protected": bool(drop_protected),
                "protected_cols": list(protected_cols),
                "label_col": label_col,
                "dp_protected_col": dp_protected_col,
            }
        except Exception:
            pass

        if return_test_df:
            return (X_train_theta, X_val_theta, X_test_theta,
                    y_train_theta, y_val_theta, y_test_theta,
                    X_test_biased_df.copy())
        if return_raw:
            return (X_train_theta, X_val_theta, X_test_theta,
                    y_train_theta, y_val_theta, y_test_theta,
                    X_test_biased_df.copy(), y_test_biased.copy())

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


    def split_acs_income(
        self,
        train_size=None,
        bias_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=10,
        bias_val: bool = False,
        dp_protected_col: str = "sex",
        acs_state: str = "CA",
        acs_year: str = "2018",
    ):
        """
        ACS Income (Ding et al. NeurIPS 2021 / folktables).

        Protected attribute: sex. Female (SEX=2, a=0) is disadvantaged;
        male (SEX=1, a=1) is advantaged. Matches census_income convention
        (minority_id=0 = disadvantaged female).

        Bias injection is group-specific: only female positives are reduced
        in the training split (and optionally val). Male positives are kept
        at their natural rate.

        Data is downloaded on first call to datasets/acs_income/ via folktables.
        """
        from folktables import ACSDataSource, ACSIncome

        data_dir = Path(self.data_path)
        data_dir.mkdir(parents=True, exist_ok=True)

        data_source = ACSDataSource(
            survey_year=acs_year, horizon="1-Year", survey="person",
            root_dir=str(data_dir),
        )
        acs_data = data_source.get_data(states=[acs_state], download=True)
        features, label, _ = ACSIncome.df_to_pandas(acs_data)

        # Fix label — folktables returns tuples (True,)/(False,)
        y_raw = np.array([int(bool(v[0])) if isinstance(v, tuple) else int(bool(v))
                          for v in label.values], dtype=int)

        # All ACSIncome features are numeric; no categorical encoding needed
        X_raw = features.values.astype(float)
        col_names = list(features.columns)

        # Protected attribute: SEX column (1=male, 2=female)
        sex_vals = features["SEX"].values
        a_raw = (sex_vals == 1).astype(np.int64)  # 1=male (adv), 0=female (disadv)

        print(f"[acs_income] Total: {len(y_raw)}, pos_rate={y_raw.mean():.3f}")
        print(f"[acs_income] Female (a=0): n={int((a_raw==0).sum())}, "
              f"pos_rate={y_raw[a_raw==0].mean():.3f}, "
              f"pos_total={int(np.sum((a_raw==0)&(y_raw==1)))}")
        print(f"[acs_income] Male   (a=1): n={int((a_raw==1).sum())}, "
              f"pos_rate={y_raw[a_raw==1].mean():.3f}, "
              f"pos_total={int(np.sum((a_raw==1)&(y_raw==1)))}")

        # Stratified split
        idx_all = np.arange(len(y_raw))
        idx_tr, idx_temp = train_test_split(
            idx_all, test_size=(val_frac + test_frac),
            random_state=self.seed, stratify=y_raw)
        rel_test = test_frac / (val_frac + test_frac)
        idx_va, idx_te = train_test_split(
            idx_temp, test_size=rel_test,
            random_state=self.seed, stratify=y_raw[idx_temp])

        X_tr_raw, y_tr, a_tr = X_raw[idx_tr], y_raw[idx_tr], a_raw[idx_tr]
        X_va_raw, y_va, a_va = X_raw[idx_va], y_raw[idx_va], a_raw[idx_va]
        X_te_raw, y_te, a_te = X_raw[idx_te], y_raw[idx_te], a_raw[idx_te]

        # Group-specific bias: reduce female (a=0) positives in train
        def apply_group_bias(X, y, a, bp, seed):
            idx_adv  = np.where(a == 1)[0]           # male — keep all
            idx_dneg = np.where((a==0) & (y==0))[0]  # female neg — keep all
            idx_dpos = np.where((a==0) & (y==1))[0]  # female pos — reduce
            keep = max(1, int(np.floor(bp * len(idx_dneg) / (1 - bp))))
            keep = min(len(idx_dpos), keep)
            rng2 = np.random.RandomState(seed)
            kept = rng2.choice(idx_dpos, size=keep, replace=False)
            sel  = np.concatenate([idx_adv, idx_dneg, kept])
            perm = np.random.RandomState(seed).permutation(len(sel))
            return X[sel[perm]], y[sel[perm]], a[sel[perm]]

        if bias_pct is not None:
            X_tr_raw, y_tr, a_tr = apply_group_bias(X_tr_raw, y_tr, a_tr, bias_pct, self.seed)
            if bias_val:
                X_va_raw, y_va, a_va = apply_group_bias(X_va_raw, y_va, a_va, bias_pct, self.seed)

        da_plus = int(np.sum((y_tr==1) & (a_tr==0)))
        n_disadv_tr = int((a_tr==0).sum())
        print(f"[acs_income/TRAIN] DA+ (female, income>50k): {da_plus} of {n_disadv_tr} female train examples")

        # Cap train_size after bias (stratified), matching census pipeline
        if train_size is not None and train_size < len(y_tr):
            X_tr_raw, _, y_tr, _, a_tr, _ = train_test_split(
                X_tr_raw, y_tr, a_tr,
                train_size=train_size, random_state=self.seed, stratify=y_tr)

        def _log(name, y_split):
            n = len(y_split); n1 = int(np.sum(y_split==1))
            print(f"[acs_income/{name}] size={n}, income>50k: {n1} ({100.*n1/n:.2f}%)")

        _log("TRAIN", y_tr)
        _log("VAL",   y_va)
        _log("TEST",  y_te)

        # StandardScaler (no OHE — all features are numeric)
        scaler = StandardScaler().fit(X_tr_raw)
        X_tr_sc = scaler.transform(X_tr_raw)
        X_va_sc = scaler.transform(X_va_raw)
        X_te_sc = scaler.transform(X_te_raw)

        X_tr_theta, X_va_theta, X_te_theta, pca_obj = self._to_theta(
            X_tr_sc, X_va_sc, X_te_sc, pca_components=pca_components)

        X_train_theta = torch.tensor(X_tr_theta, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_va_theta, dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_te_theta, dtype=torch.float32, device=self.device)
        y_train_theta = torch.tensor(y_tr, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_va, dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_te, dtype=torch.long, device=self.device)

        self.a_train = torch.tensor(a_tr, dtype=torch.long, device=self.device)
        self.a_val   = torch.tensor(a_va, dtype=torch.long, device=self.device)
        self.a_test  = torch.tensor(a_te, dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col

        self._gan_view_cache = {"supported": False}

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


    def split_ptb_xl(
        self,
        train_size=None,
        bias_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=10,
        return_test_df: bool = False,
        return_raw: bool = False,
        bias_val: bool = True,
        dp_protected_col: str = "age",
        mi_threshold: float = 50.0,
        feature_cache: bool = True,
        age_young_max: int = 50,
        age_old_min: int = 60,
    ):
        """
        PTB-XL ECG dataset: MI (y=1) vs NORM (y=0) binary classification.

        Protected attribute (dp_protected_col):
          "age"  (default) — young (<age_young_max) vs old (>=age_old_min);
                  records in [age_young_max, age_old_min) are DROPPED.
                  Young (a=0) is disadvantaged: natural MI rate ~8% vs ~47% old.
                  Cosine similarity of MI discriminant vectors = 0.627 (strong group divergence).
          "sex"  — 0=male (advantaged), 1=female (disadvantaged). Natural MI
                  rates are similar; requires heavy bias injection.

        Feature extraction: 8 per-lead statistics (mean, std, min, max, rms,
        p25, p75, iqr) across all 12 leads → 96 features total.
        Signals loaded at 100 Hz from records100/ using wfdb.

        Pipeline mirrors split_census_income: split → bias per split →
        fit scaler/PCA on train only → tensors.
        """
        import wfdb

        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1
        assert dp_protected_col in ("age", "sex"), \
            f"ptb_xl dp_protected_col must be 'age' or 'sex', got '{dp_protected_col}'"

        data_dir = Path(self.data_path)
        db_path = data_dir / "ptbxl_database.csv"
        scp_path = data_dir / "scp_statements.csv"

        if not db_path.exists():
            raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")
        if not scp_path.exists():
            raise FileNotFoundError(f"scp_statements.csv not found at {scp_path}")

        # ---- 1) Load metadata and classify labels ----
        db = pd.read_csv(db_path)
        scp_stmts = pd.read_csv(scp_path, index_col=0)

        mi_codes = set(scp_stmts[scp_stmts["diagnostic_class"] == "MI"].index.tolist())
        norm_codes = set(scp_stmts[scp_stmts["diagnostic_class"] == "NORM"].index.tolist())

        def _classify(scp_str):
            try:
                codes = ast.literal_eval(scp_str)
            except (ValueError, SyntaxError):
                return -1
            is_mi   = any(codes.get(c, 0) >= mi_threshold for c in mi_codes)
            is_norm = any(codes.get(c, 0) >= mi_threshold for c in norm_codes)
            if is_mi and not is_norm:
                return 1
            if is_norm and not is_mi:
                return 0
            return -1  # ambiguous → discard

        db["y"] = db["scp_codes"].apply(_classify)
        db = db[db["y"] >= 0].reset_index(drop=True)

        # ---- Protected attribute assignment ----
        if dp_protected_col == "age":
            # Drop records with invalid ages or in the middle age band [age_young_max, age_old_min)
            db = db[db["age"] < 120].reset_index(drop=True)  # remove invalid ages
            young_mask = db["age"] < age_young_max
            old_mask   = db["age"] >= age_old_min
            db = db[young_mask | old_mask].reset_index(drop=True)
            db["a"] = 0  # young = disadvantaged
            db.loc[db["age"] >= age_old_min, "a"] = 1  # old = advantaged
            print(f"[ptb_xl] Age groups: young(<{age_young_max})={int(young_mask.sum())} "
                  f"old(>={age_old_min})={int(old_mask.sum())} (dropped middle-age band)")
        else:
            # sex: 0=male (advantaged), 1=female (disadvantaged)
            db["a"] = db["sex"].astype(int)

        print(f"[ptb_xl] After MI/NORM filter: {len(db)} records "
              f"(MI={int((db['y']==1).sum())}, NORM={int((db['y']==0).sum())})")
        grp0_label = f"young(<{age_young_max})" if dp_protected_col == "age" else "male"
        grp1_label = f"old(>={age_old_min})"   if dp_protected_col == "age" else "female"
        print(f"[ptb_xl] Group0={grp0_label}: {int((db['a']==0).sum())}, "
              f"Group1={grp1_label}: {int((db['a']==1).sum())}")

        # ---- 2) Extract ECG features ----
        LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        STAT_NAMES = ["mean","std","min","max","rms","p25","p75","iqr"]
        feature_names = [f"{lead}_{stat}" for lead in LEAD_NAMES for stat in STAT_NAMES]

        cache_path = data_dir / "ptbxl_features_cache.npz"
        if feature_cache and cache_path.exists():
            cache = np.load(cache_path, allow_pickle=True)
            cached_ids = cache["ecg_ids"].tolist()
            cached_X   = cache["X"]
            id_to_row  = {eid: i for i, eid in enumerate(cached_ids)}
            valid_mask = db["ecg_id"].isin(set(cached_ids))
            db = db[valid_mask].reset_index(drop=True)
            row_indices = [id_to_row[eid] for eid in db["ecg_id"]]
            X_raw = cached_X[row_indices]
            print(f"[ptb_xl] Loaded {len(db)} features from cache.")
        else:
            def _extract_features(filename_lr):
                record_path = str(data_dir / filename_lr)
                try:
                    record = wfdb.rdrecord(record_path)
                except Exception:
                    return None
                sig = record.p_signal  # shape (samples, 12)
                if sig is None or sig.shape[1] != 12:
                    return None
                feats = []
                for lead_idx in range(12):
                    x = sig[:, lead_idx]
                    x = x[~np.isnan(x)]
                    if len(x) == 0:
                        feats.extend([0.0] * 8)
                        continue
                    feats.append(float(np.mean(x)))
                    feats.append(float(np.std(x)))
                    feats.append(float(np.min(x)))
                    feats.append(float(np.max(x)))
                    feats.append(float(np.sqrt(np.mean(x ** 2))))
                    feats.append(float(np.percentile(x, 25)))
                    feats.append(float(np.percentile(x, 75)))
                    feats.append(float(np.percentile(x, 75) - np.percentile(x, 25)))
                return feats

            print(f"[ptb_xl] Extracting features from {len(db)} records...")
            all_feats = []
            valid_mask = []
            for _, row in db.iterrows():
                f = _extract_features(row["filename_lr"])
                if f is not None:
                    all_feats.append(f)
                    valid_mask.append(True)
                else:
                    all_feats.append([0.0] * 96)
                    valid_mask.append(False)

            valid_mask = np.array(valid_mask)
            db = db[valid_mask].reset_index(drop=True)
            X_raw = np.array([all_feats[i] for i in range(len(valid_mask)) if valid_mask[i]], dtype=np.float32)

            if feature_cache:
                np.savez(cache_path, ecg_ids=db["ecg_id"].to_numpy(), X=X_raw)
                print(f"[ptb_xl] Feature cache saved to {cache_path}")

        y_all = db["y"].to_numpy(dtype=int)
        a_all = db["a"].to_numpy(dtype=int)
        fold_all = db["strat_fold"].to_numpy(dtype=int)

        print(f"[ptb_xl] Feature matrix: {X_raw.shape}, "
              f"MI={int((y_all==1).sum())}, NORM={int((y_all==0).sum())}")

        # ---- 3) Strat-fold split: folds 1-6 = train, 7-8 = val, 9-10 = test ----
        # Uses the official PTB-XL fold structure to prevent patient-level leakage.
        # val_frac/test_frac are ignored (splits are fixed by strat_fold).
        X_df = pd.DataFrame(X_raw, columns=feature_names)
        X_df["__fold__"] = fold_all
        A_df = pd.Series(a_all, name="sex")
        y_series = pd.Series(y_all)

        tr_mask = (fold_all <= 6)
        va_mask = (fold_all == 7) | (fold_all == 8)
        te_mask = (fold_all == 9) | (fold_all == 10)

        X_train_df = X_df[tr_mask].drop(columns=["__fold__"]).reset_index(drop=True)
        X_val_df   = X_df[va_mask].drop(columns=["__fold__"]).reset_index(drop=True)
        X_test_df  = X_df[te_mask].drop(columns=["__fold__"]).reset_index(drop=True)
        y_train = y_all[tr_mask]
        y_val   = y_all[va_mask]
        y_test  = y_all[te_mask]
        A_train = A_df[tr_mask].reset_index(drop=True)
        A_val   = A_df[va_mask].reset_index(drop=True)
        A_test  = A_df[te_mask].reset_index(drop=True)

        # ---- 4) Bias injection (group-specific: reduce only disadvantaged-group MI) ----
        # For age-based:  disadvantaged = young (a=0); old (a=1) kept at natural rate.
        # For sex-based:  disadvantaged = female (a=1); male (a=0) kept at natural rate.
        # target_pct = target positive rate *within* the disadvantaged subgroup.
        disadv_code = 0 if dp_protected_col == "age" else 1

        def apply_bias(df_split, y_split, a_split, target_pct):
            df = df_split.copy()
            df["__y__"] = y_split
            df["__a__"] = a_split.to_numpy() if hasattr(a_split, "to_numpy") else a_split

            is_disadv = df["__a__"] == disadv_code
            df_adv    = df[~is_disadv]
            df_dis    = df[is_disadv]
            df_d_neg  = df_dis[df_dis["__y__"] == 0]
            df_d_pos  = df_dis[df_dis["__y__"] == 1]

            if len(df_d_neg) == 0 or len(df_d_pos) == 0:
                out = df
            else:
                keep = int(np.floor((target_pct * len(df_d_neg)) / (1 - target_pct)))
                keep = min(len(df_d_pos), max(1, keep))
                df_d_pos_b = df_d_pos.sample(n=keep, random_state=self.seed, replace=False)
                out = (pd.concat([df_adv, df_d_neg, df_d_pos_b])
                         .sample(frac=1.0, random_state=self.seed)
                         .reset_index(drop=True))
            y_out = out["__y__"].to_numpy(dtype=int)
            a_out = out["__a__"].to_numpy(dtype=np.int64)
            X_out = out.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        if bias_pct is not None:
            X_train_b, y_train_b, a_train = apply_bias(X_train_df, y_train, A_train, bias_pct)
            if bias_val:
                X_val_b, y_val_b, a_val = apply_bias(X_val_df, y_val, A_val, bias_pct)
            else:
                X_val_b   = X_val_df.copy().reset_index(drop=True)
                y_val_b   = y_val.copy()
                a_val     = A_val.to_numpy(dtype=np.int64)
        else:
            X_train_b = X_train_df.copy().reset_index(drop=True)
            y_train_b = y_train.copy()
            a_train   = A_train.to_numpy(dtype=np.int64)
            X_val_b   = X_val_df.copy().reset_index(drop=True)
            y_val_b   = y_val.copy()
            a_val     = A_val.to_numpy(dtype=np.int64)

        X_test_b = X_test_df.copy().reset_index(drop=True)
        y_test_b = y_test.copy()
        a_test   = A_test.to_numpy(dtype=np.int64)

        # ---- Optional: subsample train ----
        if train_size is not None and train_size < len(X_train_b):
            X_train_b, _, y_train_b, _, a_train, _ = train_test_split(
                X_train_b, y_train_b, a_train,
                train_size=train_size, random_state=self.seed, stratify=y_train_b
            )

        # ---- 5) Fit scaler on train only ----
        scaler = StandardScaler()
        Xtr_z  = scaler.fit_transform(X_train_b.values)
        Xva_z  = scaler.transform(X_val_b.values)
        Xte_z  = scaler.transform(X_test_b.values)

        # ---- 6) PCA (if enabled) ----
        X_train_theta_np, X_val_theta_np, X_test_theta_np, pca = self._to_theta(
            Xtr_z, Xva_z, Xte_z, pca_components=pca_components
        )

        # ---- 7) Tensors ----
        X_train_theta = torch.tensor(X_train_theta_np, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_val_theta_np,   dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_theta_np,  dtype=torch.float32, device=self.device)

        y_train_theta = torch.tensor(y_train_b, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_b,   dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_b,  dtype=torch.long, device=self.device)

        self.a_train = torch.tensor(a_train, dtype=torch.long, device=self.device)
        self.a_val   = torch.tensor(a_val,   dtype=torch.long, device=self.device)
        self.a_test  = torch.tensor(a_test,  dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col

        # ---- Logging ----
        def _log(name, ysplit, asplit):
            n = len(ysplit)
            n1 = int(np.sum(ysplit == 1))
            n_g0 = int(np.sum(asplit == 0))
            n_g0_pos = int(np.sum((ysplit == 1) & (asplit == 0)))
            n_g1_pos = int(np.sum((ysplit == 1) & (asplit == 1)))
            print(f"[{name}] size={n}, MI={n1} ({100.*n1/n:.1f}%), "
                  f"{grp0_label}={n_g0}, {grp0_label}_MI={n_g0_pos}, {grp1_label}_MI={n_g1_pos}")

        _log("TRAIN", y_train_b, a_train)
        _log("VAL",   y_val_b,   a_val)
        _log("TEST",  y_test_b,  a_test)
        print(f"use_pca={self.use_pca} | theta_dim={X_train_theta.shape[1]}")

        # Cache GAN view
        try:
            self._gan_view_cache = {
                "supported": True,
                "X_train_unbiased_df": X_train_df.copy(),
                "y_train_unbiased": y_train.astype(int).copy(),
                "encoder": None,
                "scaler": scaler,
                "pca": pca,
                "use_pca": self.use_pca,
                "pca_components": int(pca_components),
                "cat_cols": [],
                "num_cols": feature_names,
            }
        except Exception:
            pass

        if return_test_df:
            return (
                X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta,
                X_test_b.copy()
            )
        if return_raw:
            return (
                X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta,
                X_test_b.copy(), y_test_b.copy()
            )
        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta

    def split_capture24(
        self,
        train_size=None,
        val_size=None,
        bias_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=10,
        return_test_df: bool = False,
        return_raw: bool = False,
        bias_val: bool = True,
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

        # ---- Bias injection -------------------------------------------------
        def apply_bias(df_split, y_split, a_split, target_pct):
            df = df_split.copy()
            df["__y__"] = y_split
            df["__a__"] = a_split.to_numpy() if hasattr(a_split, "to_numpy") else a_split
            maj  = df[df["__y__"] == 0]
            mino = df[df["__y__"] == 1]
            if len(maj) == 0 or len(mino) == 0:
                out = df
            else:
                keep = int(np.floor((target_pct * len(maj)) / (1 - target_pct)))
                keep = min(len(mino), max(1, keep))
                mino_b = mino.sample(n=keep, random_state=self.seed, replace=False)
                out = (pd.concat([maj, mino_b])
                         .sample(frac=1.0, random_state=self.seed)
                         .reset_index(drop=True))
            y_out = out["__y__"].to_numpy(dtype=int)
            a_out = out["__a__"].to_numpy(dtype=np.int64)
            X_out = out.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        if bias_pct is not None:
            X_train_b, y_train_b, a_train = apply_bias(X_train_df, y_train, A_train, bias_pct)
            if bias_val:
                X_val_b, y_val_b, a_val = apply_bias(X_val_df, y_val, A_val, bias_pct)
            else:
                X_val_b = X_val_df.copy().reset_index(drop=True)
                y_val_b = y_val.copy()
                a_val   = A_val.to_numpy(dtype=np.int64)
        else:
            X_train_b = X_train_df.copy().reset_index(drop=True)
            y_train_b = y_train.copy()
            a_train   = A_train.to_numpy(dtype=np.int64)
            X_val_b   = X_val_df.copy().reset_index(drop=True)
            y_val_b   = y_val.copy()
            a_val     = A_val.to_numpy(dtype=np.int64)

        X_test_b = X_test_df.copy().reset_index(drop=True)
        y_test_b = y_test.copy()
        a_test   = A_test.to_numpy(dtype=np.int64)

        # ---- Optional: subsample train --------------------------------------
        if train_size is not None and train_size < len(X_train_b):
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

        if return_test_df:
            return (X_train_theta, X_val_theta, X_test_theta,
                    y_train_theta, y_val_theta, y_test_theta,
                    X_test_b.copy())
        if return_raw:
            return (X_train_theta, X_val_theta, X_test_theta,
                    y_train_theta, y_val_theta, y_test_theta,
                    X_test_b.copy(), y_test_b.copy())
        return (X_train_theta, X_val_theta, X_test_theta,
                y_train_theta, y_val_theta, y_test_theta)

    def split_fairjob(
        self,
        train_size=None,
        bias_pct=None,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=10,
        drop_protected: bool = False,
        protected_cols=None,
        return_test_df: bool = False,
        return_raw: bool = False,
        bias_val: bool = True,
        dp_protected_col: str = "protected_attribute",
        filter_display_random: bool = True,
        pool_pos_fraction: float = None,
    ):
        """
        FairJob dataset (Vladimirova et al., NeurIPS 2024 / Criteo).
        Source: HuggingFace criteo/FairJob — downloaded to datasets/fairjob/
        on first call and cached as fairjob.parquet.

        Task: binary click prediction on job listings.
        Label: 'click' (0=no click, 1=click). Natural click rate ~0.7%.
        Protected attr: 'protected_attribute' (0=female disadv, 1=male adv).
          Gender proxy inferred from user's fashion-product history.

        pool_pos_fraction: if set (e.g. 0.05), undersample negatives in train
          and val AFTER the split so that positive examples make up this
          fraction of those splits. The test set is left at natural (~0.7%)
          distribution so AUC/F1 evaluation remains meaningful. This avoids
          needing a very large real_data_size to accumulate enough positives.
          With pool_pos_fraction=0.05 and real_data_size=3000 the train pool
          has ~5% positives, giving ~150 positives to bias_pct down from.

        Filter: displayrandom=1 (paper recommendation to remove positional
        bias from rank-based display). Controlled by filter_display_random.

        Categorical features (cat0–cat12) are one-hot encoded; numerical
        features (num16–num50, senior, rank) are StandardScaler-normalised.
        protected_attribute is included as a numerical feature by default and
        dropped if drop_protected=True.

        Bias injection is group-specific: only female (a=0) positives are
        downsampled, keeping male positives at their natural rate.

        Pipeline: filter → split → neg-undersample train+val (optional) →
        group-specific bias → OHE + scaler (train only) → PCA → tensors.
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1

        if protected_cols is None:
            protected_cols = ["protected_attribute"]

        data_dir = Path(self.data_path)
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / "fairjob.parquet"

        # ---- 1) Load (local cache or HuggingFace download) ----
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            print(f"[fairjob] Loaded from cache: {len(df)} rows")
        else:
            print("[fairjob] Downloading from HuggingFace (criteo/FairJob)…")
            try:
                from datasets import load_dataset as hf_load
            except ImportError:
                raise ImportError(
                    "HuggingFace 'datasets' library required. "
                    "Install with: pip install datasets"
                )
            hf_ds = hf_load("criteo/FairJob", split="train")
            df = hf_ds.to_pandas()
            df.to_parquet(parquet_path, index=False)
            print(f"[fairjob] Cached to {parquet_path}: {len(df)} rows")

        # ---- 2) Optional: restrict to randomised display ----
        if filter_display_random and "displayrandom" in df.columns:
            df = df[df["displayrandom"] == 1].reset_index(drop=True)
            print(f"[fairjob] After displayrandom=1 filter: {len(df)} rows")

        # ---- 3) Label and feature schema ----
        label_col = "click"
        y_raw = df[label_col].astype(int).to_numpy()
        vals, cnts = np.unique(y_raw, return_counts=True)
        print(f"[fairjob] Label dist: {dict(zip(vals.tolist(), cnts.tolist()))}, "
              f"click_rate={y_raw.mean():.4f}")

        cat_cols_all = [f"cat{i}" for i in range(13)]            # cat0–cat12
        num_cols_all = (
            [f"num{i}" for i in range(16, 51)]                   # num16–num50
            + ["senior", "rank", "protected_attribute"]           # extra numerics
        )
        # Keep only columns present in the dataframe
        cat_cols_all = [c for c in cat_cols_all if c in df.columns]
        num_cols_all = [c for c in num_cols_all if c in df.columns]

        feature_cols = num_cols_all + cat_cols_all
        X_df_raw = df[feature_cols].copy()
        A_df_raw = df[[dp_protected_col]].copy()

        # Fill missing values (defensive; FairJob is typically clean)
        for c in cat_cols_all:
            if c in X_df_raw.columns:
                X_df_raw[c] = X_df_raw[c].fillna("__missing__").astype(str)
        for c in num_cols_all:
            if c in X_df_raw.columns:
                X_df_raw[c] = X_df_raw[c].fillna(0.0)

        # ---- 4) Optional: drop protected attribute from features ----
        if drop_protected:
            drop_c = [c for c in protected_cols if c in X_df_raw.columns]
            if drop_c:
                X_df_raw = X_df_raw.drop(columns=drop_c)

        cat_cols = [c for c in cat_cols_all if c in X_df_raw.columns]
        num_cols = [c for c in num_cols_all if c in X_df_raw.columns]

        # ---- 5) Split ----
        X_train_df, X_temp_df, y_train, y_temp, A_train_df, A_temp_df = train_test_split(
            X_df_raw, y_raw, A_df_raw,
            test_size=(val_frac + test_frac),
            random_state=self.seed,
            stratify=y_raw,
        )
        rel_test = test_frac / (val_frac + test_frac)
        X_val_df, X_test_df, y_val, y_test, A_val_df, A_test_df = train_test_split(
            X_temp_df, y_temp, A_temp_df,
            test_size=rel_test,
            random_state=self.seed,
            stratify=y_temp,
        )

        # ---- 5b) Optional: enrich train+val by undersampling negatives ----
        # Test split is intentionally left at natural distribution.
        if pool_pos_fraction is not None:
            assert 0 < pool_pos_fraction < 1
            def _neg_undersample(X_df, y_arr, A_df):
                pos_idx = np.where(y_arr == 1)[0]
                neg_idx = np.where(y_arr == 0)[0]
                n_pos = len(pos_idx)
                if n_pos == 0:
                    return X_df, y_arr, A_df
                n_neg_keep = int(np.floor(n_pos * (1.0 - pool_pos_fraction) / pool_pos_fraction))
                n_neg_keep = min(len(neg_idx), max(1, n_neg_keep))
                rng = np.random.default_rng(self.seed)
                kept_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
                keep = np.sort(np.concatenate([pos_idx, kept_neg]))
                return (X_df.iloc[keep].reset_index(drop=True),
                        y_arr[keep],
                        A_df.iloc[keep].reset_index(drop=True))

            X_train_df, y_train, A_train_df = _neg_undersample(X_train_df, y_train, A_train_df)
            print(f"[fairjob] Train after neg-undersample: {len(y_train)} rows, "
                  f"pos_rate={y_train.mean():.4f}")
            X_val_df, y_val, A_val_df = _neg_undersample(X_val_df, y_val, A_val_df)
            print(f"[fairjob] Val after neg-undersample: {len(y_val)} rows, "
                  f"pos_rate={y_val.mean():.4f}")

        # Protected attribute: already binary (0=female/disadv, 1=male/adv)
        def _map_protected(a_series):
            return a_series.astype(np.int64).to_numpy()

        # ---- 6) Group-specific bias: reduce female (a=0) positives only ----
        def apply_bias(df_split, y_split, a_split_df, target_pct):
            """
            target_pct = target positive rate *within* the female (a=0) subgroup.
            Male positives are left at their natural rate.
            """
            dfb = df_split.copy()
            dfb["__y__"] = y_split
            dfb["__a__"] = a_split_df[dp_protected_col].to_numpy()

            df_male   = dfb[dfb["__a__"] != 0]
            df_female = dfb[dfb["__a__"] == 0]

            f_neg = df_female[df_female["__y__"] == 0]
            f_pos = df_female[df_female["__y__"] == 1]

            if len(f_pos) == 0 or len(f_neg) == 0:
                df_female_out = df_female
            else:
                keep = int(np.floor((target_pct * len(f_neg)) / (1.0 - target_pct)))
                keep = min(len(f_pos), max(1, keep))
                f_pos_b = f_pos.sample(n=keep, random_state=self.seed, replace=False)
                df_female_out = pd.concat([f_neg, f_pos_b], axis=0)

            out = (pd.concat([df_male, df_female_out], axis=0)
                   .sample(frac=1.0, random_state=self.seed)
                   .reset_index(drop=True))

            y_out = out["__y__"].to_numpy(dtype=int)
            a_out = _map_protected(out["__a__"])
            X_out = out.drop(columns=["__y__", "__a__"])
            return X_out, y_out, a_out

        if bias_pct is not None:
            X_train_biased_df, y_train_biased, a_train = apply_bias(
                X_train_df, y_train, A_train_df, float(bias_pct))
            if bias_val:
                X_val_biased_df, y_val_biased, a_val = apply_bias(
                    X_val_df, y_val, A_val_df, float(bias_pct))
            else:
                X_val_biased_df = X_val_df.copy().reset_index(drop=True)
                y_val_biased = y_val.copy()
                a_val = _map_protected(A_val_df[dp_protected_col])
        else:
            X_train_biased_df = X_train_df.copy().reset_index(drop=True)
            y_train_biased = y_train.copy()
            a_train = _map_protected(A_train_df[dp_protected_col])
            X_val_biased_df = X_val_df.copy().reset_index(drop=True)
            y_val_biased = y_val.copy()
            a_val = _map_protected(A_val_df[dp_protected_col])

        X_test_biased_df = X_test_df.copy().reset_index(drop=True)
        y_test_biased = y_test.copy()
        a_test = _map_protected(A_test_df[dp_protected_col])

        # ---- Optional train subsample ----
        if train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _, a_train, _ = train_test_split(
                X_train_biased_df, y_train_biased, a_train,
                train_size=train_size,
                random_state=self.seed,
                stratify=y_train_biased,
            )

        # ---- 7) Fit OHE + scaler on train only ----
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        scaler = StandardScaler()

        Xtr_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if cat_cols else np.empty((len(X_train_biased_df), 0))
        Xtr_num = scaler.fit_transform(X_train_biased_df[num_cols])  if num_cols else np.empty((len(X_train_biased_df), 0))
        Xtr_all = np.hstack([Xtr_num, Xtr_cat])

        Xva_cat = encoder.transform(X_val_biased_df[cat_cols]) if cat_cols else np.empty((len(X_val_biased_df), 0))
        Xva_num = scaler.transform(X_val_biased_df[num_cols])  if num_cols else np.empty((len(X_val_biased_df), 0))
        Xva_all = np.hstack([Xva_num, Xva_cat])

        Xte_cat = encoder.transform(X_test_biased_df[cat_cols]) if cat_cols else np.empty((len(X_test_biased_df), 0))
        Xte_num = scaler.transform(X_test_biased_df[num_cols])  if num_cols else np.empty((len(X_test_biased_df), 0))
        Xte_all = np.hstack([Xte_num, Xte_cat])

        # ---- 8) PCA ----
        X_train_theta_np, X_val_theta_np, X_test_theta_np, pca = self._to_theta(
            Xtr_all, Xva_all, Xte_all, pca_components=pca_components
        )

        # ---- 9) Tensors ----
        X_train_theta = torch.tensor(X_train_theta_np, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_val_theta_np,   dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_theta_np,  dtype=torch.float32, device=self.device)

        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_biased,   dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_biased,  dtype=torch.long, device=self.device)

        self.a_train = torch.tensor(a_train, dtype=torch.long, device=self.device)
        self.a_val   = torch.tensor(a_val,   dtype=torch.long, device=self.device)
        self.a_test  = torch.tensor(a_test,  dtype=torch.long, device=self.device)
        self.dp_protected_col = dp_protected_col

        def _log(name, y_s, a_s):
            n = len(y_s)
            f_pos = int(((y_s == 1) & (a_s == 0)).sum())
            m_pos = int(((y_s == 1) & (a_s == 1)).sum())
            print(f"[fairjob/{name}] n={n}, DA+(female)={f_pos}, AA+(male)={m_pos}")
        _log("TRAIN", y_train_biased, a_train)
        _log("VAL",   y_val_biased,   a_val)
        _log("TEST",  y_test_biased,  a_test)

        self._gan_view_cache = {
            "supported": True,
            "X_train_unbiased_df": X_train_df.copy(),
            "y_train_unbiased": y_train.astype(int).copy(),
            "encoder": encoder,
            "scaler": scaler,
            "pca": pca,
            "use_pca": self.use_pca,
            "pca_components": int(pca_components),
            "cat_cols": list(cat_cols),
            "num_cols": list(num_cols),
            "drop_protected": bool(drop_protected),
            "protected_cols": list(protected_cols),
            "label_col": label_col,
            "dp_protected_col": dp_protected_col,
        }

        if return_test_df:
            return (X_train_theta, X_val_theta, X_test_theta,
                    y_train_theta, y_val_theta, y_test_theta,
                    X_test_biased_df.copy())
        if return_raw:
            return (X_train_theta, X_val_theta, X_test_theta,
                    y_train_theta, y_val_theta, y_test_theta,
                    X_test_biased_df.copy(), y_test_biased.copy())

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta


    def get_data_splits(self, **kwargs):
        _tabular_drop = ("win_seconds", "step_seconds")
        if self.dataset_name == "census_income":
            kw = {k: v for k, v in kwargs.items() if k not in _tabular_drop}
            return self.split_census_income(**kw)
        elif self.dataset_name == "pamap2":
            # split_pamap2 doesn't accept drop_protected / protected_cols
            pamap_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ("drop_protected", "protected_cols")}
            return self.split_pamap2(**pamap_kwargs)
        elif self.dataset_name == "credit_card":
            kw = {k: v for k, v in kwargs.items() if k not in _tabular_drop}
            return self.split_credit_card(**kw)
        elif self.dataset_name == "ptb_xl":
            ptbxl_kwargs = {k: v for k, v in kwargs.items()
                            if k not in ("drop_protected", "protected_cols",
                                         "win_seconds", "step_seconds")}
            return self.split_ptb_xl(**ptbxl_kwargs)
        elif self.dataset_name == "capture24":
            c24_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ("drop_protected", "protected_cols",
                                       "win_seconds", "step_seconds")}
            return self.split_capture24(**c24_kwargs)
        elif self.dataset_name == "compas":
            kw = {k: v for k, v in kwargs.items() if k not in _tabular_drop}
            return self.split_compas(**kw)
        elif self.dataset_name == "meps":
            kw = {k: v for k, v in kwargs.items() if k not in _tabular_drop}
            return self.split_meps(**kw)
        elif self.dataset_name == "acs_income":
            _acs_drop = ("drop_protected", "protected_cols", "win_seconds", "step_seconds")
            kw = {k: v for k, v in kwargs.items() if k not in _acs_drop}
            return self.split_acs_income(**kw)
        elif self.dataset_name == "fairjob":
            kw = {k: v for k, v in kwargs.items() if k not in _tabular_drop}
            return self.split_fairjob(**kw)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


    #Check if needed still
    def rebuild_original_train_pool_theta(
        self, *, bias_pct: float | None, val_frac: float, test_frac: float,
        train_size: int | None, pca_components: int, device: torch.device,
        include_third_label: bool = False, third_activity_id: int = 13,
    ):
        """
        Build the ORIGINAL (unbiased) TRAIN pool mapped into θ-space that matches
        the experiment (i.e., scaler/PCA fitted on TRAIN-biased).
        Returns (X_pool_theta, y_pool_theta) as torch tensors on `device`.

        PAMAP2:
        - Binary (default): keep {4 (walking)->0, 24 (rope)->1}
        - Multiclass (include_third_label=True): keep {4->0, 24->1, third_activity_id->2}
        - Biasing always targets rope (class 1) vs the combined non-rope pool.
        """
        if self.dataset_name == "census_income":
            # ---- Adult (Census) ----
            column_names = [
                "age","workclass","fnlwgt","education","education-num","marital-status",
                "occupation","relationship","race","sex","capital-gain","capital-loss",
                "hours-per-week","native-country","income"
            ]
            X_df_raw = pd.read_csv(self.data_path, header=None, names=column_names,
                                na_values="?", skipinitialspace=True)
            y_raw = np.where(X_df_raw["income"].isin([">50K", ">50K."]), 1, 0).astype(int)
            X_df_raw = X_df_raw.drop(columns=["income"])

            cat_cols = [c for c in X_df_raw.columns if X_df_raw[c].dtype.name in ["category","object","bool"]]
            num_cols = [c for c in X_df_raw.columns if np.issubdtype(X_df_raw[c].dtype, np.number)]

            X_train_df, X_temp_df, y_train, y_temp = train_test_split(
                X_df_raw, y_raw, test_size=(val_frac + test_frac),
                random_state=self.seed, stratify=y_raw
            )
            rel_test = test_frac / (val_frac + test_frac)
            _X_val_df, _X_test_df, _y_val, _y_test = train_test_split(
                X_temp_df, y_temp, test_size=rel_test,
                random_state=self.seed, stratify=y_temp
            )

            def apply_bias(df_split, y_split, target_minority_pct):
                df = df_split.copy(); df["__y__"] = y_split
                maj = df[df["__y__"] == 0]; mino = df[df["__y__"] == 1]
                if len(maj)==0 or len(mino)==0:
                    out = df
                else:
                    keep_min = int(np.floor((target_minority_pct * len(maj)) / (1 - target_minority_pct)))
                    keep_min = min(len(mino), max(1, keep_min))
                    mino_b = mino.sample(n=keep_min, random_state=self.seed, replace=False)
                    out = pd.concat([maj, mino_b], axis=0).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
                y_out = out["__y__"].to_numpy(dtype=int)
                X_out = out.drop(columns=["__y__"])
                return X_out, y_out

            # TRAIN-biased for fitting transforms (skip bias if bias_pct is None)
            if bias_pct is not None:
                X_train_biased_df, y_train_biased = apply_bias(X_train_df, y_train, bias_pct)
            else:
                X_train_biased_df, y_train_biased = X_train_df.copy(), y_train.copy()
            if train_size is not None and train_size < len(X_train_biased_df):
                X_train_biased_df, _, y_train_biased, _ = train_test_split(
                    X_train_biased_df, y_train_biased,
                    train_size=train_size, random_state=self.seed, stratify=y_train_biased
                )

            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            scaler = StandardScaler()
            Xtr_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_biased_df),0))
            Xtr_num = scaler.fit_transform(X_train_biased_df[num_cols])   if len(num_cols) else np.empty((len(X_train_biased_df),0))
            Xtr_all = np.hstack([Xtr_num, Xtr_cat])

            Xpool_cat = encoder.transform(X_train_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_df),0))
            Xpool_num = scaler.transform(X_train_df[num_cols])  if len(num_cols) else np.empty((len(X_train_df),0))
            Xpool_all = np.hstack([Xpool_num, Xpool_cat])

            # Fit/identity on biased train, then transform pool consistently
            _, _, Xpool_theta_np, pca = self._to_theta(
                Xtr_all, Xtr_all, Xpool_all,
                pca_components=pca_components
            )

            X_pool_theta = torch.tensor(Xpool_theta_np, dtype=torch.float32, device=device)

            y_pool_theta = torch.tensor(y_train,   dtype=torch.long,   device=device)
            return X_pool_theta, y_pool_theta

        elif self.dataset_name == "pamap2":
            # ---- PAMAP2 ----
            rng = np.random.RandomState(self.seed)

            # constants — must match the win_seconds/step_seconds used in split_pamap2
            FS = 100
            win_seconds = getattr(self, '_win_seconds', 5.0)
            step_seconds = getattr(self, '_step_seconds', 2.5)
            WIN  = int(win_seconds * FS)
            STEP = int(step_seconds * FS)
            imu_pos = ["hand","chest","ankle"]
            ACC16 = ["acc16g_x","acc16g_y","acc16g_z"]
            GYR   = ["gyro_x","gyro_y","gyro_z"]
            drop_magnetometers = True
            use_vector_norms   = True
            stats = ("std","rms")

            data_dir = Path(self.data_path)
            def colnames():
                cols = ["timestamp","activity_id","heart_rate"]
                sub  = ["temp","acc16g_x","acc16g_y","acc16g_z","acc6g_x","acc6g_y","acc6g_z",
                        "gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z",
                        "orient_w","orient_x","orient_y","orient_z"]
                for p in imu_pos:
                    cols += [f"{p}_{s}" for s in sub]
                return cols

            def parse_sid(path):
                m = re.search(r"subject(\d+)", Path(path).stem.lower())
                return m.group(1) if m else "unknown"

            files = []
            for sub in ["Protocol","Optional","protocol","optional"]:
                d = data_dir / sub
                if d.exists():
                    files += list(d.glob("subject*.dat"))
            if not files:
                raise FileNotFoundError(f"No .dat files found under {data_dir}")

            dfs = []
            for f in sorted(files):
                df = pd.read_csv(f, sep=r"\s+", header=None, names=colnames(),
                                engine="python", na_values=["NaN","nan"])
                df["subject_id"] = parse_sid(f)
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=True)

            # keep activities (binary or 3-class)
            keep_ids = [self.MAJORITY_ID, self.MINORITY_ID] if not include_third_label else [self.MAJORITY_ID, self.MINORITY_ID, third_activity_id]
            df = df[df["activity_id"].isin(keep_ids)].copy()
            mapping = {self.MAJORITY_ID: 0, self.MINORITY_ID: 1}
            if include_third_label:
                mapping[third_activity_id] = 2
            df["y"] = df["activity_id"].map(mapping)

            # per-subject interpolation
            num = df.select_dtypes(include=[np.number]).columns.tolist()
            num = [c for c in num if c not in ("activity_id","y")]
            df = df.sort_values(["subject_id","timestamp"]).groupby("subject_id", group_keys=False).apply(
                lambda g: g.assign(**{c: g[c].interpolate(limit_direction="both") for c in num})
            )

            # base channels
            keep_triplets = [("acc", ACC16), ("gyr", GYR)]
            if not drop_magnetometers:
                keep_triplets.append(("mag", ["mag_x","mag_y","mag_z"]))
            base_cols = ["heart_rate"]
            if use_vector_norms:
                for p in imu_pos:
                    for name, axes in keep_triplets:
                        cols = [f"{p}_{a}" for a in axes]
                        df[f"{p}_{name}_norm"] = np.sqrt((df[cols].values ** 2).sum(axis=1))
                        base_cols.append(f"{p}_{name}_norm")
            else:
                for p in imu_pos:
                    for _, axes in keep_triplets:
                        for a in axes:
                            base_cols.append(f"{p}_{a}")

            # window → features
            def compute_feats(W: pd.DataFrame):
                out = {}
                if "std" in stats:
                    out.update(W.std(ddof=1).add_suffix("__std").to_dict())
                if "rms" in stats:
                    out.update((np.sqrt((W**2).mean())).add_suffix("__rms").to_dict())
                return out

            rows, labels, subjects = [], [], []
            df = df.sort_values(["subject_id","timestamp"]).reset_index(drop=True)
            for sid, g in df.groupby("subject_id", sort=False):
                g = g.reset_index(drop=True); n = len(g)
                for start in range(0, max(0, n - WIN + 1), STEP):
                    w = g.iloc[start:start + WIN]
                    if len(w) < WIN: continue
                    y_win = int(np.bincount(w["y"].astype(int)).argmax())
                    feats = compute_feats(w[base_cols])
                    feats["subject_id"] = sid
                    rows.append(feats); labels.append(y_win); subjects.append(sid)



            feat_df = pd.DataFrame(rows).fillna(0.0)
            y_all   = np.asarray(labels, dtype=int)
            subj_all= np.asarray(subjects)

            # subject-aware split (get UNBIASED TRAIN for the pool)
            unique_subjects = np.unique(subj_all); rng.shuffle(unique_subjects)
            n_subj = len(unique_subjects)
            n_test = max(1, int(round(test_frac * n_subj)))
            n_val  = max(1, int(round(val_frac  * n_subj)))
            if n_test + n_val >= n_subj:
                n_test = max(1, n_test)
                n_val  = max(1, min(n_val, n_subj - n_test - 1))
            test_subj = unique_subjects[:n_test]
            val_subj  = unique_subjects[n_test:n_test + n_val]
            train_subj= unique_subjects[n_test + n_val:]
            m_train = np.isin(subj_all, train_subj)

            X_train_unbiased_df = feat_df[m_train].copy()
            y_train_unbiased    = y_all[m_train]

            # TRAIN-biased (fit transforms) – rope vs non-rope
            def apply_bias_rope(df_split, y_split, target_minority_pct):
                dfb = df_split.copy(); dfb["__y__"] = y_split
                mask_rope = (dfb["__y__"] == 1)
                df_rope, df_nonrope = dfb[mask_rope], dfb[~mask_rope]
                n_nonrope, n_rope = len(df_nonrope), len(df_rope)
                if n_nonrope == 0 or n_rope == 0:
                    out = dfb
                else:
                    n_rope_keep = int(np.floor((target_minority_pct * n_nonrope) / (1 - target_minority_pct)))
                    n_rope_keep = min(n_rope, max(1, n_rope_keep))
                    df_rope_b = df_rope.sample(n=n_rope_keep, random_state=self.seed, replace=False)
                    out = pd.concat([df_nonrope, df_rope_b], axis=0).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
                y_out = out["__y__"].to_numpy(dtype=int)
                X_out = out.drop(columns=["__y__"])
                return X_out, y_out

            if bias_pct is not None:
                X_train_biased_df, y_train_biased = apply_bias_rope(
                    X_train_unbiased_df, y_train_unbiased, float(bias_pct)
                )
            else:
                X_train_biased_df, y_train_biased = X_train_unbiased_df.copy(), y_train_unbiased.copy()
            if train_size is not None and train_size < len(X_train_biased_df):
                X_train_biased_df, _, y_train_biased, _ = train_test_split(
                    X_train_biased_df, y_train_biased,
                    train_size=train_size, random_state=self.seed, stratify=y_train_biased
                )

            # Fit scaler+PCA on TRAIN-biased; transform UNBIASED TRAIN pool
            feature_cols = [c for c in X_train_biased_df.columns if c != "subject_id"]
            scaler = StandardScaler()
            Xtr_z = scaler.fit_transform(X_train_biased_df[feature_cols].values)

            # Transform pool
            Xpool_z = scaler.transform(X_train_unbiased_df[feature_cols].values)

            # Use shared theta mapping (PCA or identity)
            _, _, Xpool_theta_np, pca = self._to_theta(
                Xtr_z, Xtr_z, Xpool_z,
                pca_components=pca_components
            )

            X_pool_theta = torch.tensor(Xpool_theta_np, dtype=torch.float32, device=device)

            y_pool_theta = torch.tensor(y_train_unbiased, dtype=torch.long, device=device)
            return X_pool_theta, y_pool_theta

        elif self.dataset_name == "credit_card":
            # ---- Credit Card Default (.xls with 2 header rows) ----
            if include_third_label:
                raise ValueError("credit_card is binary only; include_third_label must be False.")

            # 1) Load Excel: first row X1..X23,Y; second row true names
            df = pd.read_excel(self.data_path, header=0)

            # second row contains real feature names
            true_names = [str(x).strip() for x in df.iloc[0].tolist()]
            df.columns = true_names
            df = df.iloc[1:].reset_index(drop=True)

            # coerce numeric where possible
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")

            # 2) Identify label column
            label_col = None
            for cand in ["default payment next month", "default payment next month ", "Y"]:
                if cand in df.columns:
                    label_col = cand
                    break
            if label_col is None:
                matches = [c for c in df.columns if "default" in c.lower() and "next" in c.lower()]
                if matches:
                    label_col = matches[0]
                else:
                    raise ValueError(f"Could not find label column in credit_card .xls. Columns: {list(df.columns)}")

            # Drop ID if present
            if "ID" in df.columns:
                df = df.drop(columns=["ID"])

            y_raw = df[label_col].astype(int).to_numpy()
            X_df_raw = df.drop(columns=[label_col])

            uniq = set(np.unique(y_raw).tolist())
            if not uniq.issubset({0, 1}):
                raise ValueError(f"Expected binary labels {{0,1}} but got {sorted(list(uniq))}.")

            # 3) Split raw -> get UNBIASED TRAIN pool (same split logic as others)
            X_train_df, X_temp_df, y_train, y_temp = train_test_split(
                X_df_raw, y_raw, test_size=(val_frac + test_frac),
                random_state=self.seed, stratify=y_raw
            )
            rel_test = test_frac / (val_frac + test_frac)
            _X_val_df, _X_test_df, _y_val, _y_test = train_test_split(
                X_temp_df, y_temp, test_size=rel_test,
                random_state=self.seed, stratify=y_temp
            )

            # 4) Build TRAIN-biased for fitting transforms (minority = 1 = default)
            def apply_bias(df_split, y_split, target_minority_pct):
                dfb = df_split.copy()
                dfb["__y__"] = y_split
                maj = dfb[dfb["__y__"] == 0]
                mino = dfb[dfb["__y__"] == 1]

                if len(maj) == 0 or len(mino) == 0:
                    out = dfb
                else:
                    keep_min = int(np.floor((target_minority_pct * len(maj)) / (1 - target_minority_pct)))
                    keep_min = min(len(mino), max(1, keep_min))
                    mino_b = mino.sample(n=keep_min, random_state=self.seed, replace=False)
                    out = (
                        pd.concat([maj, mino_b], axis=0)
                        .sample(frac=1.0, random_state=self.seed)
                        .reset_index(drop=True)
                    )

                y_out = out["__y__"].to_numpy(dtype=int)
                X_out = out.drop(columns=["__y__"])
                return X_out, y_out

            if bias_pct is not None:
                X_train_biased_df, y_train_biased = apply_bias(X_train_df, y_train, float(bias_pct))
            else:
                X_train_biased_df, y_train_biased = X_train_df.copy(), y_train.copy()

            # optional: subsample TRAIN after biasing
            if train_size is not None and train_size < len(X_train_biased_df):
                X_train_biased_df, _, y_train_biased, _ = train_test_split(
                    X_train_biased_df, y_train_biased,
                    train_size=train_size, random_state=self.seed, stratify=y_train_biased
                )

            # 5) Column typing (categorical by meaning)
            cat_guess = ["SEX", "EDUCATION", "MARRIAGE"]
            cat_cols = [c for c in cat_guess if c in X_train_biased_df.columns]
            num_cols = [c for c in X_train_biased_df.columns if c not in cat_cols]

            # 6) Fit transforms on TRAIN-biased; project UNBIASED TRAIN pool into θ-space
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

            scaler = StandardScaler()
            Xtr_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_biased_df),0))
            Xtr_num = scaler.fit_transform(X_train_biased_df[num_cols])   if len(num_cols) else np.empty((len(X_train_biased_df),0))
            Xtr_all = np.hstack([Xtr_num, Xtr_cat])

            Xpool_cat = encoder.transform(X_train_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_df),0))
            Xpool_num = scaler.transform(X_train_df[num_cols])  if len(num_cols) else np.empty((len(X_train_df),0))
            Xpool_all = np.hstack([Xpool_num, Xpool_cat])

            # Fit/identity on biased train, then transform pool consistently
            _, _, Xpool_theta_np, pca = self._to_theta(
                Xtr_all, Xtr_all, Xpool_all,
                pca_components=pca_components
            )

            X_pool_theta = torch.tensor(Xpool_theta_np, dtype=torch.float32, device=device)

            y_pool_theta = torch.tensor(y_train.astype(int), dtype=torch.long, device=device)
            return X_pool_theta, y_pool_theta



        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _pamap2_build_feature_table(self,
                                    win_seconds=5.0,
                                    step_seconds=2.5,
                                    drop_magnetometers=True,
                                    use_vector_norms=True,
                                    stats=("std","rms")):
        """
        Build the PAMAP2 per-window feature table used by split_pamap2.
        Returns:
        feat_df: DataFrame [n_windows x (features + subject_id)]
        y_all:   np.ndarray[int] labels per window (0=majority, 1=minority[, 2=third if multiclass])
        subj_all: np.ndarray[str] subject ids per window
        feature_cols: list[str] numeric feature columns (EXCLUDES 'subject_id')
        """
        rng = np.random.RandomState(self.seed)

        # constants
        FS = 100
        WIN = int(win_seconds * FS)
        STEP = int(step_seconds * FS)
        data_dir = Path(self.data_path)

        imu_pos = ["hand", "chest", "ankle"]
        ACC16 = ["acc16g_x", "acc16g_y", "acc16g_z"]
        GYR   = ["gyro_x", "gyro_y", "gyro_z"]
        MAG   = ["mag_x", "mag_y", "mag_z"]

        def colnames():
            cols = ["timestamp", "activity_id", "heart_rate"]
            sub = ["temp","acc16g_x","acc16g_y","acc16g_z","acc6g_x","acc6g_y","acc6g_z",
                "gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z",
                "orient_w","orient_x","orient_y","orient_z"]
            for p in imu_pos:
                cols += [f"{p}_{s}" for s in sub]
            return cols

        def parse_sid(path):
            m = re.search(r"subject(\d+)", Path(path).stem.lower())
            return m.group(1) if m else "unknown"

        # load .dat
        files = []
        for sub in ["Protocol","Optional","protocol","optional"]:
            d = data_dir / sub
            if d.exists():
                files += list(d.glob("subject*.dat"))
        if not files:
            raise FileNotFoundError(f"No .dat files found under {data_dir}")

        dfs = []
        for f in sorted(files):
            df = pd.read_csv(f, sep=r"\s+", header=None, names=colnames(),
                            engine="python", na_values=["NaN","nan"])
            df["subject_id"] = parse_sid(f)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # keep target activities
        include_third_label = self.multiclass
        keep_ids = [self.MAJORITY_ID, self.MINORITY_ID] if not include_third_label \
                else [self.MAJORITY_ID, self.MINORITY_ID, self.third_id]
        df = df[df["activity_id"].isin(keep_ids)].copy()

        # map labels: majority->0, minority->1, optional third->2
        mapping = {self.MAJORITY_ID: 0, self.MINORITY_ID: 1}
        if include_third_label:
            mapping[self.third_id] = 2
        df["y"] = df["activity_id"].map(mapping)

        # per-subject interpolation
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        num = [c for c in num if c not in ("activity_id","y")]
        df = df.sort_values(["subject_id","timestamp"]).groupby("subject_id", group_keys=False).apply(
            lambda g: g.assign(**{c: g[c].interpolate(limit_direction="both") for c in num})
        )

        # base channels
        keep_triplets = [("acc", ACC16), ("gyr", GYR)]
        if not drop_magnetometers:
            keep_triplets.append(("mag", MAG))

        base_cols = ["heart_rate"]
        if use_vector_norms:
            for p in ["hand","chest","ankle"]:
                for name, axes in keep_triplets:
                    cols = [f"{p}_{a}" for a in axes]
                    df[f"{p}_{name}_norm"] = np.sqrt((df[cols].values ** 2).sum(axis=1))
                    base_cols.append(f"{p}_{name}_norm")
        else:
            for p in ["hand","chest","ankle"]:
                for _, axes in keep_triplets:
                    for a in axes:
                        base_cols.append(f"{p}_{a}")

        # window → features
        def compute_feats(W: pd.DataFrame):
            out = {}
            if "std" in stats:
                out.update(W.std(ddof=1).add_suffix("__std").to_dict())
            if "rms" in stats:
                out.update((np.sqrt((W**2).mean())).add_suffix("__rms").to_dict())
            if "mean" in stats:
                out.update(W.mean().add_suffix("__mean").to_dict())
            return out

        rows, labels, subjects = [], [], []
        df = df.sort_values(["subject_id","timestamp"]).reset_index(drop=True)
        for sid, g in df.groupby("subject_id", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            for start in range(0, max(0, n - WIN + 1), STEP):
                w = g.iloc[start:start + WIN]
                if len(w) < WIN: 
                    continue
                y_win = int(np.bincount(w["y"].astype(int)).argmax())
                feats = compute_feats(w[base_cols])
                feats["subject_id"] = sid
                rows.append(feats); labels.append(y_win); subjects.append(sid)

        feat_df = pd.DataFrame(rows).fillna(0.0)
        y_all   = np.asarray(labels, dtype=int)
        subj_all= np.asarray(subjects)

        feature_cols = [c for c in feat_df.columns if c != "subject_id"]
        return feat_df, y_all, subj_all, feature_cols

    def get_gan_view(self) -> dict:
        gv = getattr(self, "_gan_view_cache", None)
        if gv is None:
            return {"supported": False}
        return gv

    def ctgan_training_view(self, **_):
        # Returns the cached GAN view built during split_*; all args ignored.
        return self.get_gan_view()



