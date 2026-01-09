import re
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
        },
        "pamap2": {
            "data_path": "datasets/PAMAP2_Dataset",
        },
    }
    def __init__(self, dataset_name, multiclass, minority_id, majority_id, third_id, pca_components=2, seed=42, device="cpu"):
        self.dataset_name = dataset_name
        self.MINORITY_ID = minority_id
        self.MAJORITY_ID = majority_id
        self.third_id=third_id
        self.pca_components = pca_components
        self.seed = seed
        self.device = device
        self.data_path = self.DATASET_REGISTRY[dataset_name]["data_path"]
        self.multiclass=multiclass

    from dataclasses import dataclass

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

    def _make_pca(self, n_components: int) -> PCA:
        return PCA(
            n_components=n_components,
            svd_solver="full",          # deterministic basis for same data
            random_state=self.seed,     # deterministic for randomized solvers 
        )

    def _fit_theta_transforms(
        self,
        X_train_biased_df: pd.DataFrame,
        *,
        cat_cols: list[str],
        num_cols: list[str],
        pca_components: int,
    ):
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

        scaler = StandardScaler()

        X_train_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_biased_df), 0))
        X_train_num = scaler.fit_transform(X_train_biased_df[num_cols])  if len(num_cols) else np.empty((len(X_train_biased_df), 0))
        X_train_all = np.hstack([X_train_num, X_train_cat])

        pca = self._make_pca(pca_components)
        pca.fit(X_train_all)

        return encoder, scaler, pca

    def split_census_income(self, train_size=None, bias_pct=0.2,
                    val_frac=0.20, test_frac=0.20, pca_components=2):
        """
        PIPELINE:
        1) Load raw data.
        2) Split raw data into θ_train / θ_val / θ_test (stratified by the original labels).
        3) Within each split, apply the *same bias* by downsampling the minority class:
            - Keep all majority samples.
            - bias pct is target pct
            (We avoid any pre-split upsampling to prevent duplicates across splits.)
        4) Fit OneHotEncoder + StandardScaler on θ_train *only*; transform θ_val and θ_test with those fitted transformers (no leakage).
        5) Fit PCA on θ_train *only*; transform θ_val and θ_test with that PCA (no leakage).
        6) Optionally subsample θ_train to a fixed size *after* biasing and before fitting PCA (still no leakage).
        7) Return torch tensors for train/val/test on self.device.
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1, \
            "val_frac and test_frac must be in (0,1) and sum to < 1."

        # 1) Load raw data
        data_path = self.data_path
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        X_df_raw = pd.read_csv(data_path, header=None, names=column_names, na_values="?", skipinitialspace=True)
        y_raw = np.where(X_df_raw["income"].isin(['>50K', '>50K.']), 1, 0).astype(int)
        X_df_raw = X_df_raw.drop(columns=["income"])

        # Identify column types
        cat_cols = [c for c in X_df_raw.columns if X_df_raw[c].dtype.name in ['category', 'object', 'bool']]
        num_cols = [c for c in X_df_raw.columns if np.issubdtype(X_df_raw[c].dtype, np.number)]

        # 2) Split raw into θ_train / θ_temp, then θ_val / θ_test
        X_train_df, X_temp_df, y_train, y_temp = train_test_split(
            X_df_raw, y_raw, test_size=(val_frac + test_frac),
            random_state=self.seed, stratify=y_raw
        )
        rel_test = test_frac / (val_frac + test_frac)
        X_val_df, X_test_df, y_val, y_test = train_test_split(
            X_temp_df, y_temp, test_size=rel_test,
            random_state=self.seed, stratify=y_temp
        )

        # Helper: apply the SAME bias inside a split (keep all majority; keep fraction of minority)
        def apply_bias(df_split, y_split, target_minority_pct):
            df = df_split.copy()
            df["__y__"] = y_split
            df_major = df[df["__y__"] == 0]
            df_minor = df[df["__y__"] == 1]

            n_major = len(df_major)
            n_minor = len(df_minor)

            if n_minor == 0 or n_major == 0:
                # Edge case: one class missing
                df_biased = df
            else:
                # compute how many minority to keep to hit target proportion
                keep_minority = int(np.floor((target_minority_pct * n_major) / (1 - target_minority_pct)))

                # cap to available samples
                keep_minority = min(n_minor, max(1, keep_minority))

                df_minor_biased = df_minor.sample(n=keep_minority, random_state=self.seed, replace=False)
                df_biased = pd.concat([df_major, df_minor_biased], axis=0) \
                            .sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

            y_out = df_biased["__y__"].to_numpy(dtype=int)
            X_out = df_biased.drop(columns=["__y__"])
            return X_out, y_out

        # 3) Apply same bias in each split (no cross-split duplication, distributions aligned)
        target_minority_pct = bias_pct  # e.g., bias_pct=0.25 -> 25% of dataset is minority data
        X_train_biased_df, y_train_biased = apply_bias(X_train_df, y_train, target_minority_pct)
        X_val_biased_df,   y_val_biased   = apply_bias(X_val_df,   y_val,   target_minority_pct)
        X_test_biased_df,  y_test_biased  = apply_bias(X_test_df,  y_test,  target_minority_pct)

        # Optional: subsample θ_train to fixed size *after* biasing (stratified)
        if train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _ = train_test_split(
                X_train_biased_df, y_train_biased,
                train_size=train_size, random_state=self.seed, stratify=y_train_biased
            )

        # 4) Fit encoder + scaler on θ_train only; transform val/test (no leakage)
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        scaler  = StandardScaler()

        # Fit on train
        X_train_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_biased_df), 0))
        X_train_num = scaler.fit_transform(X_train_biased_df[num_cols])   if len(num_cols) else np.empty((len(X_train_biased_df), 0))
        X_train_all = np.hstack([X_train_num, X_train_cat])

        # Transform val/test with the *fitted* encoder/scaler
        X_val_cat  = encoder.transform(X_val_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_val_biased_df), 0))
        X_val_num  = scaler.transform(X_val_biased_df[num_cols])  if len(num_cols) else np.empty((len(X_val_biased_df), 0))
        X_val_all  = np.hstack([X_val_num, X_val_cat])

        X_test_cat = encoder.transform(X_test_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_test_biased_df), 0))
        X_test_num = scaler.transform(X_test_biased_df[num_cols])  if len(num_cols) else np.empty((len(X_test_biased_df), 0))
        X_test_all = np.hstack([X_test_num, X_test_cat])

        # 5) Fit PCA on θ_train only; transform val/test
        pca = self._make_pca(pca_components)
        X_train_pca = pca.fit_transform(X_train_all)
        X_val_pca   = pca.transform(X_val_all)
        X_test_pca  = pca.transform(X_test_all)


        # 6) Convert to torch tensors on device
        X_train_theta = torch.tensor(X_train_pca, dtype=torch.float32, device=self.device)
        X_val_theta   = torch.tensor(X_val_pca,   dtype=torch.float32, device=self.device)
        X_test_theta  = torch.tensor(X_test_pca,  dtype=torch.float32, device=self.device)

        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=self.device)
        y_val_theta   = torch.tensor(y_val_biased,   dtype=torch.long, device=self.device)
        y_test_theta  = torch.tensor(y_test_biased,  dtype=torch.long, device=self.device)

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
                "pca": pca,
                "cat_cols": list(cat_cols),
                "num_cols": list(num_cols),
            }
        except Exception:
            pass

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta

    def split_pamap2(
        self,
        train_size=None,
        bias_pct=0.20,
        val_frac=0.20,
        test_frac=0.20,
        pca_components=6,
        win_seconds=5.0,
        step_seconds=2.5,
        drop_magnetometers=True,
        use_vector_norms=True,
        stats=("std", "rms"),
    ):
        """
        PAMAP2
        """
        assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1, \
            "val_frac and test_frac must be in (0,1) and sum to < 1."

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

        #subject-aware split (train / val / test)
        unique_subjects = np.unique(subj_all)
        rng.shuffle(unique_subjects)

        n_subj = len(unique_subjects)
        n_test = max(1, int(round(test_frac * n_subj)))
        n_val  = max(1, int(round(val_frac  * n_subj)))
        if n_test + n_val >= n_subj:
            n_test = max(1, n_test)
            n_val  = max(1, min(n_val, n_subj - n_test - 1))

        test_subj = unique_subjects[:n_test]
        val_subj  = unique_subjects[n_test:n_test + n_val]
        train_subj = unique_subjects[n_test + n_val:]

        def mask_for(subj_list):
            return np.isin(subj_all, subj_list)

        m_train = mask_for(train_subj)
        m_val   = mask_for(val_subj)
        m_test  = mask_for(test_subj)

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

        target_minority_pct = float(bias_pct)

        # Deterministic seeds 
        seed_local = int(self.seed)

        # Bias train/val/test
        X_train_biased_df, y_train_biased = apply_bias_rope(X_train_df, y_train, target_minority_pct, seed_local)
        X_val_biased_df,   y_val_biased   = apply_bias_rope(X_val_df,   y_val,   target_minority_pct, seed_local)
        X_test_biased_df,  y_test_biased  = apply_bias_rope(X_test_df,  y_test,  target_minority_pct, seed_local)


        #subsample TRAIN after biasing (deterministic)
        if train_size is not None and train_size < len(X_train_biased_df):
            X_train_biased_df, _, y_train_biased, _ = train_test_split(
                X_train_biased_df,
                y_train_biased,
                train_size=train_size,
                random_state=seed_local,
                stratify=y_train_biased,
            )

        #scaler + PCA on TRAIN only
        meta_cols = ["subject_id"]
        feature_cols = [c for c in X_train_biased_df.columns if c not in meta_cols]

        scaler = StandardScaler()
        Xtr_z = scaler.fit_transform(X_train_biased_df[feature_cols].values)

        # Deterministic PCA config
        pca = PCA(n_components=pca_components, svd_solver="full", random_state=seed_local)
        Xtr_p = pca.fit_transform(Xtr_z)

        def transform(df_split):
            Xz = scaler.transform(df_split[feature_cols].values)
            return pca.transform(Xz)

        Xval_p = transform(X_val_biased_df)
        Xtest_p = transform(X_test_biased_df)

        # -------- tensors on device --------
        device = self.device
        X_train_theta = torch.tensor(Xtr_p, dtype=torch.float32, device=device)
        X_val_theta   = torch.tensor(Xval_p, dtype=torch.float32, device=device)
        X_test_theta  = torch.tensor(Xtest_p, dtype=torch.float32, device=device)

        y_train_theta = torch.tensor(y_train_biased, dtype=torch.long, device=device)
        y_val_theta   = torch.tensor(y_val_biased, dtype=torch.long, device=device)
        y_test_theta  = torch.tensor(y_test_biased, dtype=torch.long, device=device)

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
        print(f"PCA comps: {Xtr_p.shape[1]} | Features in: {len(feature_cols)} | Classes: {sorted(np.unique(y_train_biased))}")

        return X_train_theta, X_val_theta, X_test_theta, y_train_theta, y_val_theta, y_test_theta



    def get_data_splits(self, **kwargs):
        if self.dataset_name == "census_income":
            return self.split_census_income(**kwargs)
        elif self.dataset_name == "pamap2":
            return self.split_pamap2(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    #Check if needed still
    def rebuild_original_train_pool_theta(
        self, *, bias_pct: float, val_frac: float, test_frac: float,
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

            # TRAIN-biased for fitting transforms
            X_train_biased_df, y_train_biased = apply_bias(X_train_df, y_train, bias_pct)
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
            pca = self._make_pca(pca_components)

            Xtr_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_biased_df),0))
            Xtr_num = scaler.fit_transform(X_train_biased_df[num_cols])   if len(num_cols) else np.empty((len(X_train_biased_df),0))
            Xtr_all = np.hstack([Xtr_num, Xtr_cat])
            _ = pca.fit_transform(Xtr_all)  # fit only

            Xpool_cat = encoder.transform(X_train_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_df),0))
            Xpool_num = scaler.transform(X_train_df[num_cols])  if len(num_cols) else np.empty((len(X_train_df),0))
            Xpool_all = np.hstack([Xpool_num, Xpool_cat])
            Xpool_pca = pca.transform(Xpool_all)

            X_pool_theta = torch.tensor(Xpool_pca, dtype=torch.float32, device=device)
            y_pool_theta = torch.tensor(y_train,   dtype=torch.long,   device=device)
            return X_pool_theta, y_pool_theta

        elif self.dataset_name == "pamap2":
            # ---- PAMAP2 ----
            rng = np.random.RandomState(self.seed)

            # constants copied from split_pamap2 defaults
            FS = 100
            win_seconds = 5.0
            step_seconds = 2.5
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

            X_train_biased_df, y_train_biased = apply_bias_rope(
                X_train_unbiased_df, y_train_unbiased, float(bias_pct)
            )
            if train_size is not None and train_size < len(X_train_biased_df):
                X_train_biased_df, _, y_train_biased, _ = train_test_split(
                    X_train_biased_df, y_train_biased,
                    train_size=train_size, random_state=self.seed, stratify=y_train_biased
                )

            # Fit scaler+PCA on TRAIN-biased; transform UNBIASED TRAIN pool
            feature_cols = [c for c in X_train_biased_df.columns if c != "subject_id"]
            scaler = StandardScaler()
            Xtr_z  = scaler.fit_transform(X_train_biased_df[feature_cols].values)
            pca    = PCA(n_components=pca_components, svd_solver="full", random_state=self.seed)
            _      = pca.fit_transform(Xtr_z)  # fit only

            Xpool_z  = scaler.transform(X_train_unbiased_df[feature_cols].values)
            Xpool_p  = pca.transform(Xpool_z)

            X_pool_theta = torch.tensor(Xpool_p, dtype=torch.float32, device=device)
            y_pool_theta = torch.tensor(y_train_unbiased, dtype=torch.long, device=device)
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

    def ctgan_training_view(self, *, bias_pct, val_frac, test_frac, train_size, pca_components, device):
        # apples-to-apples: reuse exactly what split_* used
        return self.get_gan_view()



