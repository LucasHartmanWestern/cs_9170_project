#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, json, os, random
import numpy as np
import pandas as pd
from pathlib import Path

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

# Torch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# SDV / CTGAN (pip install sdv[all])
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def brier_mean(y_true_prob1, y_true):
    y_true = y_true.astype(np.float32)
    return float(np.mean((y_true_prob1 - y_true) ** 2))

def softmax2(logits):
    m = logits.max(axis=1, keepdims=True).values
    e = torch.exp(logits - m)
    return e / e.sum(dim=1, keepdims=True)

def class_report(y_true, prob1, thresh=0.5):
    y_pred = (prob1 >= thresh).astype(int)
    f1_min = f1_score(y_true, y_pred, pos_label=1)
    f1_maj = f1_score(1 - y_true, 1 - y_pred, pos_label=1)
    f1_macro = 0.5 * (f1_min + f1_maj)
    n1 = (y_true == 1).sum()
    n0 = (y_true == 0).sum()
    f1_weighted = (f1_min * (n1/(n0+n1))) + (f1_maj * (n0/(n0+n1)))
    brier = brier_mean(prob1, y_true)
    return dict(
        f1_minority=float(f1_min),
        f1_majority=float(f1_maj),
        f1_weighted=float(f1_weighted),
        f1_macro=float(f1_macro),
        brier=float(brier),
    )

def load_adult(path="census+income/adult.data"):
    column_names = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"
    ]
    df = pd.read_csv(path, header=None, names=column_names,
                     na_values="?", skipinitialspace=True)
    y = np.where(df["income"].isin([">50K", ">50K."]), 1, 0).astype(int)
    X = df.drop(columns=["income"])
    return X, y

def split_bias(
    X_df_raw, y_raw, seed, val_frac=0.20, test_frac=0.20, bias_pct=None
):
    assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1

    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df_raw, y_raw, test_size=(val_frac + test_frac),
        random_state=seed, stratify=y_raw
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=rel_test,
        random_state=seed, stratify=y_temp
    )

    def apply_bias(df_split, y_split, target_minority_pct):
        df = df_split.copy()
        df["__y__"] = y_split
        maj = df[df["__y__"] == 0]
        mino = df[df["__y__"] == 1]
        n_major = len(maj)
        n_minor = len(mino)
        if n_major == 0 or n_minor == 0:
            biased = df
        else:
            keep_min = int(np.floor((target_minority_pct * n_major) / (1 - target_minority_pct)))
            keep_min = min(n_minor, max(1, keep_min))
            mino_b = mino.sample(n=int(keep_min), random_state=seed, replace=False)
            biased = pd.concat([maj, mino_b], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        y_out = biased["__y__"].to_numpy(dtype=int)
        X_out = biased.drop(columns=["__y__"])
        return X_out, y_out

    if bias_pct is not None:
        Xtr_b, ytr_b = apply_bias(X_train_df, y_train, bias_pct)
        Xv_b,  yv_b  = apply_bias(X_val_df,   y_val,   bias_pct)
        Xte_b, yte_b = apply_bias(X_test_df,  y_test,  bias_pct)
    else:
        Xtr_b, ytr_b = X_train_df.copy(), y_train.copy()
        Xv_b,  yv_b  = X_val_df.copy(),   y_val.copy()
        Xte_b, yte_b = X_test_df.copy(),  y_test.copy()

    return (X_train_df, y_train), (Xtr_b, ytr_b), (Xv_b, yv_b), (Xte_b, yte_b)

def identify_cols(X_df):
    cat_cols = [c for c in X_df.columns if X_df[c].dtype.name in ['category', 'object', 'bool']]
    num_cols = [c for c in X_df.columns if np.issubdtype(X_df[c].dtype, np.number)]
    return cat_cols, num_cols


# -----------------------------
# Simple FFNN classifier (α)
# -----------------------------

class SimpleFFNN(nn.Module):
    def __init__(self, d_in, hidden=(32,16), n_out=2):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_ffnn(X, y, d_in, device, batch=64, lr=1e-3, epochs=10, hidden=(32,16)):
    model = SimpleFFNN(d_in, hidden=hidden, n_out=2).to(device)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32, device=device),
                       torch.tensor(y, dtype=torch.long, device=device))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    return model


# -----------------------------
# Transform rebuild & sampling (TestSuite parity)
# -----------------------------

def rebuild_transforms_and_pool(
    *,
    X_df_raw, y_raw,
    seed: int,
    bias_pct: float,
    val_frac: float,
    test_frac: float,
    train_size: int | None,
    pca_components: int,
):
    """
    Fit encoder/scaler/PCA on TRAIN-biased (optionally capped), then
    transform the UNBIASED original TRAIN to create a real-data pool
    in the same θ-space.
    """
    cat_cols, num_cols = identify_cols(X_df_raw)

    # Split raw
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df_raw, y_raw, test_size=(val_frac + test_frac),
        random_state=seed, stratify=y_raw
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=rel_test,
        random_state=seed, stratify=y_temp
    )

    def apply_bias(df_split, y_split, target_minority_pct):
        df = df_split.copy()
        df["__y__"] = y_split
        maj = df[df["__y__"] == 0]
        mino = df[df["__y__"] == 1]
        n_major = len(maj); n_minor = len(mino)
        if n_major == 0 or n_minor == 0:
            biased = df
        else:
            keep_min = int(np.floor((target_minority_pct * n_major) / (1 - target_minority_pct)))
            keep_min = min(n_minor, max(1, keep_min))
            mino_b = mino.sample(n=int(keep_min), random_state=seed, replace=False)
            biased = pd.concat([maj, mino_b], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        y_out = biased["__y__"].to_numpy(dtype=int)
        X_out = biased.drop(columns=["__y__"])
        return X_out, y_out

    if bias_pct is not None:
        X_train_biased_df, y_train_biased = apply_bias(X_train_df, y_train, bias_pct)
    else:
        X_train_biased_df, y_train_biased = X_train_df.copy(), y_train.copy()

    # Optional cap on biased TRAIN before fitting transforms
    if train_size is not None and train_size < len(X_train_biased_df):
        X_train_biased_df, _, y_train_biased, _ = train_test_split(
            X_train_biased_df, y_train_biased,
            train_size=train_size, random_state=seed, stratify=y_train_biased
        )

    # Fit encoder/scaler on TRAIN-biased; PCA on TRAIN-biased
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    scaler = StandardScaler()

    X_train_cat = encoder.fit_transform(X_train_biased_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_biased_df), 0))
    X_train_num = scaler.fit_transform(X_train_biased_df[num_cols])   if len(num_cols) else np.empty((len(X_train_biased_df), 0))
    X_train_all = np.hstack([X_train_num, X_train_cat])

    pca = PCA(n_components=pca_components)
    _ = pca.fit_transform(X_train_all)  # fit only

    # Build unbiased TRAIN pool in the same θ-space
    X_pool_cat = encoder.transform(X_train_df[cat_cols]) if len(cat_cols) else np.empty((len(X_train_df), 0))
    X_pool_num = scaler.transform(X_train_df[num_cols])  if len(num_cols) else np.empty((len(X_train_df), 0))
    X_pool_all = np.hstack([X_pool_num, X_pool_cat])
    X_pool_theta = pca.transform(X_pool_all)

    return (encoder, scaler, pca, cat_cols, num_cols,
            torch.tensor(X_pool_theta, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long))

def sample_real_minority(x_pool: torch.Tensor, y_pool: torch.Tensor, add_n: int, seed: int | None):
    add_n = int(max(0, add_n))
    if add_n == 0:
        return None, None
    with torch.no_grad():
        y1_mask = (y_pool == 1)
        if y1_mask.sum().item() == 0:
            return None, None
        idx_all = torch.nonzero(y1_mask, as_tuple=False).squeeze(1)
        gen = torch.Generator(device=x_pool.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        # With replacement (TestSuite behavior)
        rand_idx = torch.randint(0, idx_all.shape[0], (add_n,), device=idx_all.device, generator=gen)
        sel = idx_all[rand_idx]
        Xr = x_pool[sel]
        yr = torch.ones(add_n, dtype=y_pool.dtype, device=y_pool.device)
        return Xr, yr


# -----------------------------
# CTGAN (class-conditional)
# -----------------------------

def train_ctgan_conditional(
    X_unbiased: pd.DataFrame,
    y_unbiased: np.ndarray,
    *,
    label_col: str = "__y__",
    ctgan_epochs: int = 300,
    seed: int = 42,
    cap_ctgan_train: int | None = None,
):
    """
    Train CTGAN on the FULL original TRAIN (both classes) with the label column included,
    so we can condition on class during sampling.
    """
    df = X_unbiased.copy()
    df[label_col] = y_unbiased.astype(int)

    # Optional cap (stratified)
    if cap_ctgan_train is not None and cap_ctgan_train < len(df):
        frac = cap_ctgan_train / len(df)
        df = (df.groupby(label_col, group_keys=False)
                .apply(lambda g: g.sample(n=max(1, int(round(len(g)*frac))), random_state=seed))
                .reset_index(drop=True))

    df[label_col] = df[label_col].astype(str)# SDV CTGAN expects object, not category

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # extra seeding
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    synth = CTGANSynthesizer(
        metadata,
        epochs=int(ctgan_epochs),
        verbose=True,
        cuda=torch.cuda.is_available(),
    )
    synth.fit(df)
    return synth

def sample_conditional_minority(
    synth,
    n_samples: int,
    *,
    label_col: str = "__y__",
    label_value: int = 1,
) -> pd.DataFrame:
    """
    Robust conditional sampling across SDV versions:
    - Newer SDV: use sample_from_conditions with sdv.sampling.Condition
    - Older SDV: fall back to sample_conditions(DataFrame with 'num_rows')
    - Last resort: try list-of-dicts format
    """
    if n_samples <= 0:
        return pd.DataFrame()

    # Always match the dtype used in training (we casted label to string)
    label_value_str = str(label_value)

    # 1) Preferred (newer SDV): Condition objects
    try:
        try:
            from sdv.sampling import Condition  # SDV >= 1.0
        except Exception:
            Condition = None

        if Condition is not None and hasattr(synth, "sample_from_conditions"):
            cond = Condition(
                num_rows=int(n_samples),
                column_values={label_col: label_value_str},
            )
            out = synth.sample_from_conditions([cond])
            return out.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    except Exception:
        pass  # fall through to older API

    # 2) Older SDV: sample_conditions expects a DataFrame with conditioned columns + 'num_rows'
    try:
        if hasattr(synth, "sample_conditions"):
            cond_df = pd.DataFrame({
                label_col: [label_value_str],
                "num_rows": [int(n_samples)]
            })
            out = synth.sample_conditions(cond_df)
            return out.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    except Exception:
        pass  # try last resort

    # 3) Some versions accept list-of-dicts to sample_from_conditions
    try:
        if hasattr(synth, "sample_from_conditions"):
            cond_spec = [{"num_rows": int(n_samples), "column_values": {label_col: label_value_str}}]
            out = synth.sample_from_conditions(cond_spec)
            return out.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    except Exception as e:
        raise RuntimeError(
            f"Conditional sampling failed across all strategies. "
            f"SDV version={getattr(synth, '__class__', type('x', (), {})).__name__}. "
            f"Original error: {e}"
        )



# -----------------------------
# Main experiment
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="census+income/adult.data")
    ap.add_argument("--bias_pct", type=float, default=None, help="Target minority share inside each split after biasing (None = no bias)")
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--train_size", type=int, default=None, help="Optional cap on BIASED θ_train size (stratified)")
    ap.add_argument("--pca_components", type=int, default=2)

    # CTGAN controls
    ap.add_argument("--ctgan_epochs", type=int, default=300)
    ap.add_argument("--n_synth", type=int, default=2000, help="How many synthetic minority to add")
    ap.add_argument("--cap_ctgan_train", type=int, default=None, help="Optional cap on UNBIASED CTGAN training rows")

    # FFNN controls
    ap.add_argument("--ffnn_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ffnn_hidden", type=str, default="32,16")

    # Alpha+REAL controls (parity with TestSuite α_same+REAL)
    ap.add_argument("--real_add_n", type=int, default=2000, help="How many real minority to add in α_same+REAL")

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_dir", type=str, default="ctgan_runs")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    hidden = tuple(int(h) for h in args.ffnn_hidden.split(","))

    t0 = time.time()

    # 1) Load raw data
    X_raw, y_raw = load_adult(args.data_path)
    cat_cols_all, num_cols_all = identify_cols(X_raw)

    # 2) Split raw & create per-split BIASED versions (for alpha baseline & transforms)
    (Xtr_unb, ytr_unb), (Xtr_b, ytr_b), (Xv_b, yv_b), (Xte_b, yte_b) = split_bias(
        X_raw, y_raw, seed=args.seed,
        val_frac=args.val_frac, test_frac=args.test_frac, bias_pct=args.bias_pct
    )

    # 3) Optional cap on BIASED θ_train (classifier train + transforms)
    if args.train_size is not None and args.train_size < len(Xtr_b):
        Xtr_b, _, ytr_b, _ = train_test_split(
            Xtr_b, ytr_b, train_size=args.train_size,
            random_state=args.seed, stratify=ytr_b
        )

    # 4) Fit enc+scaler on θ_train (biased); transform VAL/TEST; fit PCA on θ_train; transform others
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    scaler = StandardScaler()

    Xtr_cat = encoder.fit_transform(Xtr_b[cat_cols_all]) if len(cat_cols_all) else np.empty((len(Xtr_b), 0))
    Xtr_num = scaler.fit_transform(Xtr_b[num_cols_all])   if len(num_cols_all) else np.empty((len(Xtr_b), 0))
    Xtr_all = np.hstack([Xtr_num, Xtr_cat])

    Xv_cat = encoder.transform(Xv_b[cat_cols_all]) if len(cat_cols_all) else np.empty((len(Xv_b), 0))
    Xv_num = scaler.transform(Xv_b[num_cols_all])  if len(num_cols_all) else np.empty((len(Xv_b), 0))
    Xv_all = np.hstack([Xv_num, Xv_cat])

    Xte_cat = encoder.transform(Xte_b[cat_cols_all]) if len(cat_cols_all) else np.empty((len(Xte_b), 0))
    Xte_num = scaler.transform(Xte_b[num_cols_all])  if len(num_cols_all) else np.empty((len(Xte_b), 0))
    Xte_all = np.hstack([Xte_num, Xte_cat])

    pca = PCA(n_components=args.pca_components)
    Xtr_theta = pca.fit_transform(Xtr_all)
    Xv_theta  = pca.transform(Xv_all)
    Xte_theta = pca.transform(Xte_all)

    ytr = ytr_b.astype(int)
    yv  = yv_b.astype(int)
    yte = yte_b.astype(int)

    # 5) Alpha (real-only) on biased TRAIN
    alpha = train_ffnn(
        Xtr_theta, ytr, d_in=Xtr_theta.shape[1],
        device=device, batch=args.batch_size, lr=args.lr, epochs=args.ffnn_epochs, hidden=hidden
    )
    alpha.eval()
    with torch.no_grad():
        logits = alpha(torch.tensor(Xte_theta, dtype=torch.float32, device=device)).cpu()
        p1 = softmax2(logits)[:, 1].numpy()
    alpha_metrics = class_report(yte, p1, 0.5)

    # 6) α_raw_original: train on original TRAIN (unbiased) mapped into the same θ-space
    (_enc_pool, _sc_pool, _pca_pool, _cat_cols_pool, _num_cols_pool,
     X_pool_theta, y_pool_theta) = rebuild_transforms_and_pool(
        X_df_raw=X_raw, y_raw=y_raw, seed=args.seed, bias_pct=args.bias_pct,
        val_frac=args.val_frac, test_frac=args.test_frac, train_size=args.train_size,
        pca_components=args.pca_components
    )
    alpha_raw = train_ffnn(
        X_pool_theta.cpu().numpy(), y_pool_theta.cpu().numpy(),
        d_in=X_pool_theta.shape[1], device=device,
        batch=args.batch_size, lr=args.lr, epochs=args.ffnn_epochs, hidden=hidden
    )
    alpha_raw.eval()
    with torch.no_grad():
        logits_ar = alpha_raw(torch.tensor(Xte_theta, dtype=torch.float32, device=device)).cpu()
        p1_ar = softmax2(logits_ar)[:, 1].numpy()
    alpha_raw_metrics = class_report(yte, p1_ar, 0.5)

    # 7) α_same+REAL: biased TRAIN + real minority from original TRAIN pool (with replacement)
    Xr_t, yr_t = sample_real_minority(X_pool_theta.to(device), y_pool_theta.to(device), int(args.real_add_n), args.seed)
    if Xr_t is not None:
        X_aug_real = np.vstack([Xtr_theta, Xr_t.cpu().numpy()])
        y_aug_real = np.concatenate([ytr, yr_t.cpu().numpy()])
        added_real = int(args.real_add_n)
    else:
        X_aug_real, y_aug_real = Xtr_theta, ytr
        added_real = 0
    pos_frac_real = float((y_aug_real == 1).mean())

    alpha_plus_real = train_ffnn(
        X_aug_real, y_aug_real, d_in=X_aug_real.shape[1],
        device=device, batch=args.batch_size, lr=args.lr, epochs=args.ffnn_epochs, hidden=hidden
    )
    alpha_plus_real.eval()
    with torch.no_grad():
        logits_pr = alpha_plus_real(torch.tensor(Xte_theta, dtype=torch.float32, device=device)).cpu()
        p1_pr = softmax2(logits_pr)[:, 1].numpy()
    alpha_plus_real_metrics = class_report(yte, p1_pr, 0.5)

    # 8) CTGAN (conditional on label) on UNBIASED TRAIN; generate minority; map to θ-space; α+CTGAN
    ctgan = train_ctgan_conditional(
        X_unbiased=Xtr_unb, y_unbiased=ytr_unb,
        label_col="__y__", ctgan_epochs=args.ctgan_epochs,
        seed=args.seed, cap_ctgan_train=args.cap_ctgan_train
    )

    syn_df = sample_conditional_minority(
        ctgan, n_samples=args.n_synth, label_col="__y__", label_value=1
    ) if args.n_synth > 0 else pd.DataFrame()

    if len(syn_df) > 0:
        syn_df_no_y = syn_df.drop(columns=["__y__"], errors="ignore")
        syn_cat = encoder.transform(syn_df_no_y[cat_cols_all]) if len(cat_cols_all) else np.empty((len(syn_df_no_y), 0))
        syn_num = scaler.transform(syn_df_no_y[num_cols_all])  if len(num_cols_all) else np.empty((len(syn_df_no_y), 0))
        syn_all = np.hstack([syn_num, syn_cat])
        syn_theta = pca.transform(syn_all)
        ys = np.ones(len(syn_theta), dtype=int)  # conditioned minority
        Xtr_aug = np.vstack([Xtr_theta, syn_theta])
        ytr_aug = np.concatenate([ytr, ys])
    else:
        Xtr_aug, ytr_aug = Xtr_theta, ytr

    pos_frac_syn = float((ytr_aug == 1).mean())

    alpha_ct = train_ffnn(
        Xtr_aug, ytr_aug, d_in=Xtr_aug.shape[1],
        device=device, batch=args.batch_size, lr=args.lr, epochs=args.ffnn_epochs, hidden=hidden
    )
    alpha_ct.eval()
    with torch.no_grad():
        logits_ct = alpha_ct(torch.tensor(Xte_theta, dtype=torch.float32, device=device)).cpu()
        p1_ct = softmax2(logits_ct)[:, 1].numpy()
    alpha_ct_metrics = class_report(yte, p1_ct, 0.5)

    # 9) Report + save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== RESULTS (θ_test) ===")
    def pr(name, m):
        print(f"{name:>18s} | F1(min)={m['f1_minority']:.4f} | F1(maj)={m['f1_majority']:.4f} "
              f"| F1(w)={m['f1_weighted']:.4f} | F1(macro)={m['f1_macro']:.4f} | Brier={m['brier']:.4f}")
    pr("Alpha (biased)",         alpha_metrics)
    pr("α_raw_original",         alpha_raw_metrics)
    print(f"[α_same+REAL] added={added_real} | final_N={len(y_aug_real)} | pos_frac={pos_frac_real:.3f}")
    pr("α_same+REAL",            alpha_plus_real_metrics)
    print(f"[CTGAN]       added={len(syn_df)} | final_N={len(ytr_aug)} | pos_frac={pos_frac_syn:.3f}")
    pr("Alpha + CTGAN",          alpha_ct_metrics)

    result = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "seed": args.seed,
        "bias_pct": args.bias_pct,
        "train_size_biased": None if args.train_size is None else int(args.train_size),
        "pca_components": args.pca_components,
        "ffnn_hidden": list(hidden),
        "ffnn_epochs": args.ffnn_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,

        "ctgan_epochs": args.ctgan_epochs,
        "n_synth": args.n_synth,
        "cap_ctgan_train": None if args.cap_ctgan_train is None else int(args.cap_ctgan_train),

        "real_add_n": int(args.real_add_n),

        "alpha_biased": alpha_metrics,
        "alpha_raw_original": alpha_raw_metrics,
        "alpha_plus_real": {**alpha_plus_real_metrics,
                            "added_real": int(added_real),
                            "final_N": int(len(y_aug_real)),
                            "pos_frac": float(pos_frac_real)},
        "alpha_plus_ctgan": {**alpha_ct_metrics,
                             "added_syn": int(len(syn_df)),
                             "final_N": int(len(ytr_aug)),
                             "pos_frac": float(pos_frac_syn)},
    }

    with open(out_dir / "ctgan_full_comparable_results.json", "w") as f:
        json.dump(result, f, indent=2)

    row = {
        "timestamp": result["timestamp"],
        "seed": args.seed,
        "bias_pct": args.bias_pct,
        "train_size_biased": args.train_size if args.train_size is not None else "",
        "pca_components": args.pca_components,
        "ffnn_hidden": "/".join(map(str, hidden)),
        "ffnn_epochs": args.ffnn_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,

        "ctgan_epochs": args.ctgan_epochs,
        "n_synth": args.n_synth,
        "cap_ctgan_train": args.cap_ctgan_train if args.cap_ctgan_train is not None else "",

        # Alpha (biased)
        "alpha_f1_minority": result["alpha_biased"]["f1_minority"],
        "alpha_f1_majority": result["alpha_biased"]["f1_majority"],
        "alpha_f1_weighted": result["alpha_biased"]["f1_weighted"],
        "alpha_f1_macro": result["alpha_biased"]["f1_macro"],
        "alpha_brier": result["alpha_biased"]["brier"],

        # α_raw_original
        "alpha_raworig_f1_minority": result["alpha_raw_original"]["f1_minority"],
        "alpha_raworig_f1_majority": result["alpha_raw_original"]["f1_majority"],
        "alpha_raworig_f1_weighted": result["alpha_raw_original"]["f1_weighted"],
        "alpha_raworig_f1_macro": result["alpha_raw_original"]["f1_macro"],
        "alpha_raworig_brier": result["alpha_raw_original"]["brier"],

        # α_same+REAL
        "alpha_plusreal_added": result["alpha_plus_real"]["added_real"],
        "alpha_plusreal_final_N": result["alpha_plus_real"]["final_N"],
        "alpha_plusreal_pos_frac": result["alpha_plus_real"]["pos_frac"],
        "alpha_plusreal_f1_minority": result["alpha_plus_real"]["f1_minority"],
        "alpha_plusreal_f1_majority": result["alpha_plus_real"]["f1_majority"],
        "alpha_plusreal_f1_weighted": result["alpha_plus_real"]["f1_weighted"],
        "alpha_plusreal_f1_macro": result["alpha_plus_real"]["f1_macro"],
        "alpha_plusreal_brier": result["alpha_plus_real"]["brier"],

        # Alpha + CTGAN
        "alphaCT_added": result["alpha_plus_ctgan"]["added_syn"],
        "alphaCT_final_N": result["alpha_plus_ctgan"]["final_N"],
        "alphaCT_pos_frac": result["alpha_plus_ctgan"]["pos_frac"],
        "alphaCT_f1_minority": result["alpha_plus_ctgan"]["f1_minority"],
        "alphaCT_f1_majority": result["alpha_plus_ctgan"]["f1_majority"],
        "alphaCT_f1_weighted": result["alpha_plus_ctgan"]["f1_weighted"],
        "alphaCT_f1_macro": result["alpha_plus_ctgan"]["f1_macro"],
        "alphaCT_brier": result["alpha_plus_ctgan"]["brier"],
    }
    csv_path = out_dir / "ctgan_full_comparable_results.csv"
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    print(f"\nSaved: {csv_path}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
