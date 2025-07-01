import os
import math
import random
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_processing import get_xy_from_data
from training        import evaluate_model

def generate_offline_trajectories(
    train_set: pd.DataFrame,
    val_set:   pd.DataFrame,
    test_set:  pd.DataFrame,
    target_features:    list[str],
    base_agent,
    horizon:  int,
    seed:     int,
    device:   str,
    output_path: str
):
    """
    Build & save trajectories of length `horizon`, covering every held-out row once.
    Ensures val‐set rows used as synthetic actions are excluded from evaluation.
    """

    # 1) fix RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2) form held‐out pool = all of val_set + test_set
    pool_df = pd.concat(
        [val_set.reset_index(drop=True),
         test_set.reset_index(drop=True)],
        ignore_index=True
    )
    held_out = pool_df
    N_pool = len(held_out)

    # 3) how many trajectories to cover the pool at least once
    num_trajs = math.ceil(N_pool / horizon)

    # 4) shuffle pool indices for coverage
    indices = list(range(N_pool))
    random.shuffle(indices)
    ptr = 0

    # 5) precompute train & val tensors
    xtr_df, ytr_df = get_xy_from_data(train_set, target_features)
    x_train = torch.tensor(xtr_df.values, dtype=torch.float32, device=device)
    y_train = torch.tensor(ytr_df.values, dtype=torch.float32, device=device)
    N_train = x_train.size(0)

    # For evaluation, start with the full val_set
    xval_df, yval_df = get_xy_from_data(val_set, target_features)
    x_val_full = torch.tensor(xval_df.values, dtype=torch.float32, device=device)
    y_val_full = torch.tensor(yval_df.values, dtype=torch.float32, device=device)

    N_val = len(val_set)  # number of rows in val_set

    # column names needed for state
    ts_col  = 'Timestamp'
    age_col = 'Age'
    act_col = 'Activity ID'
    sex_col = 'Sex - Female'

    train_female_sum = xtr_df[sex_col].sum()

    trajectories = []
    print(f"Generating {num_trajs} trajectories × {horizon} steps each")

    for t_idx in range(num_trajs):
        base_agent.reset()

        obs_list = []
        act_list = []
        rew_list = []

        used_val_indices = []   # track which val rows are used
        n_synth = 0
        synth_f_sum = 0.0

        for step in range(horizon):
            # pick an index
            if ptr < N_pool:
                idx = indices[ptr]
                ptr += 1
            else:
                idx = random.randrange(N_pool)

            row = held_out.iloc[idx]

            # if this row came from val_set (not test_set), record its val‐index
            if idx < N_val:
                used_val_indices.append(idx)

            # build state
            timestamp   = float(row[ts_col])
            age         = float(row[age_col])
            activity_id = float(row[act_col])
            mf_ratio    = (train_female_sum + synth_f_sum) / (N_train + n_synth)
            n_samples   = float(n_synth)

            obs_list.append(np.array([
                timestamp, mf_ratio, n_samples, age, activity_id
            ], dtype=np.float32))

            # record action = full feature+label row
            xr_df, yr_df = get_xy_from_data(row.to_frame().T, target_features)
            action = np.concatenate([
                xr_df.values.flatten(),
                yr_df.values.flatten()
            ]).astype(np.float32)
            act_list.append(action)

            # update synthetic stats
            synth_f_sum += float(row[sex_col])
            n_synth     += 1

            rew_list.append(0.0)  # no reward until final

        # train on combined real+synthetic
        Xs = torch.tensor(
            np.stack(act_list)[:,:xr_df.shape[1]],
            dtype=torch.float32, device=device
        )
        Ys = torch.tensor(
            np.stack(act_list)[:,xr_df.shape[1]:],
            dtype=torch.float32, device=device
        )
        Xc = torch.cat([x_train, Xs], dim=0)
        Yc = torch.cat([y_train, Ys], dim=0)
        dl = DataLoader(
            TensorDataset(Xc, Yc),
            batch_size=Xc.size(0),
            shuffle=True
        )
        _ = base_agent.train(dl)

        # prepare val‐set **excluding** rows we sampled from it
        if used_val_indices:
            mask = np.ones(N_val, dtype=bool)
            mask[used_val_indices] = False
            x_val = x_val_full[mask]
            y_val = y_val_full[mask]
        else:
            x_val = x_val_full
            y_val = y_val_full

        # evaluate
        val_mse, _, _ = evaluate_model(
            base_agent, x_val, y_val, xval_df.columns.get_loc(sex_col)
        )

        obs_arr = np.stack(obs_list, axis=0)      # (T, state_dim)
        act_arr = np.stack(act_list, axis=0)      # (T, action_dim)
        rew_arr = np.array(rew_list, dtype=np.float32)  # (T,)

        next_obs_arr = np.concatenate([obs_arr[1:], obs_arr[-1:]], axis=0)  # (T, state_dim)

        # terminal flag: only the last step is done
        terminals = np.zeros(horizon, dtype=bool)
        terminals[-1] = True


        # final reward = –val_mse
        rew_list[-1] = -float(val_mse)

        trajectories.append({
            "observations":      obs_arr,
            "actions":           act_arr,
            "rewards":           rew_arr,
            "next_observations": next_obs_arr,
            "terminals":         terminals
        })

        print(f" Traj {t_idx+1}/{num_trajs}, val_mse={val_mse:.4f}")

    # save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"Saved {len(trajectories)} trajectories → {output_path}")
