import os
import json
import copy
import torch
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import pandas as pd
from data_processing import get_xy_from_data
from torch.utils.data import TensorDataset, DataLoader

def evaluate_ffnn(agent, x_t, y_t, sex_female_idx):
    # Determine the device from the agent’s model
    device = next(agent.model.parameters()).device

    x_t = x_t.to(device)
    y_t = y_t.to(device)

    preds = agent.predict(x_t)
    if isinstance(preds, torch.Tensor):
        preds = preds.to(device)
    else:
        preds = torch.tensor(preds, dtype=torch.float32, device=device)

    # Compute losses
    mse = F.mse_loss(preds, y_t).item()
    mae = F.l1_loss(preds, y_t).item()

    # Female‐specific evaluation
    female_mask = x_t[:, sex_female_idx] == 1
    if female_mask.any():
        fmse = F.mse_loss(preds[female_mask], y_t[female_mask]).item()
    else:
        fmse = float('nan')

    return mse, mae, fmse


def plot_ffnn_losses(losses):
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('FFNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

#RETHINK THIS
def generate_state(tensor, timestamp_idx, mf_ratio, n_samples, rng, device):

    timestamps = tensor[:, timestamp_idx]
    t_min, t_max = timestamps.min().item(), timestamps.max().item()

    timestamp   = (t_max - t_min) * torch.rand(1, generator=rng, device=device) + t_min
    age         = 24.0 + 7.0 * torch.rand(1, generator=rng, device=device)
    activity_id = torch.randint(1, 3, (1,), generator=rng, device=device).float()

    if not torch.is_tensor(mf_ratio):
        mf_ratio = torch.tensor([mf_ratio], dtype=torch.float32, device=device)
    elif mf_ratio.dim() == 0:
        mf_ratio = mf_ratio.unsqueeze(0)

    if not torch.is_tensor(n_samples):
        n_samples = torch.tensor([n_samples], dtype=torch.float32, device=device)
    elif n_samples.dim() == 0:
        n_samples = n_samples.unsqueeze(0)
    
    timestamps = tensor[:, timestamp_idx]
    
    t_min, t_max = timestamps.min().item(), timestamps.max().item()
    timestamp    = (t_max - t_min) * torch.rand(1, generator=rng, device=device) + t_min
    age = 24.0 + 7.0 * torch.rand(1, generator=rng, device=device)
    
    activity_id = torch.randint(1, 3, (1,), generator=rng, device=device).float()

    state_vector = torch.cat([timestamp, mf_ratio, n_samples, age, activity_id], dim=0)
    return state_vector

#m/f ratio reward 
def compute_mini_reward(synthetic_data, mf_ratio):
    # mf_ratio: 0-dim tensor
    #setting a cap for maximum std
    max_cap = 2
    std = synthetic_data.std(dim=0, unbiased=False).mean()
    # print(f'std {std}')
    std_term = min(max_cap,synthetic_data.std(dim=0, unbiased=False).mean())
    gauss    = max_cap*torch.exp(-((mf_ratio - 0.5)**2) / 0.1)
    mini_reward = (std_term + gauss).item()
    # print(f' mini reward {mini_reward:.2f} std = {std:.2f}, gaus {gauss:.2f}')
    return mini_reward


def train_ffnn_baseline(
    ffnn_agent,
    df_train,
    df_val,
    df_test,
    target_features,
    save_location,
    show_loss_plots=True,
    seed=42,
    device='cpu',
    shuffle=False
):
    """
    Trains and evaluates three FFNN models (baseline, oversample, undersample)
    entirely in torch.
    """
    # fix seeds
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    # 1) Extract numpy, then convert to torch tensors on device
    x_train_df, y_train_df = get_xy_from_data(df_train, target_features)
    x_val_df,   y_val_df   = get_xy_from_data(df_val,   target_features)
    x_test_df,  y_test_df  = get_xy_from_data(df_test,  target_features)

    sex_female_idx = x_train_df.columns.get_loc('Sex - Female')

    x_train = torch.tensor(x_train_df.values, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_df.values, dtype=torch.float32, device=device)
    x_val   = torch.tensor(x_val_df.values,   dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val_df.values,   dtype=torch.float32, device=device)
    x_test  = torch.tensor(x_test_df.values,  dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test_df.values,  dtype=torch.float32, device=device)

    results: dict[str, dict[str, dict[str, float]]] = {}

    def _run_and_eval(agent, x_tr, y_tr, x_v, y_v, x_te, y_te, tag):
        print(f"\n--- {tag} ---")
        dataset = TensorDataset(x_tr, y_tr)
        loader = DataLoader(dataset, batch_size=x_tr.size(0), shuffle=True)
        losses = agent.train(loader)
        if show_loss_plots:
            plot_ffnn_losses(losses)
        m_tr, _, fm_tr = evaluate_ffnn(agent, x_tr, y_tr, sex_female_idx)
        m_v,  _, fm_v  = evaluate_ffnn(agent, x_tr, y_tr, sex_female_idx)
        m_te, _, fm_te = evaluate_ffnn(agent, x_tr, y_tr, sex_female_idx)
        print(f"{tag} Train MSE: {m_tr:.4f} | Female MSE: {fm_tr:.4f}")
        print(f"{tag} Val   MSE: {m_v:.4f}  | Female MSE: {fm_v:.4f}")
        print(f"{tag} Test  MSE: {m_te:.4f} | Female MSE: {fm_te:.4f}\n")
        return {
            "train": {"mse": m_tr,  "female_mse": fm_tr},
            "val":   {"mse": m_v,   "female_mse": fm_v},
            "test":  {"mse": m_te,  "female_mse": fm_te},
        }

    # Experiment 1: baseline
    print("\nTraining FFNN baseline on original data…")
    base_agent = copy.deepcopy(ffnn_agent)
    results["baseline"] = _run_and_eval(
        base_agent, x_train, y_train, x_val, y_val, x_test, y_test, "Baseline")

    # Experiment 2: oversampling minority
    df_min = df_train[df_train["Sex - Female"] == 1]
    df_maj = df_train[df_train["Sex - Female"] == 0]
    maj_size = len(df_maj)
    if shuffle:
        df_min_os  = df_min.sample(n=maj_size, replace=True,  random_state=seed)
        df_train_os = pd.concat([df_maj, df_min_os]).sample(frac=1,random_state=seed).reset_index(drop=True)
    else:
        # Get how many times you need to repeat the minority class
        repeat_times = maj_size // len(df_min)
        remainder = maj_size % len(df_min)
        # Repeat the full minority class as needed
        df_min_os = pd.concat([df_min] * repeat_times + [df_min.iloc[:remainder]], ignore_index=True)
        # Concatenate with majority class
        df_train_os = pd.concat([df_maj, df_min_os], ignore_index=True)
        df_train_os.reset_index(inplace=True, drop=True)
    x_os_df, y_os_df = get_xy_from_data(df_train_os, target_features)
    x_os = torch.tensor(x_os_df.values, dtype=torch.float32, device=device)
    y_os = torch.tensor(y_os_df.values, dtype=torch.float32, device=device)

    print("\nTraining FFNN with minority oversampling…")
    os_agent = copy.deepcopy(ffnn_agent)
    results["oversample"] = _run_and_eval(
        os_agent, x_os, y_os, x_val, y_val, x_test, y_test, "Oversampled")

    # Experiment 3: undersampling majority
    min_size = len(df_min)
    if shuffle:
        df_maj_us  = df_maj.sample(n=min_size, random_state=seed)
        df_train_us = pd.concat([df_min, df_maj_us]).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        rand_begin = int(torch.randint(0, maj_size - min_size, (1,), generator=rng).item())
        df_maj_us  = df_maj.iloc[rand_begin:(rand_begin+min_size)]
        df_train_us =  pd.concat([df_min, df_maj_us]).reset_index(drop=True)
        
    x_us_df, y_us_df = get_xy_from_data(df_train_us, target_features)
    x_us = torch.tensor(x_us_df.values, dtype=torch.float32, device=device)
    y_us = torch.tensor(y_us_df.values, dtype=torch.float32, device=device)

    print("\nTraining FFNN with majority undersampling…")
    us_agent = copy.deepcopy(ffnn_agent)
    results["undersample"] = _run_and_eval(
        us_agent, x_us, y_us, x_val, y_val, x_test, y_test, "Undersampled")

    # Persist
    os.makedirs(save_location, exist_ok=True)
    with open(os.path.join(save_location, "baseline_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved all metrics to {save_location}/baseline_metrics.json")

    return results




def train_agents(
        df_train, 
        df_val, 
        df_test, 
        target_features,
        dqn_agent, 
        ppo_agent, 
        ffnn_agent, 
        continuous_columns, 
        episodes, 
        synthetic_data_amount, 
        accuracy_reward_multiplier,
        save_location,
        eval_val_only=True,
        show_loss_plots=True,
        seed=42,
        device='cpu'
    ):
    overall_start_time = time.time()

    torch.manual_seed(seed)
    rng = torch.Generator(device=device).manual_seed(seed)
    # For the DataLoader shuffle
    cpu_rng = torch.Generator(device='cpu').manual_seed(seed)

    
    rewards = []
    val_accuracies = []
    test_accuracies = []
    train_accuracies = []
    val_female_accuracies = []
    test_female_accuracies = []
    train_female_accuracies = []

    episode_times = []

    x_train_df, y_train_df = get_xy_from_data(df_train, target_features)
    x_val_df,   y_val_df   = get_xy_from_data(df_val,   target_features)
    x_test_df,  y_test_df  = get_xy_from_data(df_test,  target_features)

    sex_female_idx = x_train_df.columns.get_loc('Sex - Female')
    hr_idx         = x_train_df.columns.get_loc('Heart Rate')

    x_train = torch.tensor(x_train_df.values, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_df.values, dtype=torch.float32, device=device)
    x_val   = torch.tensor(x_val_df.values,   dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val_df.values,   dtype=torch.float32, device=device)
    x_test  = torch.tensor(x_test_df.values,  dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test_df.values,  dtype=torch.float32, device=device)


    N = x_train.size(0)
    mf_ratio = train_female_ratio  = x_train[:, sex_female_idx].mean()
    n_samples  = torch.tensor(0.0, dtype=torch.float32, device=device)
    timestamp_idx = x_train_df.columns.get_loc('Timestamp')
    
    state = generate_state(x_train, timestamp_idx, mf_ratio, n_samples, rng, device)

    for episode in range(episodes):
        episode_start_time = time.time()
        print(f"Episode {episode + 1}/{episodes}: Generating Synthetic Data")

        #Resets
        synthetic_data = torch.empty(synthetic_data_amount, x_train.shape[1], device=device)
        synthetic_labels = torch.empty(synthetic_data_amount, y_train.shape[1], device=device)
        sum_synth_female = 0.0

        for i in range(synthetic_data_amount):

            discrete_action = torch.as_tensor(
                dqn_agent.predict(state),
                dtype=torch.float32,
                device=device
            ).flatten()
            
            continuous_action = torch.as_tensor(
                ppo_agent.predict(state),
                dtype=torch.float32,
                device=device
            ).flatten()

            row = torch.zeros(x_train.size(1), device=device)
            row[sex_female_idx] = discrete_action[0]
            hr_idx = x_train_df.columns.get_loc("Heart Rate")
            row[hr_idx]           = discrete_action[1]
            cont_idx = x_train_df.columns.get_indexer(continuous_columns).tolist()
            row[cont_idx]         = continuous_action
            synthetic_data[i] = row

            # build synthetic label row
            age = state[3].unsqueeze(0)
            preds = discrete_action[2:6]
            tgt_vals = torch.cat([
                preds[:2],   # shape (2,)
                age,         # shape (1,)
                preds[2:]    # shape (2,)
            ], dim=0)
            
            synthetic_labels[i] = tgt_vals

            sum_synth_female += row[sex_female_idx].item()
            n_synth = i + 1
            mf_ratio = (N * train_female_ratio  + sum_synth_female) / (N + n_synth)
      
            mini_reward = compute_mini_reward(synthetic_data[: i+1 ], mf_ratio)

            done        = (i == synthetic_data_amount - 1)

            #Once all synthetic data samples have been generated
            if done:
                print(f"Episode {episode + 1}/{episodes}: Training FFNN")
                
                ffnn_agent.reset()

                # concatenate real + synthetic
                combined_data   = torch.cat([x_train, synthetic_data], dim=0)    # (N+n, D)
                combined_labels = torch.cat([y_train, synthetic_labels], dim=0) # (N+n, 5)

                combined_dataset = TensorDataset(combined_data, combined_labels)
                loader = DataLoader(
                    combined_dataset,
                    batch_size=combined_data.size(0),
                    shuffle=True,
                    generator=cpu_rng,
                    pin_memory=(False)#Don't enable on cpu
                )

                # Train FFNN
                losses = ffnn_agent.train(loader)
                if show_loss_plots:
                    plot_ffnn_losses(losses)

                print(f"Episode {episode + 1}/{episodes}: Evaluating FFNN")

                val_mse, val_mae, val_female_mse = evaluate_ffnn(ffnn_agent, x_val, y_val, sex_female_idx)
                val_accuracies.append(val_mse)
                val_female_accuracies.append(val_female_mse)
                if not eval_val_only:
                    train_mse, train_mae, train_female_mse = evaluate_ffnn(ffnn_agent, x_train, y_train, sex_female_idx)
                    test_mse, test_mae, test_female_mse = evaluate_ffnn(ffnn_agent, x_test, y_test, sex_female_idx)
                    train_accuracies.append(train_mse)
                    test_accuracies.append(test_mse)
                    train_female_accuracies.append(train_female_mse)
                    test_female_accuracies.append(test_female_mse)
                # Reward is based on validation performance and mini reward
                reward = (accuracy_reward_multiplier * val_mse * -1) + (mini_reward)
                print(f'mini reward: {mini_reward}')

                print(f"Episode {episode + 1}/{episodes} | Reward: {reward:.4f}")
                print(f"Val MSE: {val_mse:.4f} | Val Female MSE: {val_female_mse:.4f}")

                if not eval_val_only:
                    print(f"Train MSE: {train_mse:.4f} | Train Female MSE: {train_female_mse:.4f}")
                    print(f"Test MSE: {test_mse:.4f} | Test Female MSE: {test_female_mse:.4f}")
                print("\n--------------------------------\n")

            else:
                reward = mini_reward

            next_state = generate_state(x_train, timestamp_idx, mf_ratio, torch.tensor(len(synthetic_data)+1., dtype=torch.float32, device=device), rng, device)
            dqn_agent.learn(state, discrete_action, reward, next_state, done)
            ppo_agent.learn(state, continuous_action, reward, next_state, done)

            rewards.append(reward)
            state = next_state

        # Track and print episode time
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_times.append(episode_duration)
        print(f"Episode {episode + 1}/{episodes} completed in {episode_duration:.2f} seconds.")

        if eval_val_only:
            metrics = {
                'rewards': rewards,
                'val_mse': val_accuracies,
                'episode_times': episode_times
            }
        else:
            metrics = {
                'rewards': rewards,
                'train_mse': train_accuracies,
                'val_mse': val_accuracies,
                'test_mse': test_accuracies,
                'train_female_mse': train_female_accuracies,
                'val_female_mse': val_female_accuracies,
                'test_female_mse': test_female_accuracies,
                'episode_times': episode_times
            }

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    print(f"All episodes completed in {overall_duration:.2f} seconds.")

    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, 'training_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {save_path}")
    return metrics

