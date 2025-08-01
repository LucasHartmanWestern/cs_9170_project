import os
import json
import csv
import copy
import torch
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import pandas as pd
from data_processing import get_xy_from_data
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal
from datetime import datetime


def evaluate_model(agent, x: torch.Tensor, y: torch.Tensor, sex_idx: int, mean: bool =True) -> tuple[float, float]:
    # agent.model.eval()
    # with torch.no_grad():
    #     preds = agent.predict(x)

    #     if not isinstance(preds, torch.Tensor):
    #         preds = torch.tensor(preds, device=y.device, dtype=y.dtype)
    
    # # Compute overall MSE
    # mse = F.mse_loss(preds, y).item()
    # mae = F.l1_loss(preds, y).item()

    # # Compute female-only MSE
    # female_mask = y[:, sex_idx] == 1
    # if female_mask.any():
    #     pred_f = preds[female_mask]
    #     y_f = y[female_mask]
    #     female_mse = F.mse_loss(pred_f, y_f).item()
    # else:
    #     female_mse = float('nan')
    # return mse, mae, female_mse
    mf_e, f_e, label = evaluate_se(agent, x, y, sex_idx, mean)
    return mf_e, f_e, label
    


def evaluate_mape(agent, x: torch.Tensor, y: torch.Tensor, sex_idx: int, mean: bool =False) -> tuple[float, float]:
    agent.model.eval()
    with torch.no_grad():
        preds = agent.predict(x)

        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, device=y.device, dtype=y.dtype)
    
    # Compute overall APE
    # Avoid division by zero
    epsilon = 1e-8
    
    # Compute absolute percentage error per element
    ape = torch.abs((y - preds) / (y + epsilon))


    # Compute female-only APE
    female_mask = y[:, sex_idx] == 1
    if female_mask.any():
        pred_f = preds[female_mask]
        y_f = y[female_mask]
        female_ape = torch.abs((y_f - pred_f) / (y_f + epsilon))
    else:
        female_ape = float('nan')

    printl(f'')

    if mean:
        mf_ape = ape.mean().item()
        f_ape = female_ape.mean().item()
    else:
        mf_ape = ape.mean(axis=1)
        f_ape = female_ape.mean(axis=1)
        
    return mf_ape, f_ape


def evaluate_mae(agent, x: torch.Tensor, y: torch.Tensor, sex_idx: int, mean: bool =False) -> tuple[float, float]:
    agent.model.eval()
    with torch.no_grad():
        preds = agent.predict(x)

        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, device=y.device, dtype=y.dtype)
    
    # Compute overall ae
    # Compute mean error per element
    ae = torch.abs((y - preds))


    # Compute female-only AE
    female_mask = y[:, sex_idx] == 1
    if female_mask.any():
        pred_f = preds[female_mask]
        y_f = y[female_mask]
        female_ae = torch.abs((y_f - pred_f))
    else:
        female_ae = float('nan')

    if mean:
        mf_ae = ae.mean().item()
        f_ae = female_ae.mean().item()
    else:
        mf_ae = ae.mean(axis=1)
        f_ae = female_ae.mean(axis=1)
        
    return mf_ae, f_ae

def evaluate_se(agent, x: torch.Tensor, y: torch.Tensor, sex_idx: int, mean: bool =False) -> tuple[float, float]:
    agent.model.eval()
    with torch.no_grad():
        preds = agent.predict(x)

        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, device=y.device, dtype=y.dtype)
    
    # Compute overall ae
    # Compute mean error per element
    se = (y - preds)**2


    # Compute female-only AE
    female_mask = y[:, sex_idx] == 1
    if female_mask.any():
        pred_f = preds[female_mask]
        y_f = y[female_mask]
        female_se = (y_f - pred_f)**2
    else:
        female_ae = float('nan')

    if mean:
        mf_se = se.mean().item()
        f_se = female_se.mean().item()
        label = 'MSE'
    else:
        mf_se = se.mean(axis=1)
        f_se = female_se.mean(axis=1)
        label = 'SE'
        
    return mf_se, f_se, label


def plot_losses(losses):
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
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

    state_vector = torch.cat([timestamp, age, activity_id, mf_ratio, n_samples], dim=0).to(device)
    return state_vector

#m/f ratio reward 
def calculate_mini_reward(synthetic_features, mf_ratio):
    # mf_ratio: 0-dim tensor
    #setting a cap for maximum std
    max_cap = torch.tensor(2.0)
    std = synthetic_features.std(dim=0, unbiased=False).mean()
    # printl(f'std {std}')
    std_term = torch.minimum(max_cap,std)
    gauss    = torch.exp(-((mf_ratio - 0.5)**2) / 0.1)
    mini_reward = (std_term + gauss)
    # printl(f' mini reward {mini_reward:.2f} std = {std:.2f}, gaus {gauss:.2f}')
    return mini_reward


def train_model_baseline(
    agent,
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
    Trains and evaluates three FFNN or LSTM models (baseline, oversample, undersample)
    entirely in torch.
    """
    overall_start_time = time.time()
    # fix seeds
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    # Generating print log
    date = datetime.now()
    printl = print_log(date)

    # 1) Extract numpy, then convert to torch tensors on device
    x_train_df, y_train_df = get_xy_from_data(df_train, target_features)
    x_val_df,   y_val_df   = get_xy_from_data(df_val,   target_features)
    x_test_df,  y_test_df  = get_xy_from_data(df_test,  target_features)

    sex_female_idx = y_train_df.columns.get_loc('Sex - Female')

    x_train = torch.tensor(x_train_df.values, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_df.values, dtype=torch.float32, device=device)
    x_val   = torch.tensor(x_val_df.values,   dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val_df.values,   dtype=torch.float32, device=device)
    x_test  = torch.tensor(x_test_df.values,  dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test_df.values,  dtype=torch.float32, device=device)

    results = {}



    # Experiment 1: baseline
    printl(f"\nTraining {type(agent).__name__} baseline on original data…")
    one_start_time = time.time()
    base_agent = copy.deepcopy(agent)
    results["baseline"],_ = _run_and_eval(base_agent, sex_female_idx, x_train, y_train, x_val, y_val, x_test, y_test, "Baseline", shuffle=shuffle)
    one_end_time = time.time()
    printl(f'Experiment 1 time: {one_end_time - one_start_time}\n')
    # Experiment 2: oversampling minority
    two_start_time = time.time()
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

    printl(f"\nTraining {type(agent).__name__} with minority oversampling…")
    os_agent = copy.deepcopy(agent)
    results["oversample"],_ = _run_and_eval(os_agent, sex_female_idx, sex_female_idx, x_os, y_os, x_val, y_val, x_test, y_test, "Oversampled", shuffle=shuffle)
    two_end_time = time.time()
    printl(f'Experiment 2 time: { two_end_time - two_start_time}\n')
    # Experiment 3: undersampling majority
    three_start_time = time.time()
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

    printl(f"\nTraining {type(agent).__name__} with majority undersampling…")
    us_agent = copy.deepcopy(agent)

    results["undersample"],_ = _run_and_eval(us_agent, sex_female_idx, x_us, y_us, x_val, y_val, x_test, y_test, "Undersampled",shuffle=shuffle)
    three_end_time = time.time()
    printl(f'Experiment 3 time: { three_end_time - three_start_time}\n')

    # Persist
    os.makedirs(save_location, exist_ok=True)
    with open(os.path.join(save_location, "baseline_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    printl(f"Saved all metrics to {save_location}/baseline_metrics.json")

    return results


def _run_and_eval(agent, sex_female_idx, x_tr, y_tr, x_v, y_v, x_te, y_te, tag, shuffle=False):
    print(f"\n--- {tag} ---")
    dataset = TensorDataset(x_tr, y_tr)
    loader = DataLoader(dataset, batch_size=agent.batch_size, shuffle=shuffle)
    losses = agent.train(loader)

    # m_tr, _, fm_tr = evaluate_model(agent, x_tr, y_tr, sex_female_idx)
    # m_v,  _, fm_v  = evaluate_model(agent, x_v, y_v, sex_female_idx)
    # m_te, _, fm_te = evaluate_model(agent, x_te, y_te, sex_female_idx)

    m_tr, fm_tr, _     = evaluate_model(agent, x_tr, y_tr, sex_female_idx)
    m_v, fm_v, _       = evaluate_model(agent, x_v, y_v, sex_female_idx)
    m_te, fm_te, label = evaluate_model(agent, x_te, y_te, sex_female_idx)

    print(f"{tag} Train {label}: {m_tr:.2f} | Female {label}: {fm_tr:.2f}")
    print(f"{tag} Val   {label}: {m_v:.2f}  | Female {label}: {fm_v:.2f}")
    print(f"{tag} Test  {label}: {m_te:.2f} | Female {label}: {fm_te:.2f}")
    return {
        "train": {"mse": m_tr,  "female_mse": fm_tr},
        "val":   {"mse": m_v,   "female_mse": fm_v},
        "test":  {"mse": m_te,  "female_mse": fm_te},
    }




def train_agents(
        df_train, 
        df_val, 
        df_test, 
        target_features,
        dqn_agent, 
        ppo_agent, 
        base_agent, 
        action_col, 
        episodes, 
        synthetic_features_amount, 
        accuracy_reward_multiplier,
        save_location,
        eval_val_only=True,
        show_loss_plots=True,
        seed=42,
        device='cpu',
        shuffle=False
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
    metrics = []

    # Generating print log
    date = datetime.now()
    printl = print_log(date)

    # Location to save results
    save_path = f'{save_location}/training_metrics.csv'
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"previous {save_path} has been removed.")
    elif not os.path.exists(save_location):
        os.makedirs(save_location)

    x_train_df, y_train_df = get_xy_from_data(df_train, target_features)
    x_val_df,   y_val_df   = get_xy_from_data(df_val,   target_features)
    x_test_df,  y_test_df  = get_xy_from_data(df_test,  target_features)

    # Defining columns index
    sex_female_idx = y_train_df.columns.get_loc('Sex - Female')
    sex_female_action = action_col.index('Sex - Female')
    hr_idx         = y_train_df.columns.get_loc('Heart Rate')

    x_train = torch.tensor(x_train_df.values, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_df.values, dtype=torch.float32, device=device)
    x_val   = torch.tensor(x_val_df.values,   dtype=torch.float32, device=device)
    y_val   = torch.tensor(y_val_df.values,   dtype=torch.float32, device=device)
    x_test  = torch.tensor(x_test_df.values,  dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test_df.values,  dtype=torch.float32, device=device)

    #Generate discriminator based on the baseline with oringinal data
    discriminator = copy.deepcopy(base_agent)
    printl(f"\nTraining {type(discriminator).__name__} discriminator on original data…")
    results = _run_and_eval(discriminator, sex_female_idx, x_train, y_train, x_val, y_val, x_test, y_test, "Original Discriminator", shuffle=shuffle)

    N = x_train.size(0)
    mf_ratio = x_train[:, sex_female_idx].mean()
    n_samples  = torch.tensor(0.0, dtype=torch.float32, device=device)
    timestamp_idx = x_train_df.columns.get_loc('Timestamp')
    
    state = generate_state(x_train, timestamp_idx, mf_ratio, n_samples, rng, device)

    for episode in range(1, episodes + 1):
        episode_start_time = time.time()
        printl(f"Episode {episode}/{episodes}: Generating Synthetic Data")

        #Resets
        synthetic_features = torch.zeros(synthetic_features_amount, x_train.shape[1], device=device)
        synthetic_targets  = torch.zeros(synthetic_features_amount, y_train.shape[1], device=device)
        sum_synth_female = 0.0

        states_ep = []
        next_states_ep = []
        continuous_action_ep =[]

        for i in range(synthetic_features_amount):

            # discrete_action = torch.as_tensor(
            #     dqn_agent.predict(state),
            #     dtype=torch.float32,
            #     device=device
            # ).flatten()
            
            # continuous_action = torch.as_tensor(
            #     ppo_agent.predict(state),
            #     dtype=torch.float32,
            #     device=device
            # ).flatten()
            # continuous_action_ep.append(continuous_action)

            # row = torch.zeros(x_train.size(1), device=device)
            # row[sex_female_idx] = discrete_action[0]
            # hr_idx = x_train_df.columns.get_loc("Heart Rate")
            # row[hr_idx]         = discrete_action[1]
            # cont_idx = x_train_df.columns.get_indexer(action_col).tolist()
            # row[cont_idx]       = continuous_action
            # synthetic_features[i] = row


            # synthetic_features[i,sex_female_idx] = F.softmax(continuous_action[-6])
            # synthetic_features[i,hr_idx]         = continuous_action[-5]
            # cont_idx = x_train_df.columns.get_indexer(action_col[:-6]).tolist()
            # synthetic_features[i,cont_idx]       = continuous_action[:-6]

            # # build synthetic label row
            # age = state[3].unsqueeze(0)
            # preds = continuous_action[-4:]
            # tgt_vals = torch.cat([
            #     preds[:2],   # shape (2,)
            #     age,         # shape (1,)
            #     preds[2:]    # shape (2,)
            # ], dim=0)
            # Get state defined fields

            with torch.no_grad():
                mean, log_std = ppo_agent.actor(state)
                dist = Normal(mean, log_std.exp())
                continuous_action = dist.sample()
                log_prob = dist.log_prob(continuous_action).sum()
                value = ppo_agent.critic(state)


            # Constraining categoricals
            # For sex - Female
            continuous_action[sex_female_action] = F.softmax(continuous_action[sex_female_action],dim=0)
            
            timestamp = state[0].unsqueeze(0)
            age = state[1].unsqueeze(0)
            activity_id = state[2].unsqueeze(0)

            #cat feature variables
            features = torch.cat([
                timestamp, #timestamp from state
                activity_id, #activity_id from state
                continuous_action[:-4], #all sensor data + Resting HR and max HR
                # discrete_action[2:4] #"Resting HR", "Max HR",
            ])

            synthetic_features[i] = features

            labels = torch.cat([
                continuous_action[-4:],   #  Height, Weight, Sex, Heart Rate
                age,                   # Age from state
            ], dim=0)          
            synthetic_targets[i] = labels


            sum_synth_female += synthetic_targets[i,sex_female_idx].item() #Get number of female entires
            n_synth = i + 1
            # mf_ratio = (N * train_female_ratio  + sum_synth_female) / (N + n_synth)
      
            # mini_reward = compute_mini_reward(synthetic_features[: i+1 ], mf_ratio)

            done        = (i == synthetic_features_amount - 1)

            #Once all synthetic data samples have been generated
            if done:
                to_print = f"Episode {episode}/{episodes}: Training {type(base_agent).__name__} "
                printl(to_print)
            
                base_agent.reset()

                # concatenate real + synthetic
                combined_data   = torch.cat([x_train, synthetic_features], dim=0)    # (N+n, D)
                combined_labels = torch.cat([y_train, synthetic_targets], dim=0) # (N+n, 5)

                combined_dataset = TensorDataset(combined_data, combined_labels)
                loader = DataLoader(
                    combined_dataset,
                    batch_size=base_agent.batch_size,
                    shuffle=shuffle,
                    generator=cpu_rng,
                    pin_memory=(False)#Don't enable on cpu
                )

                # Train FFNN or LSTM
                losses = base_agent.train(loader)
                printl(f"Episode {episode}/{episodes}: Evaluating {type(base_agent).__name__} ")

                val_e, val_female_e, e_label = evaluate_model(base_agent, x_val, y_val, sex_female_idx, mean=True)

                val_accuracies.append(val_e)
                val_female_accuracies.append(val_female_e)

                #Discriminator evaluation
                # Calculating male female ration on synthetic data
                mf_ratio_synthetic = synthetic_targets[:, sex_female_idx].mean()

                disc_val_e, disc_val_female_e, e_label = evaluate_model(discriminator, 
                                                                    synthetic_features, synthetic_targets,
                                                                    sex_female_idx, mean=False)
                
                printl(f"Episode {episode}/{episodes}: Discriminator on sythetic m-f val " + 
                      f"M{e_label} {disc_val_e.mean():.2f}, female val M{e_label} {disc_val_female_e.mean():.2f}," +
                      f" mf-ratio {100*mf_ratio_synthetic:.2f}% female")
                # if not eval_val_only:
                #     train_mse, train_mae, train_female_mse = evaluate_model(base_agent, x_train, y_train, sex_female_idx)
                #     test_mse, test_mae, test_female_mse = evaluate_model(base_agent, x_test, y_test, sex_female_idx)
                #     train_accuracies.append(train_mse)
                #     test_accuracies.append(test_mse)
                #     train_female_accuracies.append(train_female_mse)
                #     test_female_accuracies.append(test_female_mse)

                
                # Reward is based on validation performance and mini reward
                # reward = (accuracy_reward_multiplier * val_mse * -1) + (mini_reward)

                # reward = -1*(val_mse+val_female_mse)
                w1 = 1
                w2 = 0.25
                w3 = 1
                w4 = 1

                # Current model predictor mape performance
                obj_1 = torch.log(torch.ones(synthetic_features_amount)*val_e + 1e-8)
                # Discriminator or predictor trained with original and evaluated with synthetic data ape
                obj_2 = torch.exp(-disc_val_e)
                # Minority performance with current predictor trained with combined data ape
                obj_3 = torch.log(torch.ones(synthetic_features_amount)*val_female_e + 1e-8)
                # Diversify results
                # NEEDS DISCUSSION! Should we consider the MF only synthetic ratio or the combined dataset ratio?
                mf_ratio = synthetic_targets[:, sex_female_idx].mean()
                diverse_reward = calculate_mini_reward(synthetic_features, mf_ratio)
                obj_4 = torch.ones(synthetic_features_amount)*diverse_reward

                # rewards_ep = w1*obj_1 + w2*obj_2 + w3*obj_3 - w4*obj_4
                rewards_ep = w1*obj_1 + w3*obj_3
                printl(f"Episode {episode}/{episodes}: Reward mean {rewards_ep.mean():.2f}," +
                      f"by objectives: {obj_1.mean().item():.2f} | {obj_2.mean().item():.2f} |" +
                      f"{obj_3.mean().item():.2f} | {obj_4.mean().item():.2f}")
                

                printl(f"Episode {episode}/{episodes} | Reward: {rewards_ep.mean():.2f}")
                printl(f"Val {e_label}: {val_e:.2f} | Val Female {e_label}: {val_female_e:.2f}")

                # if not eval_val_only:
                #     printl(f"Train MSE: {train_mse:.2f} | Train Female MSE: {train_female_mse:.2f}")
                #     printl(f"Test MSE: {test_mse:.2f} | Test Female MSE: {test_female_mse:.2f}")

                dones_ep = torch.zeros(len(rewards_ep), dtype=torch.bool)
                dones_ep[-1] = done

                ppo_agent.finish_trajectory(final_reward=rewards_ep.mean())
                ppo_agent.update()
                # dqn_agent.learn(state, discrete_action, reward, next_state, done)
                # ppo_agent.learn(states_ep, continuous_action_ep, rewards_ep, next_states_ep, dones_ep)  
                rewards.append(rewards_ep.mean().item())

            # else:
            #     reward = mini_reward

            next_state = generate_state(x_train, timestamp_idx, mf_ratio, torch.tensor(i + 1, dtype=torch.float32, device=device), rng, device)
            states_ep.append(state)
            next_states_ep.append(next_state)
            ppo_agent.store(state, continuous_action, done, value.item(), log_prob)
            # dqn_agent.learn(state, discrete_action, reward, next_state, done)
            # ppo_agent.learn(state, continuous_action, reward, next_state, done)

            # rewards.append(reward)
            state = next_state

            


        # Track and print episode time
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_times.append(episode_duration)
        printl(f"Episode {episode}/{episodes} completed in {episode_duration:.2f} seconds.")
        printl("\n--------------------------------\n")
        
        # if eval_val_only:
        #     metrics = {
        #         'rewards': rewards_e,
        #         'val_mse': val_accuracies,
        #         'episode_times': episode_times
        #     }
        # else:
        #     metrics = {
        #         'rewards': rewards,
        #         'train_mse': train_accuracies,
        #         'val_mse': val_accuracies,
        #         'test_mse': test_accuracies,
        #         'train_female_mse': train_female_accuracies,
        #         'val_female_mse': val_female_accuracies,
        #         'test_female_mse': test_female_accuracies,
        #         'episode_times': episode_times
        #     }
        metric = {
                'episode': episode,
                'rewards': rewards_ep.mean().item(),
                f'm-f_val_M{e_label}': val_e,
                f'f_eval_M{e_label}': val_female_e,
                'episode_times': episode_duration
        }
        metrics.append(metric)

        if (episode % 10 == 0) | (episode == episodes):
            save_to_csv(metrics,save_path)
            printl(f"Metrics saved to {save_path}")
            metrics = []

        
        

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    printl(f"All episodes completed in {overall_duration:.2f} seconds.")

    # with open(save_path, 'w') as f:
    #     json.dump(metrics, f)
    return metrics

def print_log(date):
    path = 'logs/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'{path}/{date}-training_logs.txt'
    def printl(to_print):
        with open(filename, 'a') as file:
                print(to_print, file=file)
        print(to_print)

    return printl

def save_to_csv(data, filename, append=True):
    mode = 'a' if append else 'w'
    file_exists = os.path.isfile(filename)
    if isinstance(data[0], dict):
        keys = data[0].keys()  # Extract the headers from the first dictionary
        with open(filename, mode, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            if not append or not file_exists:
                writer.writeheader()  # Write header if not appending or file doesn't exist
            writer.writerows(data)  # Write data rows

