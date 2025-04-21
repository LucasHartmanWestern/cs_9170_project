import os
import json
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_processing import get_xy_from_data


def evaluate_ffnn(ffnn_agent, data, labels):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()
    if isinstance(labels, (pd.DataFrame, pd.Series)):
        labels = labels.to_numpy()

    predictions = ffnn_agent.predict(data)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)

    # Optional: female-specific evaluation
    female_mask = data[:, -1] == 1
    if female_mask.sum() > 0:
        female_preds = predictions[female_mask]
        female_labels = labels[female_mask]
        female_mse = mean_squared_error(female_labels, female_preds)
    else:
        female_mse = float('nan')

    return mse, mae, female_mse


def plot_ffnn_losses(losses):
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('FFNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def generate_state(df, mf_ratio, n_samples):
    timestamp = np.random.uniform(df['Timestamp'].min(), df['Timestamp'].max())
    male_female_ratio = mf_ratio 
    num_samples = n_samples
    age = np.random.uniform(24, 31)
    activity_id = np.random.choice([1, 2])
    return np.array([timestamp, male_female_ratio, num_samples, age, activity_id])


def compute_mini_reward(synthetic_data, mf_ratio):
    column_std = np.std(synthetic_data, axis=0).mean()
    gaussian_penalty = np.exp(-((mf_ratio - 0.5) ** 2) / 0.1)
    return column_std + gaussian_penalty


def train_ffnn_baseline_OLD(
        x_train, 
        y_train, 
        x_val, 
        y_val, 
        x_test, 
        y_test, 
        ffnn_agent,
        save_location
    ):

    # Train on real data only
    print("\nTraining FFNN on real data only (no synthetic data)...")
    losses = ffnn_agent.train(x_train.to_numpy(), y_train.to_numpy())
    plot_ffnn_losses(losses)

    # Evaluate on all splits
    train_mse, train_mae, train_female_mse = evaluate_ffnn(ffnn_agent, x_train, y_train)
    val_mse, val_mae, val_female_mse = evaluate_ffnn(ffnn_agent, x_val, y_val)
    test_mse, test_mae, test_female_mse = evaluate_ffnn(ffnn_agent, x_test, y_test)

    # Print results
    print("\n========== FFNN Baseline (No Synthetic Data) ==========")
    print(f"Train MSE: {train_mse:.4f} | Train MAE: {train_mae:.4f} | Female MSE: {train_female_mse:.4f}")
    print(f"Val   MSE: {val_mse:.4f} | Val   MAE: {val_mae:.4f} | Female MSE: {val_female_mse:.4f}")
    print(f"Test  MSE: {test_mse:.4f} | Test  MAE: {test_mae:.4f} | Female MSE: {test_female_mse:.4f}")
    print("=======================================================\n")

    metrics = {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'train_female_mse': train_female_mse,
        'val_female_mse': val_female_mse,
        'test_female_mse': test_female_mse
    }

    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, 'baseline_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {save_path}")

    return metrics


def train_ffnn_baseline(
        ffnn_agent, 
        df_train, 
        df_val, 
        df_test, 
        target_features,
        save_location,
        show_loss_plots=True
    ):
    """
    Trains and evaluates three FFNN models:
      1. Baseline (original training set)
      2. Baseline with minority oversampling (resampling)
      3. Baseline with majority elimination (undersampling)
      
    Args:
        ffnn_agent: The feed-forward neural network agent (untrained).
        df_train: Full training DataFrame (should include "Sex - Female" column).
        df_val: Full validation DataFrame.
        df_test: Full testing DataFrame.
        target_features: List of feature names to be used for training.
        
    Returns:
        A dictionary containing train, validation, and test errors for each method.
    """
    # ---- Prepare baseline splits using the helper ----
    x_train, y_train = get_xy_from_data(df_train, target_features)
    x_val, y_val = get_xy_from_data(df_val, target_features)
    x_test, y_test = get_xy_from_data(df_test, target_features)
    
    # ---------------------------------------------------------
    # Experiment 1: Baseline (No rebalancing)
    # ---------------------------------------------------------
    print("\nTraining FFNN baseline on original training data (no rebalancing)...")
    baseline_agent = copy.deepcopy(ffnn_agent)
    baseline_losses = baseline_agent.train(x_train.to_numpy(), y_train.to_numpy())

    if show_loss_plots:
        plot_ffnn_losses(baseline_losses)

    baseline_train_mse, baseline_train_mae, baseline_train_female_mse = evaluate_ffnn(baseline_agent, x_train, y_train)
    baseline_val_mse, baseline_val_mae, baseline_val_female_mse = evaluate_ffnn(baseline_agent, x_val, y_val)
    baseline_test_mse, baseline_test_mae, baseline_test_female_mse = evaluate_ffnn(baseline_agent, x_test, y_test)

    print("\n========== FFNN Baseline (No Rebalancing) ==========")
    print(f"Train MSE: {baseline_train_mse:.4f} | Train MAE: {baseline_train_mae:.4f} | Female MSE: {baseline_train_female_mse:.4f}")
    print(f"Val   MSE: {baseline_val_mse:.4f} | Val   MAE: {baseline_val_mae:.4f} | Female MSE: {baseline_val_female_mse:.4f}")
    print(f"Test  MSE: {baseline_test_mse:.4f} | Test  MAE: {baseline_test_mae:.4f} | Female MSE: {baseline_test_female_mse:.4f}")
    print("=======================================================\n")
    
    # ---------------------------------------------------------
    # Experiment 2: Baseline with Minority Oversampling (Resampling)
    # ---------------------------------------------------------
    # Split the training set based on "Sex - Female"
    df_minority = df_train[df_train["Sex - Female"] == 1]
    df_majority = df_train[df_train["Sex - Female"] == 0]
    
    # Oversample minority: sample with replacement until the counts are equal
    df_minority_oversampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)
    # Combine oversampled minority with majority and shuffle
    df_train_oversampled = pd.concat([df_majority, df_minority_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    x_train_oversampled, y_train_oversampled = get_xy_from_data(df_train_oversampled, target_features)
    
    print("\nTraining FFNN on training data with minority oversampling (resampling)...")
    resampling_agent = copy.deepcopy(ffnn_agent)
    resampling_losses = resampling_agent.train(x_train_oversampled.to_numpy(), y_train_oversampled.to_numpy())

    if show_loss_plots:
        plot_ffnn_losses(resampling_losses)

    resamp_train_mse, resamp_train_mae, resamp_train_female_mse = evaluate_ffnn(resampling_agent, x_train_oversampled, y_train_oversampled)
    resamp_val_mse, resamp_val_mae, resamp_val_female_mse = evaluate_ffnn(resampling_agent, x_val, y_val)
    resamp_test_mse, resamp_test_mae, resamp_test_female_mse = evaluate_ffnn(resampling_agent, x_test, y_test)
    
    print("\n========== FFNN with Minority Oversampling (Resampling) ==========")
    print(f"Train MSE: {resamp_train_mse:.4f} | Train MAE: {resamp_train_mae:.4f} | Female MSE: {resamp_train_female_mse:.4f}")
    print(f"Val   MSE: {resamp_val_mse:.4f} | Val   MAE: {resamp_val_mae:.4f} | Female MSE: {resamp_val_female_mse:.4f}")
    print(f"Test  MSE: {resamp_test_mse:.4f} | Test  MAE: {resamp_test_mae:.4f} | Female MSE: {resamp_test_female_mse:.4f}")
    print("=======================================================\n")
    
    # ---------------------------------------------------------
    # Experiment 3: Baseline with Majority Elimination (Undersampling)
    # ---------------------------------------------------------
    # Undersample majority: take a subset equal to the number of minority samples
    df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=42)
    df_train_undersampled = pd.concat([df_minority, df_majority_undersampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    x_train_undersampled, y_train_undersampled = get_xy_from_data(df_train_undersampled, target_features)
    
    print("\nTraining FFNN on training data with majority elimination (undersampling)...")
    elimination_agent = copy.deepcopy(ffnn_agent)
    elimination_losses = elimination_agent.train(x_train_undersampled.to_numpy(), y_train_undersampled.to_numpy())

    if show_loss_plots:
        plot_ffnn_losses(elimination_losses)

    elim_train_mse, elim_train_mae, elim_train_female_mse = evaluate_ffnn(elimination_agent, x_train_undersampled, y_train_undersampled)
    elim_val_mse, elim_val_mae, elim_val_female_mse = evaluate_ffnn(elimination_agent, x_val, y_val)
    elim_test_mse, elim_test_mae, elim_test_female_mse = evaluate_ffnn(elimination_agent, x_test, y_test)
    
    print("\n========== FFNN with Majority Elimination (Undersampling) ==========")
    print(f"Train MSE: {elim_train_mse:.4f} | Train MAE: {elim_train_mae:.4f} | Female MSE: {elim_train_female_mse:.4f}")
    print(f"Val   MSE: {elim_val_mse:.4f} | Val   MAE: {elim_val_mae:.4f} | Female MSE: {elim_val_female_mse:.4f}")
    print(f"Test  MSE: {elim_test_mse:.4f} | Test  MAE: {elim_test_mae:.4f} | Female MSE: {elim_test_female_mse:.4f}")
    print("=======================================================\n")

    results = {
        "baseline": {
            "train": {
                "mse": baseline_train_mse, 
                "mae": baseline_train_mae, 
                "female_mse": baseline_train_female_mse
            },
            "val": {
                "mse": baseline_val_mse, 
                "mae": baseline_val_mae, 
                "female_mse": baseline_val_female_mse
            },
            "test": {
                "mse": baseline_test_mse, 
                "mae": baseline_test_mae, 
                "female_mse": baseline_test_female_mse
            },
        },
        "resampling": {
            "train": {
                "mse": resamp_train_mse, 
                "mae": resamp_train_mae, 
                "female_mse": resamp_train_female_mse
            },
            "val": {
                "mse": resamp_val_mse, 
                "mae": resamp_val_mae, 
                "female_mse": resamp_val_female_mse
            },
            "test": {
                "mse": resamp_test_mse, 
                "mae": resamp_test_mae, 
                "female_mse": resamp_test_female_mse
            },
        },
        "elimination": {
            "train": {
                "mse": elim_train_mse, 
                "mae": elim_train_mae, 
                "female_mse": elim_train_female_mse
            },
            "val": {
                "mse": elim_val_mse, 
                "mae": elim_val_mae, 
                "female_mse": elim_val_female_mse
            },
            "test": {
                "mse": elim_test_mse, 
                "mae": elim_test_mae, 
                "female_mse": elim_test_female_mse
            },
        }
    }

    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, 'baseline_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(results, f)
    print(f"Basline results saved to {save_path}")

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
        show_loss_plots=True
    ):
    rewards = []
    val_accuracies = []
    test_accuracies = []
    train_accuracies = []
    val_female_accuracies = []
    test_female_accuracies = []
    train_female_accuracies = []

    synthetic_data = []
    synthetic_labels = []

    # ---- Prepare baseline splits using the helper ----
    x_train, y_train = get_xy_from_data(df_train, target_features)
    x_val, y_val = get_xy_from_data(df_val, target_features)
    x_test, y_test = get_xy_from_data(df_test, target_features)

    # Initial male-female ratio
    sex_female_idx = x_train.columns.get_loc('Sex - Female')
    mf_ratio = np.mean(x_train.iloc[:, sex_female_idx])
    state = generate_state(x_train, mf_ratio, 0)

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}: Generating Synthetic Data")
        for i in range(synthetic_data_amount):
            if synthetic_data:
                synthetic_array = np.array(synthetic_data)
                if synthetic_array.ndim == 1:
                    synthetic_array = synthetic_array.reshape(1, -1)
                combined_array = np.vstack([x_train.to_numpy(), synthetic_array])
                combined = pd.DataFrame(combined_array, columns=x_train.columns)
            else:
                combined = x_train.copy()

            sex_female_idx = combined.columns.get_loc('Sex - Female')
            mf_ratio = np.mean(combined.iloc[:, sex_female_idx])

            # Predict actions from RL agents
            discrete_action = np.array(dqn_agent.predict(state), ndmin=1).flatten()

            # First 2 values are features: 'Sex - Female', 'Heart Rate'
            sex_value = discrete_action[0]
            heart_rate = discrete_action[1]

            # Predicted target values: Resting HR, Max HR, Weight, Height (4 values)
            predicted_targets = discrete_action[2:6]

            # Age comes from the state (4th element)
            age_from_state = state[3]

            # Combine into full target: [Resting HR, Max HR, Age, Weight, Height]
            target_values = np.insert(predicted_targets, 2, age_from_state)  # insert age at index 2
            # Resulting shape: (5,) — matches label format

            # Predict continuous features
            continuous_action = np.array(ppo_agent.predict(state), ndmin=1)  # shape (1, num_continuous_features)

            # Create synthetic feature row
            synthetic_row = np.zeros(x_train.shape[1])

            # Get column indices
            discrete_indices = x_train.columns.get_indexer(['Sex - Female', 'Heart Rate'])
            continuous_indices = x_train.columns.get_indexer(continuous_columns)

            # Assign values to synthetic row
            synthetic_row[discrete_indices[0]] = sex_value
            synthetic_row[discrete_indices[1]] = heart_rate
            synthetic_row[continuous_indices] = continuous_action.flatten()

            # Add to synthetic dataset
            synthetic_data.append(synthetic_row)
            synthetic_labels.append(target_values)

            mini_reward = compute_mini_reward(np.array(synthetic_data), mf_ratio)
            done = i == synthetic_data_amount - 1


            if done:
                print(f"Episode {episode + 1}/{episodes}: Training FFNN")
                
                ffnn_agent.reset()

                synthetic_data_np = np.array(synthetic_data)                    # (n_samples, num_features)
                synthetic_labels_np = np.array(synthetic_labels).reshape(-1, 5) # (n_samples, 5)

                combined_data = np.vstack([x_train.to_numpy(), synthetic_data_np])
                combined_labels = np.vstack([y_train, synthetic_labels_np])


                # Shuffle combined training data
                indices = np.arange(combined_data.shape[0])
                np.random.shuffle(indices)
                combined_data = combined_data[indices]
                combined_labels = combined_labels[indices]

                # Train FFNN
                losses = ffnn_agent.train(combined_data, combined_labels)
                if show_loss_plots:
                    plot_ffnn_losses(losses)

                print(f"Episode {episode + 1}/{episodes}: Evaluating FFNN")

                train_mse, train_mae, train_female_mse = evaluate_ffnn(ffnn_agent, x_train, y_train)
                val_mse, val_mae, val_female_mse = evaluate_ffnn(ffnn_agent, x_val, y_val)
                test_mse, test_mae, test_female_mse = evaluate_ffnn(ffnn_agent, x_test, y_test)

                # Reward is based on validation performance and mini reward
                reward = (accuracy_reward_multiplier * val_mse * -1) + (mini_reward)

                train_accuracies.append(train_mse)
                val_accuracies.append(val_mse)
                test_accuracies.append(test_mse)
                train_female_accuracies.append(train_female_mse)
                val_female_accuracies.append(val_female_mse)
                test_female_accuracies.append(test_female_mse)

                print(f"Episode {episode + 1}/{episodes} | Reward: {reward:.4f}")
                print(f"Train MSE: {train_mse:.4f} | Train Female MSE: {train_female_mse:.4f}")
                print(f"Val MSE: {val_mse:.4f} | Val Female MSE: {val_female_mse:.4f}")
                print(f"Test MSE: {test_mse:.4f} | Test Female MSE: {test_female_mse:.4f}")
                print("\n--------------------------------\n")


                synthetic_data = []
                synthetic_labels = []
            else:
                reward = mini_reward

            next_state = generate_state(x_train, mf_ratio, len(synthetic_data) + 1)
            dqn_agent.learn(state, discrete_action, reward, next_state, done)
            ppo_agent.learn(state, continuous_action, reward, next_state, done)

            rewards.append(reward)
            state = next_state

        metrics = {
            'rewards': rewards,
            'train_mse': train_accuracies,
            'val_mse': val_accuracies,
            'test_mse': test_accuracies,
            'train_female_mse': train_female_accuracies,
            'val_female_mse': val_female_accuracies,
            'test_female_mse': test_female_accuracies
        }


    os.makedirs(save_location, exist_ok=True)
    save_path = os.path.join(save_location, 'training_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {save_path}")

    return metrics

