import os
import random
import numpy as np
import torch
import pandas as pd
from dqn_agent2 import DQNAgent
from ppo_agent3 import PPOAgent
from ffnn_agent2 import FFNNAgent
from lstm_agent import LSTMAgent
from data_processing import load_preprocessed_dataset, get_xy_from_data
from training import train_agents, train_model_baseline, evaluate_model
from visualize import plot_rewards, plot_episode_rewards, plot_cumulative_rewards, plot_mse_hist, plot_female_mse_hist, plot_episode_rewards_combined, plot_cumulative_rewards_combined




def main_loop():
    ##### Experiment Config ######
    seed = 123
    shuffle = False
    
    accuracy_reward_multiplier = 10
    synthetic_data_amount = 1000
    num_episodes = 1000
    batch_size = 32
    
    experiment_name = "experiment_1"
    results_folder = "latest_results"

    #Load Data#############
    # # Load the preprocessed dataset
    df = load_preprocessed_dataset()
    
    activity_split = 2
    val_test = df[df['Activity ID'] <= activity_split]
    
    val_test_female = val_test[val_test['Sex - Female'] == 1]
    val_test_male   = val_test[val_test['Sex - Female'] == 0]
    
    def split_half(df):
        n_val = int(len(df)*0.5)
        val_df  = df.iloc[:n_val].copy()   
        test_df = df.iloc[n_val:].copy()
        return val_df, test_df
    
    female_val, female_test = split_half(val_test_female)
    male_val, male_test = split_half(val_test_male)
    
    
    # Step 4: Combine validation and test sets (sex-balanced)
    df_val = pd.concat([ female_val, male_val], ignore_index=True).reset_index(drop=True)
    
    df_test = pd.concat([female_test, male_test], ignore_index=True).reset_index(drop=True)
    
    # Step 5: Training data = everything else
    df_train = df[df['Activity ID'] > activity_split].copy()
    df_train.reset_index(drop=True,inplace=True)
    
    # Shuffle the train set
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Report split sizes
    print(f"\nTrain samples (excluding Activity 1 & 2): {df_train.shape[0]}")
    print(f"Validation samples (Activity 1): {df_val.shape[0]}")
    print(f"Test samples (Activity 2): {df_test.shape[0]}")
    
    def split_half(df):
        n_val = int(len(df)*0.5)
        val_df  = df.iloc[:n_val].copy()   
        test_df = df.iloc[n_val:].copy()
        return val_df, test_df
    
    female_val, female_test = split_half(val_test_female)
    male_val,   male_test   = split_half(val_test_male)
    
    # Step 4: Combine validation and test sets (sex-balanced)
    df_val = pd.concat([female_val, male_val], ignore_index=True).reset_index(drop=True)
    
    df_test = pd.concat([female_test, male_test], ignore_index=True).reset_index(drop=True)
    
    # Step 5: Training data = everything else
    df_train = df.loc[df['Activity ID'] > activity_split].copy()
    # df_train.reset_index(drop=True,inplace=True)
    
    # Shuffle train set
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Report split sizes
    print(f"\nTrain samples (excluding Activity 1 & 2): {df_train.shape[0]}")
    print(f"Validation samples 50% Activities 1 and 2: {df_val.shape[0]}")
    print(f"Test samples 50% Activities 1 and 2: {df_test.shape[0]}")

    ######Initialize System ####
    
    random.seed(seed)
    np.random.seed(seed)
    
    # continuous_columns = [
    # 'Timestamp',  'Hand Sensor - Temperature',
    # 'Hand Sensor - Accelerometer - X', 'Hand Sensor - Accelerometer - Y',
    # 'Hand Sensor - Accelerometer - Z', 'Hand Sensor - Gyroscope - X',
    # 'Hand Sensor - Gyroscope - Y', 'Hand Sensor - Gyroscope - Z',
    # 'Hand Sensor - Magnetometer - X', 'Hand Sensor - Magnetometer - Y',
    # 'Hand Sensor - Magnetometer - Z', 'Chest Sensor - Temperature',
    # 'Chest Sensor - Accelerometer - X', 'Chest Sensor - Accelerometer - Y',
    # 'Chest Sensor - Accelerometer - Z', 'Chest Sensor - Gyroscope - X',
    # 'Chest Sensor - Gyroscope - Y', 'Chest Sensor - Gyroscope - Z',
    # 'Chest Sensor - Magnetometer - X', 'Chest Sensor - Magnetometer - Y',
    # 'Chest Sensor - Magnetometer - Z', 'Ankle Sensor - Temperature',
    # 'Ankle Sensor - Accelerometer - X', 'Ankle Sensor - Accelerometer - Y',
    # 'Ankle Sensor - Accelerometer - Z', 'Ankle Sensor - Gyroscope - X',
    # 'Ankle Sensor - Gyroscope - Y', 'Ankle Sensor - Gyroscope - Z',
    # 'Ankle Sensor - Magnetometer - X', 'Ankle Sensor - Magnetometer - Y',
    # 'Ankle Sensor - Magnetometer - Z'
    # ]
    
    # discrete_columns = [
    #      'Sex - Female', 'Heart Rate', "Resting HR", "Max HR", "Weight", "Height"
    # ]
    
    
    # #discrete action size columns
    # dqn_config = {
    #     'state_size': 5,  
    #     'action_size': len(discrete_columns),  
    #     'hidden_size': 64,
    #     'lr': 1e-2, # HYPERPARAMETER FOR EXPERIMENTS
    #     'gamma': 0.8, # HYPERPARAMETER FOR EXPERIMENTS
    #     'batch_size': batch_size,
    #     'memory_size': 10000,
    #     'epsilon_start': 1.0,
    #     'epsilon_min': 0.01,
    #     'epsilon_decay': 0.995,
    #     'seed': seed
    # }

    #reordering columns to be feature categorical-continous vs target categorical-continous
    col_order = ['Timestamp','Activity ID',
                 'Hand Sensor - Temperature', 'Hand Sensor - Accelerometer - X',
                 'Hand Sensor - Accelerometer - Y', 'Hand Sensor - Accelerometer - Z',
                 'Hand Sensor - Gyroscope - X', 'Hand Sensor - Gyroscope - Y',
                 'Hand Sensor - Gyroscope - Z', 'Hand Sensor - Magnetometer - X',
                 'Hand Sensor - Magnetometer - Y', 'Hand Sensor - Magnetometer - Z',
                 'Chest Sensor - Temperature', 'Chest Sensor - Accelerometer - X',
                 'Chest Sensor - Accelerometer - Y', 'Chest Sensor - Accelerometer - Z',
                 'Chest Sensor - Gyroscope - X', 'Chest Sensor - Gyroscope - Y',
                 'Chest Sensor - Gyroscope - Z', 'Chest Sensor - Magnetometer - X',
                 'Chest Sensor - Magnetometer - Y', 'Chest Sensor - Magnetometer - Z',
                 'Ankle Sensor - Temperature', 'Ankle Sensor - Accelerometer - X',
                 'Ankle Sensor - Accelerometer - Y', 'Ankle Sensor - Accelerometer - Z', 
                 'Ankle Sensor - Gyroscope - X', 'Ankle Sensor - Gyroscope - Y', 
                 'Ankle Sensor - Gyroscope - Z', 'Ankle Sensor - Magnetometer - X',
                 'Ankle Sensor - Magnetometer - Y', 'Ankle Sensor - Magnetometer - Z',
                 'Resting HR', 'Max HR',
                 'Height', 'Weight', 'Sex - Female', 'Heart Rate','Age']
    
    df = df[col_order]

    actions_cols = [
            'Hand Sensor - Temperature', 'Hand Sensor - Accelerometer - X',
             'Hand Sensor - Accelerometer - Y', 'Hand Sensor - Accelerometer - Z',
             'Hand Sensor - Gyroscope - X', 'Hand Sensor - Gyroscope - Y',
             'Hand Sensor - Gyroscope - Z', 'Hand Sensor - Magnetometer - X',
             'Hand Sensor - Magnetometer - Y', 'Hand Sensor - Magnetometer - Z',
             'Chest Sensor - Temperature', 'Chest Sensor - Accelerometer - X',
             'Chest Sensor - Accelerometer - Y', 'Chest Sensor - Accelerometer - Z',
             'Chest Sensor - Gyroscope - X', 'Chest Sensor - Gyroscope - Y',
             'Chest Sensor - Gyroscope - Z', 'Chest Sensor - Magnetometer - X',
             'Chest Sensor - Magnetometer - Y', 'Chest Sensor - Magnetometer - Z',
             'Ankle Sensor - Temperature', 'Ankle Sensor - Accelerometer - X',
             'Ankle Sensor - Accelerometer - Y', 'Ankle Sensor - Accelerometer - Z', 
             'Ankle Sensor - Gyroscope - X', 'Ankle Sensor - Gyroscope - Y', 
             'Ankle Sensor - Gyroscope - Z', 'Ankle Sensor - Magnetometer - X',
             'Ankle Sensor - Magnetometer - Y', 'Ankle Sensor - Magnetometer - Z',
             'Resting HR', 'Max HR',
            'Height', 'Weight', 'Sex - Female', 'Heart Rate'
    ]

    target_features = ['Height', 'Weight', 'Sex - Female', 'Heart Rate','Age']
    # gpu_id = 0
    # device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    #continuous
    # ppo_config = {
    #     'state_size': 5,  
    #     'action_size': len(actions_cols),   
    #     'hidden_size': 64,
    #     'lr': 1e-2, # HYPERPARAMETER FOR EXPERIMENTS
    #     'gamma': 0.8, # HYPERPARAMETER FOR EXPERIMENTS
    #     'clip_epsilon': 0.2,
    #     'update_epochs': 10,
    #     'batch_size': batch_size,
    #     'c1': 0.5,
    #     'c2': 0.01,
    #     'seed': seed
    # }

    ppo_config = {
        'state_size': 5,  
        'action_size': len(actions_cols),   
        # 'hidden_size': 64,
        'lr': 1e-2, # HYPERPARAMETER FOR EXPERIMENTS
        'gamma': 0.8, # HYPERPARAMETER FOR EXPERIMENTS
        'clip_epsilon': 0.2,
        'lam': 0.95,
        # 'update_epochs': 10,
        # 'batch_size': batch_size,
        # 'c1': 0.5,
        # 'c2': 0.01,
        'seed': seed,
        'device':device
    }
    # classes = [1, 2, 3, 17, 16, 13, 4, 7, 6]
    
    ffnn_config = { # DO NOT CHANGE THIS CONFIG
        'input_size': df.shape[1] - 5,
        'hidden_sizes': [16, 16],
        'output_size': 5,
        'learning_rate': 0.001,
        'batch_size': batch_size,
        'epochs': 20,
        'type': 'regression',
        'classes': None,
        'seed': seed,
        'device': device
    }
    
    lstm_config = {
        'input_dim': df.shape[1] - 5,  # input features (excluding targets)
        'hidden_dim': 64,
        'output_dim': 5,               # number of target features
        'num_layers': 3,
        'bidirectional': True,
        'learning_rate': 0.001,
        'batch_size': batch_size,
        'epochs': 100,
        'seed': seed,
        'device': device
    }
    


    
    dqn_agent = None
    ppo_agent = PPOAgent(**ppo_config)
    ffnn_agent = FFNNAgent(**ffnn_config)
    ffnn_agent_og = FFNNAgent(**ffnn_config)
    lstm_agent = LSTMAgent(**lstm_config)



    # train agents stage ###########
    save_path = os.path.join(results_folder, experiment_name, "metrics")
    os.makedirs(save_path, exist_ok=True)
    
    # baseline_results = train_model_baseline(
    #     ffnn_agent_og, df_train, df_val, df_test, target_features,
    #     save_path, show_loss_plots=False, seed=seed, shuffle=shuffle
    # )
    
    training_results = train_agents(
        df_train, df_val, df_test, target_features,
        dqn_agent, ppo_agent, ffnn_agent, 
        actions_cols, num_episodes, synthetic_data_amount, accuracy_reward_multiplier, 
        save_path, show_loss_plots=False, seed=seed
    )
    
    # save trained models
    save_path = os.path.join(results_folder, experiment_name, "saved_models")
    os.makedirs(save_path, exist_ok=True)
    # dqn_agent.save(os.path.join(save_path, "dqn_trained_model.pth"))
    ppo_agent.save(os.path.join(save_path, "ppo_trained_model.pth"))


if __name__ == '__main__':
    main_loop()