import json
import numpy as np
import matplotlib.pyplot as plt


def load_agent_metrics(metrics_path):
    """
    Loads the training metrics from the specified JSON file.

    :param metrics_path: Path to the JSON file containing the metrics
    :return: Dictionary containing the metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def plot_rewards(save_path='training_metrics.json'):
    """
    Plots rewards from the saved training metrics JSON file.

    :param save_path: Path to the saved metrics file
    """
    # Load the agent metrics from the saved file
    metrics = load_agent_metrics(save_path)

    rewards = metrics['rewards']

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards', color='tab:blue')
    plt.xlabel('Timestep')
    plt.ylabel('Rewards')
    plt.title('Training Rewards Over Timesteps')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rewards_per_episode(save_path='training_metrics.json'):
    """
    Plots rewards from the saved training metrics JSON file.

    :param save_path: Path to the saved metrics file
    """
    # Load the agent metrics from the saved file
    metrics = load_agent_metrics(save_path)

    rewards = metrics['rewards']

    episode_rewards = []
    for i in range(0, 100):
        episode_rewards.append(sum(rewards[i * 1000:(i + 1) * 1000]) / 1000)

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Rewards', color='tab:blue')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Training Rewards Over Episodes')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rewards_cum_avg(save_path='training_metrics.json'):
    """
    Plots rewards from the saved training metrics JSON file.

    :param save_path: Path to the saved metrics file
    """
    # Load the agent metrics from the saved file
    metrics = load_agent_metrics(save_path)

    rewards = metrics['rewards']

    episode_rewards = []
    for i in range(0, 100):
        episode_rewards.append(np.mean(rewards[i * 1000:(i + 1) * 1000]))

    cum_avg_rewards = np.cumsum(episode_rewards) / np.arange(1, 101)

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(cum_avg_rewards, label='Rewards', color='tab:blue')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Cumulative Average Training Rewards Over Episodes')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mse_histogram(save_path='training_metrics.json'):
    """
    Plots bar charts for the MSE values of Train, Validation, and Test datasets
    from the saved training metrics JSON file.

    :param save_path: Path to the saved metrics file
    """
    # Load the agent metrics from the saved file
    metrics = load_agent_metrics(save_path)

    # Extract MSE values for Train, Validation, and Test
    train_mse = metrics['train_mse']
    val_mse = metrics['val_mse']
    test_mse = metrics['test_mse']

    # Compute the mean MSE for each dataset
    mean_train_mse = sum(train_mse) / len(train_mse)
    mean_val_mse = sum(val_mse) / len(val_mse)
    mean_test_mse = sum(test_mse) / len(test_mse)

    # Plot bar chart for the mean MSE values of each dataset
    plt.figure(figsize=(10, 5))

    # Plot bar chart for Mean MSE for Train, Validation, and Test
    bar_container = plt.bar(['Train', 'Validation', 'Test'],
                            [mean_train_mse, mean_val_mse, mean_test_mse],
                            color=['tab:blue', 'tab:orange', 'tab:green'])
    plt.bar_label(bar_container, fmt="{:.2f}")

    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Mean MSE Value')
    plt.title('Mean MSE Values for Train, Validation, and Test')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_female_mse_histogram(save_path='training_metrics.json'):
    """
    Plots histograms for the MSE values of Train Female, Validation Female, and Test Female
    from the saved training metrics JSON file as bar plots.

    :param save_path: Path to the saved metrics file
    """
    # Load the agent metrics from the saved file
    metrics = load_agent_metrics(save_path)

    # Extract MSE values for Train Female, Validation Female, and Test Female
    train_female_mse = metrics['train_female_mse']
    val_female_mse = metrics['val_female_mse']
    test_female_mse = metrics['test_female_mse']

    mean_train_female_mse = sum(train_female_mse) / len(train_female_mse)
    mean_val_female_mse = sum(val_female_mse) / len(val_female_mse)
    mean_test_female_mse = sum(test_female_mse) / len(test_female_mse)

    # Plot bar charts for the MSE values for each dataset
    plt.figure(figsize=(10, 5))

    # Plot Regular Female MSE for Train, Validation, and Test
    bar_container = plt.bar(['Train', 'Validation', 'Test'],
                            [mean_train_female_mse, mean_val_female_mse, mean_test_female_mse],
                            color=['tab:red', 'tab:purple', 'tab:olive'])
    plt.bar_label(bar_container, fmt="{:.2f}")

    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Mean MSE Value')
    plt.title('Mean Female MSE Values for Train, Validation, and Test')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_baseline_mse_histogram(baseline_results):
    """
    Plots histograms for the MSE of Train, Validation, and Test datasets from the baseline FFNN model.
    Includes histograms for both regular MSE and Female MSE.
    """

    # Extract MSE values for Train, Validation, and Test (Regular and Female MSE)
    train_mse = baseline_results['train'][0]
    val_mse = baseline_results['val'][0]
    test_mse = baseline_results['test'][0]

    train_female_mse = baseline_results['train'][2]
    val_female_mse = baseline_results['val'][2]
    test_female_mse = baseline_results['test'][2]

    # Plot histograms for Regular MSE
    plt.figure(figsize=(10, 5))

    # Plot Regular MSE (Train, Validation, Test) in the first subplot
    plt.subplot(1, 2, 1)
    plt.bar(['Train', 'Validation', 'Test'], [train_mse, val_mse, test_mse],
            color=['tab:blue', 'tab:orange', 'tab:green'])
    plt.xlabel('Dataset')
    plt.ylabel('MSE')
    plt.title('Regular MSE for Train, Validation, and Test')

    # Plot Female MSE (Train, Validation, Test) in the second subplot
    plt.subplot(1, 2, 2)
    plt.bar(['Train', 'Validation', 'Test'], [train_female_mse, val_female_mse, test_female_mse],
            color=['tab:red', 'tab:purple', 'tab:olive'])
    plt.xlabel('Dataset')
    plt.ylabel('Female MSE')
    plt.title('Female MSE for Train, Validation, and Test')

    plt.tight_layout()
    plt.show()