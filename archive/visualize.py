import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ----- Helper Functions ----- #

def load_agent_metrics(metrics_path):
    """
    Loads the training metrics from the specified JSON file.

    :param metrics_path: Path to the JSON file containing the metrics
    :return: Dictionary containing the metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics

# ----- Individual Plot Functions ----- #

def plot_rewards(metrics_path, save_path):
    """
    Plots rewards across timesteps from the saved training metrics JSON file.

    :param metrics_path: Path to the saved metrics file
    :param save_path: Path to where the plot will be saved
    """
    # Load the agent metrics from the saved file
    metrics = load_agent_metrics(metrics_path)

    rewards = metrics['rewards']

    # Plot rewards
    plt.figure()
    plt.plot(rewards, label='Rewards', color='tab:blue')
    plt.xlabel('Timestep')
    plt.ylabel('Rewards')
    plt.title('Training Rewards Over Timesteps')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'timestep_rewards.png'))
    plt.show()


def plot_episode_rewards(metrics_path, save_path):
    """
    Plots rewards across episodes from the saved training metrics JSON file.

    :param metrics_path: Path to the saved metrics file
    :param save_path: Path to where the plot will be saved
    """
    # Load the agent metrics from the saved file
    metrics = load_agent_metrics(metrics_path)

    rewards = metrics['rewards']

    episode_rewards = []
    for i in range(0, 100):
        episode_rewards.append(sum(rewards[i * 1000:(i + 1) * 1000]) / 1000)

    # Plot rewards
    plt.figure()
    plt.plot(episode_rewards, label='Rewards', color='tab:blue')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Training Rewards Over Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'episode_rewards.png'))
    plt.show()


def plot_cumulative_rewards(metrics_path, save_path):
    """
    Plots cumulative rewards across episodes from the saved training metrics JSON file.

    :param metrics_path: Path to the saved metrics file
    :param save_path: Path to where the plot will be saved
    """
    # Load the agent metrics from the saved file
    metrics = load_agent_metrics(metrics_path)

    rewards = metrics['rewards']

    episode_rewards = []
    for i in range(0, 100):
        episode_rewards.append(np.mean(rewards[i * 1000:(i + 1) * 1000]))

    cum_avg_rewards = np.cumsum(episode_rewards) / np.arange(1, 101)

    # Plot rewards
    plt.figure()
    plt.plot(cum_avg_rewards, label='Rewards', color='tab:blue')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Cumulative Average Training Rewards Over Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cumulative_rewards.png'))
    plt.show()


def plot_mse_hist(baseline_metrics_path, training_metrics_path, save_path):
    """
    Plots the trained agent's MSE values against the baseline MSE values in a histogram.

    :param baseline_results: Dictionary containing the baseline results
    :param training_metrics_path: Path to the saved training metrics file
    :param save_path: Path to where the plot will be saved
    """

    # Extract agent results
    training_metrics = load_agent_metrics(training_metrics_path)
    train_mse = training_metrics['train_mse'][-1]
    val_mse = training_metrics['val_mse'][-1]
    test_mse = training_metrics['test_mse'][-1]

    # Extract baseline results
    baseline_metrics = load_agent_metrics(baseline_metrics_path)
    b_train_mse = baseline_metrics['baseline']['train']['mse']
    b_val_mse = baseline_metrics['baseline']['val']['mse']
    b_test_mse = baseline_metrics['baseline']['test']['mse']

    highest_num = max(train_mse, val_mse, test_mse, b_train_mse, b_val_mse, b_test_mse)

    # make new dict with baseline and agent results
    categories = ['Train', 'Validation', 'Test']
    results = {
        'Baseline': (b_train_mse, b_val_mse, b_test_mse),
        'Agent': (train_mse, val_mse, test_mse),
    }
    x = np.arange(len(categories))
    multiplier = 0
    width = 0.40

    plt.figure()
    for model, value in results.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, value, width, label=model)
        plt.bar_label(rects, fmt="{:.2f}", padding=3)
        multiplier += 1

    plt.ylabel('MSE')
    plt.title('Baseline vs. Agent MSE')
    plt.xticks(x + width, categories)
    plt.ylim(0, highest_num + 25)
    plt.legend()

    # plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mse_hist.png'))
    plt.show()


def plot_female_mse_hist(baseline_metrics_path, training_metrics_path, save_path):
    """
    Plots the trained agent's Female MSE values against the baseline Female MSE values in a histogram.

    :param baseline_metrics_path: Path to the saved baseline metrics file
    :param training_metrics_path: Path to the saved training metrics file
    :param save_path: Path to where the plot will be saved
    """

    # Extract agent results
    training_metrics = load_agent_metrics(training_metrics_path)
    train_female_mse = training_metrics['train_female_mse'][-1]
    val_female_mse = training_metrics['val_female_mse'][-1]
    test_female_mse = training_metrics['test_female_mse'][-1]

    # Extract baseline results
    baseline_metrics = load_agent_metrics(baseline_metrics_path)
    b_train_female_mse = baseline_metrics['baseline']['train']['female_mse']
    b_val_female_mse = baseline_metrics['baseline']['val']['female_mse']
    b_test_female_mse = baseline_metrics['baseline']['test']['female_mse']

    highest_num = max(train_female_mse, val_female_mse, test_female_mse, b_train_female_mse, b_val_female_mse, b_test_female_mse)

    # make new dict with baseline and agent results
    categories = ['Train', 'Validation', 'Test']
    results = {
        'Baseline': (b_train_female_mse, b_val_female_mse, b_test_female_mse),
        'Agent': (train_female_mse, val_female_mse, test_female_mse),
    }
    x = np.arange(len(categories))
    multiplier = 0
    width = 0.40

    plt.figure()
    for model, value in results.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, value, width, label=model)
        plt.bar_label(rects, fmt="{:.2f}", padding=3)
        multiplier += 1

    plt.ylabel('MSE')
    plt.title('Baseline vs. Agent Female MSE')
    plt.xticks(x + width, categories)
    plt.ylim(0, highest_num + 25)
    plt.legend()

    # plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'female_mse_hist.png'))
    plt.show()

# ----- Combined Plot Functions ----- #

def plot_episode_rewards_combined(metrics_path_list, legend_list, save_path):
    """
    Plots episode rewards for multiple runs from the saved training metrics JSON files for each agent.

    :param metrics_path_list: List of paths to the saved metrics files for each agent
    :param legend_list: List of legend labels for each agent
    :param save_path: Path to where the plot will be saved
    """
    plt.figure()

    for i in range(len(metrics_path_list)):
        metrics = load_agent_metrics(metrics_path_list[i])
        rewards = metrics['rewards']
        episode_rewards = []
        for episode in range(0, 100):
            episode_rewards.append(sum(rewards[episode * 1000:(episode + 1) * 1000]) / 1000)
        plt.plot(episode_rewards, label=legend_list[i])

    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Training Rewards Over Episodes')
    # plt.legend(loc='upper right', ncol=3)
    plt.legend()

    # plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison_episode_rewards.png'))
    plt.show()


def plot_cumulative_rewards_combined(metrics_path_list, legend_list, save_path):
    """
    Plots cumulative rewards across episodes for multiple runs from the saved training metrics JSON files for each agent.

    :param metrics_path_list: List of paths to the saved metrics files for each agent
    :param legend_list: List of legend labels for each agent
    :param save_path: Path to where the plot will be saved
    """
    plt.figure()

    for i in range(len(metrics_path_list)):
        metrics = load_agent_metrics(metrics_path_list[i])
        rewards = metrics['rewards']
        episode_rewards = []
        for episode in range(0, 100):
            episode_rewards.append(np.mean(rewards[episode * 1000:(episode + 1) * 1000]))
        cum_avg_rewards = np.cumsum(episode_rewards) / np.arange(1, 101)
        plt.plot(cum_avg_rewards, label=legend_list[i])

    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Cumulative Average Training Rewards Over Episodes')
    # plt.legend(loc='upper right', ncol=3)
    plt.legend()

    # plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison_cumulative_rewards.png'))
    plt.show()

