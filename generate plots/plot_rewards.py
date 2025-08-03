import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=50):
    """
    Compute the average over a 1D array.

    Parameters:
    - x: input array
    - window: size of the moving window

    Returns:
    - Smoothed array using moving average
    """
    return np.convolve(x, np.ones(window) / window, mode='valid')


def extract_returns_per_run(csv_path, reward_key="ep_reward"):
    """
    Extracts the reward values from a CSV file for a single training run.

    Parameters:
    - csv_path: path to the CSV file
    - reward_key: column name for the reward metric (default = 'ep_reward')

    Returns:
    - A NumPy array of episode returns
    """
    returns = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if reward_key not in row:
                raise KeyError(f"Column '{reward_key}' not found in: {csv_path}\nHeaders: {list(row.keys())}")
            returns.append(float(row[reward_key]))
    return np.array(returns)


def load_all_runs(folder, env_name):
    """
    Loads all CSV files for a given environment, grouped by β value.

    Parameters:
    - folder: root directory containing the CSV files
    - env_name: name of the environment to filter files

    Returns:
    - Dictionary mapping beta labels to lists of reward arrays (one per run)
      Format: { 'β = 0.3': [run1, run2, ...], 'DQN': [...], ... }
    """
    methods = {}
    for filename in os.listdir(folder):
        if not filename.endswith(".csv") or env_name not in filename:
            continue

        try:
            beta_str = filename.split("beta")[1].split("_")[0]
            beta = float(beta_str)
            label = f"β = {beta:.1f}" if beta > 0 else "DQN"

            full_path = os.path.join(folder, filename)
            run_data = extract_returns_per_run(full_path, reward_key="ep_reward")

            if label not in methods:
                methods[label] = []
            methods[label].append(run_data)

        except Exception as e:
            print(f"Skipping file {filename}: {e}")

    return methods


def plot_with_rliable(returns_dict, save_path=None, max_episodes=None):
    """
    Plots learning curves with 95% confidence intervals.

    Parameters:
    - returns_dict: dictionary {label -> list of runs}, where each run is an array of returns
    - save_path: path to save the plot (optional)
    - max_episodes: truncate to this many episodes (optional)
    """
    # Truncate all runs to the same length
    min_len = min(min(len(run) for run in runs) for runs in returns_dict.values())
    if max_episodes:
        min_len = min(min_len, max_episodes)

    for key in returns_dict:
        returns_dict[key] = [r[:min_len] for r in returns_dict[key]]

    # Compute mean and 95% CI for each method
    for label, runs in returns_dict.items():
        runs_array = np.stack(runs)  # shape: [seeds x episodes]
        mean = np.mean(runs_array, axis=0)
        stderr = np.std(runs_array, axis=0, ddof=1) / np.sqrt(len(runs_array))
        ci95 = 1.96 * stderr  # Approximate 95% CI

        if len(mean) < 10:
            print(f"Skipping {label}: too few episodes to smooth.")
            continue

        # Smooth the curve
        episodes = np.arange(len(mean))
        smoothed_mean = moving_average(mean)
        smoothed_ci95 = moving_average(ci95)
        episodes_smooth = np.arange(len(smoothed_mean))

        # Plot the curve
        plt.plot(episodes_smooth, smoothed_mean, label=label)
        lower_bound = np.clip(smoothed_mean - smoothed_ci95, 0, 1)
        upper_bound = np.clip(smoothed_mean + smoothed_ci95, 0, 1)
        plt.fill_between(episodes_smooth, lower_bound, upper_bound, alpha=0.2)

    # Final plot formatting
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    env_title = ENV.replace("MiniGrid-", "").replace("-v0", "")
    plt.title(f"{env_title} – Learning Curves (95% CI)")
    plt.grid(True)
    plt.legend()

    # Save or display
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    ENV = "MiniGrid-MultiGoal-v0"
    results_folder = f"../results/{ENV}"
    save_path = f"../plots/{ENV}/{ENV}_rliable_curve.png"

    data = load_all_runs(results_folder, ENV)
    plot_with_rliable(data, save_path=save_path)
