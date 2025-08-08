import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def extract_intrinsic_per_run(csv_path):
    bonuses = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "intrinsic_bonus" not in row:
                raise KeyError(f"Column 'intrinsic_bonus' not found in: {csv_path}")
            bonuses.append(float(row["intrinsic_bonus"]))
    return np.array(bonuses)

def load_all_intrinsic(folder, env_name):
    methods = {}
    for filename in os.listdir(folder):
        if not filename.endswith(".csv"):
            continue
        if env_name not in filename:
            continue

        try:
            beta_str = filename.split("beta")[1].split("_")[0]
            beta = float(beta_str)

            if beta == 0.0:
                continue

            label = f"β = {beta:.1f}"

            full_path = os.path.join(folder, filename)
            run_data = extract_intrinsic_per_run(full_path)

            if label not in methods:
                methods[label] = []
            methods[label].append(run_data)

        except Exception as e:
            print(f"Skipping file {filename}: {e}")
    return methods

def plot_intrinsic_dynamics(bonus_dict, save_path=None, max_episodes=None):
    min_len = min(min(len(run) for run in runs) for runs in bonus_dict.values())
    if max_episodes:
        min_len = min(min_len, max_episodes)

    for key in bonus_dict:
        bonus_dict[key] = [r[:min_len] for r in bonus_dict[key]]

    for label, runs in bonus_dict.items():
        runs_array = np.stack(runs)
        mean = np.mean(runs_array, axis=0)
        stderr = np.std(runs_array, axis=0, ddof=1) / np.sqrt(len(runs_array))
        ci95 = 1.96 * stderr

        smoothed_mean = moving_average(mean)
        smoothed_ci = moving_average(ci95)
        episodes = np.arange(len(smoothed_mean))

        lower = np.clip(smoothed_mean - smoothed_ci, 0, None)
        upper = smoothed_mean + smoothed_ci

        plt.plot(episodes, smoothed_mean, label=label)
        plt.fill_between(episodes, lower, upper, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Intrinsic Bonus (Smoothed)")
    env_title = ENV.replace("MiniGrid-", "").replace("-v0", "")
    plt.title(f"{env_title} – Intrinsic Reward Dynamics (β > 0 only)")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    ENV = "MiniGrid-FourRooms-v0"
    results_folder = f"../results/{ENV}"
    save_path = f"../plots/{ENV}/{ENV}_intrinsic_dynamics.png"

    data = load_all_intrinsic(results_folder, ENV)
    plot_intrinsic_dynamics(data, save_path=save_path)
