import os
import csv
import numpy as np
import matplotlib.pyplot as plt

"""
Computes and plots episodes-to-first-success for each β in a MiniGrid environment,
including 95% confidence intervals, from CSV log files.
"""


def extract_first_success(csv_path):
    """
    Extracts the first success episode from a CSV log.
    Returns np.inf if the agent never solved the task.
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if "success" not in row:
                raise KeyError(f"Column 'success' not found in: {csv_path}")
            if float(row["success"]) == 1.0:
                return i
    return np.inf

def load_all_first_success(folder, env_name):
    """
    Loads first-success episodes for each β value.
    Returns: dict[label -> list of episodes to solve]
    """
    methods = {}
    for filename in os.listdir(folder):
        if not filename.endswith(".csv"):
            continue
        if env_name not in filename:
            continue

        try:
            beta_str = filename.split("beta")[1].split("_")[0]
            beta = float(beta_str)
            label = f"β = {beta:.1f}" if beta > 0 else "DQN"

            full_path = os.path.join(folder, filename)
            first_success = extract_first_success(full_path)

            methods.setdefault(label, []).append(first_success)

        except Exception as e:
            print(f"Skipping file {filename}: {e}")

    return methods

def plot_episodes_to_solve(data, env_name, save_path=None, max_episode_display=350):
    """
    Plots the average episodes-to-solve per β with 95% CI.
    """
    labels, means, errors, colors = [], [], [], []

    # Optional: stable ordering
    order = ["DQN", "β = 0.3", "β = 0.5", "β = 0.7"]
    items = sorted(data.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 999)

    for label, values in items:
        successes = [v for v in values if np.isfinite(v)]

        if len(successes) == 0:
            labels.append(f"{label} (unsolved)")
            means.append(float(max_episode_display))
            errors.append(0.0)
            colors.append("gray")
        else:
            arr = np.array(successes, dtype=float)
            mean = max(0.0, float(np.mean(arr)))
            stderr = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            ci95 = max(0.0, 1.96 * stderr)

            labels.append(label)
            means.append(mean)
            errors.append(ci95)

            if "DQN" in label:
                colors.append("blue")
            elif "0.3" in label:
                colors.append("orange")
            elif "0.5" in label:
                colors.append("green")
            elif "0.7" in label:
                colors.append("red")
            else:
                colors.append("gray")


    plt.figure(figsize=(8, 6))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errors, capsize=5, color=colors)
    plt.xticks(x, labels)
    plt.ylabel("Episodes to First Success")

    env_title = env_name.replace("MiniGrid-", "").replace("-v0", "")
    plt.title(f"{env_title} – Sample Efficiency (95% CI)")
    plt.grid(axis="y")


    top = max([m + e for m, e in zip(means, errors)] + [max_episode_display])
    plt.ylim(0, top * 1.05)  # bottom=0 ensures no negative ticks

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    ENV = "MiniGrid-LavaGapS7-v0"
    results_folder = f"../results/{ENV}"
    save_path = f"../plots/{ENV}/{ENV}_episodes_to_solve.png"

    data = load_all_first_success(results_folder, ENV)
    plot_episodes_to_solve(data, env_name=ENV, save_path=save_path)
