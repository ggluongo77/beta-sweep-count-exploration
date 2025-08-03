import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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

            if label not in methods:
                methods[label] = []
            methods[label].append(first_success)

        except Exception as e:
            print(f"⚠️ Skipping file {filename}: {e}")

    return methods

def plot_episodes_to_solve(data, save_path=None, max_episode_display=350):
    """
    Plots the average episodes-to-solve per β, with 95% CI.
    Bars are gray if no runs solved the task.
    """
    labels = []
    means = []
    errors = []
    colors = []

    for label, values in data.items():
        successes = [v for v in values if np.isfinite(v)]

        if len(successes) == 0:
            print(f"{label} never solved the task.")
            labels.append(f"{label} (unsolved)")
            means.append(max_episode_display)
            errors.append(0)
            colors.append("gray")
        else:
            arr = np.array(successes)
            mean = np.mean(arr)
            stderr = np.std(arr, ddof=1) / np.sqrt(len(arr))
            ci95 = 1.96 * stderr  # Approximate 95% CI
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

    # Plotting
    plt.figure(figsize=(8, 6))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errors, capsize=5, color=colors)
    plt.xticks(x, labels)
    plt.ylabel("Episodes to First Success")
    env_title = ENV.replace("MiniGrid-", "").replace("-v0", "")
    plt.title(f"{env_title} – Sample Efficiency (95% CI)")
    plt.grid(axis="y")

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    ENV = "MiniGrid-MultiGoal-v0"
    results_folder = f"../results/{ENV}"
    save_path = f"../plots/{ENV}/{ENV}_episodes_to_solve.png"

    data = load_all_first_success(results_folder, ENV)
    plot_episodes_to_solve(data, save_path=save_path)
