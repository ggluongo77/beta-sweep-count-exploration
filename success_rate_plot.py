import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from rliable import plot_utils as rplot
from rliable import library as rly  # Compatibile con versioni più vecchie

# ITALIANO #TODO
def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def extract_success_per_run(csv_path):
    """
    Estrae i valori binari di successo da un file CSV per un singolo run.
    """
    successes = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "success" not in row:
                raise KeyError(f"❌ Column 'success' not found in: {csv_path}\nHeaders: {list(row.keys())}")
            successes.append(float(row["success"]))  # 0.0 or 1.0
    return np.array(successes)

def load_all_successes(folder, env_name):
    """
    Carica tutti i CSV relativi a un ambiente specifico, raggruppati per valore di β.
    Ritorna: dict[label_beta, list of np.ndarray] con ogni lista contenente i run per quel β.
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
            run_data = extract_success_per_run(full_path)

            if label not in methods:
                methods[label] = []
            methods[label].append(run_data)

        except Exception as e:
            print(f"⚠️ Skipping file {filename}: {e}")

    return methods

def plot_success_rate(success_dict, save_path=None, max_episodes=None):
    # Tronca tutti i run alla stessa lunghezza
    min_len = min(min(len(run) for run in runs) for runs in success_dict.values())
    if max_episodes:
        min_len = min(min_len, max_episodes)

    for key in success_dict:
        success_dict[key] = [r[:min_len] for r in success_dict[key]]

    for label, runs in success_dict.items():
        runs_array = np.stack(runs)  # shape: [seeds x episodes]
        mean = np.mean(runs_array, axis=0)
        stderr = np.std(runs_array, axis=0, ddof=1) / np.sqrt(len(runs_array))
        ci95 = 1.96 * stderr  # 95% CI

        if len(mean) < 10:
            print(f"⚠️ Skipping {label}: too few episodes to smooth.")
            continue

        smoothed_mean = moving_average(mean)
        smoothed_ci95 = moving_average(ci95)
        episodes_smooth = np.arange(len(smoothed_mean))

        lower_bound = np.clip(smoothed_mean - smoothed_ci95, 0, 1)
        upper_bound = np.clip(smoothed_mean + smoothed_ci95, 0, 1)

        plt.plot(episodes_smooth, smoothed_mean, label=label)
        plt.fill_between(episodes_smooth, lower_bound, upper_bound, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Success Rate (Smoothed)")
    env_title = ENV.replace("MiniGrid-", "").replace("-v0", "")
    plt.title(f"{env_title} – Success Rate (95% CI)")

    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()

# === MAIN ===
if __name__ == "__main__":
    ENV = "MiniGrid-FourRooms-v0"
    results_folder = "results"
    save_path = f"plots/{ENV}_success_rate.png"

    data = load_all_successes(results_folder, ENV)
    plot_success_rate(data, save_path=save_path)
