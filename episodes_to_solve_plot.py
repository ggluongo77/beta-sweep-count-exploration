import os
import csv
import numpy as np
import matplotlib.pyplot as plt
# ITALIANO #TODO
def extract_first_success(csv_path):
    """
    Estrae l'indice (episodio) del primo successo in un singolo run.
    Ritorna `np.inf` se non ha mai avuto successo.
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if "success" not in row:
                raise KeyError(f"❌ Column 'success' not found in: {csv_path}")
            if float(row["success"]) == 1.0:
                return i  # episodio trovato
    return np.inf  # mai risolto

def load_all_first_success(folder, env_name):
    """
    Ritorna: dict[label_beta, list of episode indices of first success]
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

def plot_episodes_to_solve(data, save_path=None):
    labels = list(data.keys())
    means = [np.mean(v) for v in data.values()]
    stds = [np.std(v) for v in data.values()]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=5, color=["blue", "orange", "green", "red"])
    plt.ylabel("Episodes to First Success")
    env_title = ENV.replace("MiniGrid-", "").replace("-v0", "")
    plt.title(f"{env_title} Sample Efficiency – Episodes to Solve")
    plt.grid(axis="y")

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()

# === MAIN ===
if __name__ == "__main__":
    ENV = "MiniGrid-FourRooms-v0"
    results_folder = "results"
    save_path = f"plots/{ENV}_episodes_to_solve.png"

    data = load_all_first_success(results_folder, ENV)
    plot_episodes_to_solve(data, save_path=save_path)
