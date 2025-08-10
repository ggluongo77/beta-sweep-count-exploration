import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_visit_counts(env_name, beta_value, seeds, folder="../results/visit_counts"):
    combined_counts = {}
    for seed in seeds:
        filename = f"{env_name}_beta{beta_value:.2f}_seed{seed}_spatial.pkl"

        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        with open(file_path, "rb") as f:
            visit_counts = pickle.load(f)
        for pos, count in visit_counts.items():
            if pos not in combined_counts:
                combined_counts[pos] = 0
            combined_counts[pos] += count
    return combined_counts

def plot_heatmap(counts, grid_size, title="", save_path=None):
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    for key, count in counts.items():
        if isinstance(key, tuple) and len(key) == 2:
            x, y = key
            if 0 <= x < grid_size and 0 <= y < grid_size:
                heatmap[y, x] = count
        else:
            print(f"Invalid key format: {key} (expected (x, y))")

    plt.figure(figsize=(6, 6))
    im = plt.imshow(heatmap, cmap='viridis', origin='lower')
    plt.colorbar(im, label="Visit count")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(np.arange(grid_size))
    plt.yticks(np.arange(grid_size))
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    env = "MiniGrid-MultiGoal-v0"
    beta = 0.7
    seeds = [0,1,2,3,4]
    grid_size = 8  # adjust if needed #fourrooms: 19 lavagap: 7

    counts = load_visit_counts(env, beta, seeds)
    title = f"State Visitation Heatmap\n{env} – β = {beta}"
    plot_heatmap(counts, grid_size, title=title, save_path=f"../plots/{env}/{env}_beta{beta:.2f}_heatmap.png")

