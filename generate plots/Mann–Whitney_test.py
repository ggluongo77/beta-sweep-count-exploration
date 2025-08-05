import os
import csv
import itertools
import numpy as np
from scipy.stats import mannwhitneyu

def extract_metric(csv_path, key="ep_reward"):
    values = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if key not in row:
                raise KeyError(f"Column '{key}' not found in: {csv_path}")
            values.append(float(row[key]))
    return np.array(values)

def load_groups_by_beta(folder, env_name, key="ep_reward"):
    groups = {}
    for filename in os.listdir(folder):
        if not filename.endswith(".csv"):
            continue
        if env_name not in filename:
            continue
        try:
            beta_str = filename.split("beta")[1].split("_")[0]
            beta = float(beta_str)
            label = f"β = {beta:.1f}"
            full_path = os.path.join(folder, filename)
            values = extract_metric(full_path, key=key)

            if label not in groups:
                groups[label] = []
            groups[label].append(values)
        except Exception as e:
            print(f"Skipping file {filename}: {e}")
    return groups

def flatten(group):
    return np.concatenate(group)

def pairwise_mannwhitney(groups, metric_name, output_path=None):
    output_lines = []
    header = f"\nMann–Whitney U-Test on {metric_name.upper()} (pairwise β comparison)\n" + "-" * 60
    print(header)
    output_lines.append(header)

    labels = sorted(groups.keys())
    for a, b in itertools.combinations(labels, 2):
        data_a = flatten(groups[a])
        data_b = flatten(groups[b])
        stat, p = mannwhitneyu(data_a, data_b, alternative='two-sided')
        result = f"{a} vs {b} → p = {p:.4f} | {'Significant' if p < 0.05 else 'Not significant'}"
        print(result)
        output_lines.append(result)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

if __name__ == "__main__":
    ENV = "MiniGrid-LavaGapS7-v0"
    results_folder = f"../results/{ENV}"

    # Run test on episode return
    rewards = load_groups_by_beta(results_folder, ENV, key="ep_reward")
    output_path = os.path.join("..", "plots", ENV, f"{ENV}_mannwhitney_ep_reward.txt")
    pairwise_mannwhitney(rewards, metric_name="ep_reward", output_path=output_path)

    # Run test on success rate
    success = load_groups_by_beta(results_folder, ENV, key="success")
    output_path = os.path.join("..", "plots", ENV, f"{ENV}_mannwhitney_success.txt")
    pairwise_mannwhitney(success, metric_name="success", output_path=output_path)

