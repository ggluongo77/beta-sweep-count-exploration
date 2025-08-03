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

def pairwise_mannwhitney(groups, metric_name):
    print(f"\n📊 Mann–Whitney U-Test on {metric_name.upper()} (pairwise β comparison)\n" + "-" * 60)
    labels = sorted(groups.keys())
    for a, b in itertools.combinations(labels, 2):
        data_a = flatten(groups[a])
        data_b = flatten(groups[b])
        stat, p = mannwhitneyu(data_a, data_b, alternative='two-sided')
        print(f"{a} vs {b} → p = {p:.4f} | {'✅ Significant' if p < 0.05 else '✴️ Not significant'}")

if __name__ == "__main__":
    ENV = "MiniGrid-MultiGoal-v0"
    results_folder = "../results"

    # Test on episode return
    rewards = load_groups_by_beta(results_folder, ENV, key="ep_reward")
    pairwise_mannwhitney(rewards, metric_name="ep_reward")

    # Optionally: test on success rate
    success = load_groups_by_beta(results_folder, ENV, key="success")
    pairwise_mannwhitney(success, metric_name="success")
