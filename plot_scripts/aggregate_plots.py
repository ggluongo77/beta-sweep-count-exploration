import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from rliable import metrics
from rliable import library as rly
from rliable import plot_utils as rly_plot

"""
Generates aggregate performance plots for multiple β values across different MiniGrid environments
using the rliable library.
"""

results_root = "../results"
output_dir = "../plots"
betas = [0.3, 0.5, 0.7]
seeds = [0, 1, 2, 3, 4]
last_k = 100  # Episodes to average over
target_metrics = [
    "success",
    "ep_reward",
    "steps_in_episode"
]


def extract_metric(csv_path, metric_name, last_k):
    values = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric_name not in row:
                raise ValueError(f"Metric '{metric_name}' not found in {csv_path}")
            values.append(float(row[metric_name]))
    if len(values) < last_k:
        raise ValueError(f"Not enough episodes in {csv_path}")
    return np.mean(values[-last_k:])


os.makedirs(output_dir, exist_ok=True)

# Loop over all target metrics
for target_metric in target_metrics:
    print(f"\nProcessing: {target_metric}")

    beta_scores = {f"β = {b:.1f}": [] for b in betas}

    for env_folder in sorted(os.listdir(results_root)):
        env_path = os.path.join(results_root, env_folder)
        if not os.path.isdir(env_path):
            continue

        for beta in betas:
            label = f"β = {beta:.1f}"
            scores = []
            for seed in seeds:
                filename = f"{env_folder}_beta{beta:.2f}_seed{seed}.csv"
                full_path = os.path.join(env_path, filename)
                if os.path.exists(full_path):
                    try:
                        score = extract_metric(full_path, target_metric, last_k)
                        scores.append(score)
                    except Exception as e:
                        print(f" Skipping {filename}: {e}")
                else:
                    print(f"Missing file: {full_path}")
            if scores:
                beta_scores[label].append(scores)

    # Convert into rliable format: dict[label] -> np.array(num_envs x num_seeds)
    score_matrix = {label: np.array(env_runs) for label, env_runs in beta_scores.items()}

    if not any(len(v) > 0 for v in score_matrix.values()):
        print(f"Skipping metric '{target_metric}' due to missing data.")
        continue

    # Define aggregation
    agg_func = lambda x: np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x)
    ])

    # Run bootstrapped confidence intervals
    agg_scores, agg_cis = rly.get_interval_estimates(score_matrix, agg_func, reps=50000)

    # Plot
    fig, axes = rly_plot.plot_interval_estimates(
        agg_scores,
        agg_cis,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=list(score_matrix.keys()),
        xlabel=target_metric.replace("_", " ").title() + f" (Last {last_k} Episodes)"
    )

    fig.subplots_adjust(bottom=0.4)
    # Save
    plot_path = os.path.join(output_dir, f"{target_metric}_summary.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")
