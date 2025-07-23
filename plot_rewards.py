import csv
import os
import matplotlib.pyplot as plt

def extract_rewards_from_csv(csv_path):
    frames = []
    rewards = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            rewards.append(float(row["avg_reward"]))
    return frames, rewards

def plot_rewards_multiple(csv_paths: dict, save_path=None):
    plt.figure(figsize=(10, 6))
    for label, path in csv_paths.items():
        frames, rewards = extract_rewards_from_csv(path)
        plt.plot(frames, rewards, label=label)

    plt.xlabel("Frames")
    plt.ylabel("AvgReward (10 episodes)")
    plt.title("Average Reward vs Frames (Comparison by β)")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    csv_paths = {
        "β = 0.3": "results/LavaGapS7_rewards_beta0.30.csv",
        "β = 0.5": "results/LavaGapS7_rewards_beta0.50.csv",
        "β = 0.7": "results/LavaGapS7_rewards_beta0.70.csv",
        "only DQN": "results/LavaGapS7_rewards_beta0.00.csv"

    }

    plot_rewards_multiple(csv_paths, save_path="plots/LavaGap50k.png") #todo
