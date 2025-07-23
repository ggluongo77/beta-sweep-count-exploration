import pickle
import matplotlib.pyplot as plt

# 🔁 Change this to the correct file path you saved
file_path = "results/visit_counts/MiniGrid-LavaGapS7-v0_beta0.70_seed0_visitcounts.pkl"

with open(file_path, "rb") as f:
    visit_counts = pickle.load(f)
counts = list(visit_counts.values())

plt.figure(figsize=(8, 5))
plt.hist(counts, bins=30, edgecolor='black')
plt.xlabel("Visit count per state")
plt.ylabel("Number of states")
plt.title("State Visitation Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()
