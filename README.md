# Evaluating the Decay Exponent β in Count-Based Exploration Bonuses

The work investigates how the decay exponent **β** in count-based intrinsic rewards affects the performance of a DQN agent across multiple MiniGrid environments with varying exploration demands.

Experiments compare **β ∈ {0.3, 0.5, 0.7}** across:

* **Empty-Random-6x6** (simple open space)
* **FourRooms** (structured navigation)
* **LavaGapS7** (risk-sensitive navigation)
* **MultiGoal** (custom environment)



## Requirements & Installation

This project uses Python **3.11**.

### Setup steps:

```bash
# Install uv package manager
pip install uv

# Create a new Python 3.11 virtual environment
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate  

# Install project dependencies
make install
```



## Running Training

Training is launched via the main entry point:

```bash
python dqn.py
```

Parameters and hyperparameters are configured using Hydra in `configs/agent/dqn.yaml`.



## Plotting and Analysis

After training, use the scripts in `plot_scripts/` to generate analysis figures.
These scripts are standalone and can be run from the project root.

**Examples:**

```bash
python plot_scripts/aggregate_plots.py
python plot_scripts/plot_rewards.py
python plot_scripts/plot_success_rate.py
python plot_scripts/mann_whitney_test.py
```

Each script reads the CSV logs from `results/` and produces figures.



## Repository Structure

```
beta-sweep-count-exploration/
│
├── dqn.py                  # Main training script
├── buffers.py              # Replay buffer implementation
├── networks.py             # Q-network architecture
│
├── agent/
│   ├── abstract_agent.py   # Base agent interface
│   └── buffer.py           # Alternate buffer implementation
│
├── environments/
│   ├── custom_env.py       # Custom MultiGoal environment
│   └── minigrid_env.py     # MiniGrid environment wrapper
│
├── configs/
│   └── agent/dqn.yaml      # Hydra configuration file
│
├── plot_scripts/           # Standalone analysis scripts
│   ├── aggregate_plots.py
│   ├── plot_rewards.py
│   ├── plot_success_rate.py
│   ├── plot_episodes_to_solve.py
│   ├── plot_intrinsic_reward.py
│   ├── heatmaps.py
│   └── mann_whitney_test.py
│
└── plots/                  # Generated figures
```

---

## Notes

All experiments are fully reproducible by setting the seed in `configs/agent/dqn.yaml`.
