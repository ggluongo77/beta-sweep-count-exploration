import time
from collections import defaultdict
import gym
import numpy as np
from minigrid_env import make_minigrid_env  # ✅ import your helper!
np.bool8 = np.bool_
# ------------------------------------
# 1️⃣ Configuration
# ------------------------------------
ENV_NAME = 'MiniGrid-DoorKey-5x5-v0'  # ✅ Always exists in new minigrid
SEED = 42
FULL_OBSERVABILITY = True
BETA = 0.5
NUM_EPISODES = 100

# ------------------------------------
# 2️⃣ Create the Environment
# ------------------------------------
env = make_minigrid_env(
    env_name=ENV_NAME,
    seed=SEED,
    full_obs=FULL_OBSERVABILITY
)

print(f"\nTiming test for environment: {env.spec.id}")
print(f"Full Observability: {FULL_OBSERVABILITY}")
print(f"Beta value: {BETA}")
print(f"Number of Episodes: {NUM_EPISODES}\n")

# ------------------------------------
# 3️⃣ Counting Setup
# ------------------------------------
counts = defaultdict(int)

def get_obs_array(obs):
    """
    Handles different observation formats:
    - tuple (obs, info)
    - dict with 'image' key
    - raw array
    """
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict) and 'image' in obs:
        return obs['image']
    return obs

def hash_observation(obs_array):
    return hash(obs_array.tobytes())

# ------------------------------------
# 4️⃣ Dummy Random Agent
# ------------------------------------
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

    def learn(self, *args):
        pass

agent = RandomAgent(env.action_space)

# ------------------------------------
# 5️⃣ Timing Loop
# ------------------------------------
start_time = time.time()

for episode in range(NUM_EPISODES):
    obs = env.reset()
    done = False

    while not done:
        # Robust observation handling
        obs_array = get_obs_array(obs)

        # Count-based intrinsic bonus
        state_key = hash_observation(obs_array)
        counts[state_key] += 1
        intrinsic_bonus = 1 / (counts[state_key] ** BETA)

        # Agent action
        action = agent.act(obs)
        obs, extrinsic_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Total reward = extrinsic + intrinsic
        total_reward = extrinsic_reward + intrinsic_bonus

        # Dummy learn step
        agent.learn(obs, action, total_reward, done)

end_time = time.time()

# ------------------------------------
# 6️⃣ Report
# ------------------------------------
elapsed_time = end_time - start_time
print(f"✅ Completed {NUM_EPISODES} episodes in {elapsed_time:.2f} seconds\n")
