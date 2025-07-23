"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from agent import AbstractAgent
from buffers import ReplayBuffer
from networks import QNetwork
from minigrid_env import make_minigrid_env
from collections import defaultdict
import sys
import os
import csv



def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q‐Learning agent with ε‐greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        obs_dim: int,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        beta: float = 0.5
    ) -> None:
        """
        Initialize replay buffer, Q‐networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini‐batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target‐network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        self.obs_dim = obs_dim

        set_seed(env, seed)

        n_actions = env.action_space.n

        # main Q‐network and frozen target
        self.q = QNetwork(self.obs_dim, n_actions)
        self.target_q = QNetwork(self.obs_dim, n_actions)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

        self.visit_counts = defaultdict(int)  # Count visits to each state
        self.beta = beta
        self.eta = 1.0  # Scaling factor for the bonus

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -1.0 * self.total_steps / self.epsilon_decay
        )

    def hash_state(self, state: np.ndarray) -> int:
        """
        Hash a flattened state into a unique key for counting.
        """
        return hash(state.tobytes())

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε‐greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        # Handle dict observations from MiniGrid
        if isinstance(state, dict) and "image" in state:
            state = state["image"]

        state = state.flatten()

        if evaluate:
            # purely greedy
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = self.q(t)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                action = self.env.action_space.sample()
            else:
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    qvals = self.q(t)
                action = int(torch.argmax(qvals, dim=1).item())

        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        # Flatten and sanitize states
        flat_states = []
        flat_next_states = []

        for s in states:
            if isinstance(s, dict) and "image" in s:
                s = s["image"]
            flat_states.append(s.flatten())

        for s_next in next_states:
            if isinstance(s_next, dict) and "image" in s_next:
                s_next = s_next["image"]
            flat_next_states.append(s_next.flatten())

        # Convert to tensors
        s = torch.tensor(np.array(flat_states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(flat_next_states), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)

        # current Q estimates for taken actions
        pred = self.q(s).gather(1, a).squeeze(1)

        # compute TD target with frozen network
        with torch.no_grad():
            next_q = self.target_q(s_next).max(1)[0]
            target = r + self.gamma * next_q * (1 - mask)

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        self.training_log = []  # ← new list to store (frame, avg_reward)
        self.episode_logs = []

        state, _ = self.env.reset()

        # Fix dict observation format
        if isinstance(state, dict) and "image" in state:
            state = state["image"]

        state = state.flatten()

        ep_reward = 0.0
        steps_in_episode = 0
        recent_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            steps_in_episode += 1
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            if reward > 0:
                print(f" GOAL REACHED at step {frame} with reward {reward}")

            # Compute count-based bonus
            # Ensure state is flattened before hashing
            if isinstance(state, dict) and "image" in state:
                state = state["image"]
            state = state.flatten()

            state_key = self.hash_state(state)

            self.visit_counts[state_key] += 1
            bonus = 1.0 / (self.visit_counts[state_key] ** self.beta)
            shaped_reward = reward + 0.01 * bonus
            #shaped_reward = reward

            if isinstance(next_state, dict) and "image" in next_state:
                next_state = next_state["image"]

            next_state = next_state.flatten()

            # store and step
            self.buffer.add(state, action, shaped_reward, next_state, done or truncated, {})

            if frame % 1000 == 0:
                print(
                    f"[Step {frame}] Count = {self.visit_counts[state_key]}, Bonus = {bonus:.4f}, Total Reward = {shaped_reward:.2f}")

            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                # Log intrinsic bonus (mean bonus over this episode if you prefer)
                state_key = self.hash_state(state)
                visit_count = self.visit_counts[state_key]

                # Success if environment reward > 0
                success = 1 if ep_reward > 0 else 0

                self.episode_logs.append({
                    "episode": len(self.episode_logs),
                    "frame": frame,
                    "ep_reward": ep_reward,
                    "success": success,
                    "intrinsic_bonus": bonus,  # Last bonus seen in episode
                    "total_shaped_reward": shaped_reward,
                    "visit_count": visit_count,
                    "steps_in_episode": steps_in_episode
                })

                # Reset episode state
                state, _ = self.env.reset()
                ep_reward = 0.0
                steps_in_episode = 0  # ← reset for next episode

    print("Training complete.")

"""
            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging

                avg = np.mean(recent_rewards[-10:])
                print(
                    f"Frame {frame}, AvgReward(10): {avg:.2f}, epsilon={self.epsilon():.3f}"
                )
                self.training_log.append((frame, avg))
"""






@hydra.main(config_path="configs/agent", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    print("✅ Hydra config loaded!")  # DEBUG
    print(cfg)

    env_name = cfg.env.name.replace("MiniGrid-", "").replace("-v0", "").replace("-", "")
    beta_str = f"{cfg.agent.beta:.2f}".replace(".", "")
    log_filename = f"log_{env_name}_beta{beta_str}.txt"


    # 1) build env (with full observability)
    env = make_minigrid_env(cfg.env.name, seed=cfg.seed, full_obs=True)
    print(env)

    # 2) flatten one observation to determine obs_dim
    sample_obs, _ = env.reset()

    # Extract the 'image' field if observation is a dict
    if isinstance(sample_obs, dict) and "image" in sample_obs:
        sample_obs = sample_obs["image"]

    sample_obs = sample_obs.flatten()

    obs_dim = sample_obs.shape[0]

    # 3) map config → agent kwargs
    agent_kwargs = dict(
        obs_dim=obs_dim,  # ← ADD THIS
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
        beta=cfg.agent.beta
    )

    # 4) instantiate & train
    agent = DQNAgent(env, **agent_kwargs)
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)
    # Create directory
    os.makedirs("results", exist_ok=True)
    filename = f"{cfg.env.name}_beta{cfg.agent.beta:.2f}_seed{cfg.seed}.csv"
    csv_path = os.path.join("results", filename)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "frame", "ep_reward", "success",
            "intrinsic_bonus", "total_shaped_reward", "visit_count", "steps_in_episode"
        ])
        writer.writeheader()
        for row in agent.episode_logs:
            writer.writerow(row)

    print(f"✅ CSV saved to: {csv_path}")

    # Save full state visitation counts for later analysis
    import pickle

    visit_dir = "results/visit_counts"
    os.makedirs(visit_dir, exist_ok=True)

    visit_filename = f"{cfg.env.name}_beta{cfg.agent.beta:.2f}_seed{cfg.seed}_visitcounts.pkl"
    visit_path = os.path.join(visit_dir, visit_filename)

    with open(visit_path, "wb") as f:
        pickle.dump(agent.visit_counts, f)

    print(f"✅ Visit counts saved to: {visit_path}")


if __name__ == "__main__":
     main()