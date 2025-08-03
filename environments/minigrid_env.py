import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

def make_minigrid_env(env_name: str, seed: int = 0, render_mode: str = None, full_obs: bool = False):
    """
    Create and return a configured MiniGrid environment.

    Args:
        env_name (str): e.g. 'MiniGrid-ObstructedMaze-2Dlh-v0'.
        seed (int): random seed.
        render_mode (str): 'human' or None
        full_obs (bool): if True it's fully observable

    Returns:
        env: seeded, optionally wrapped environment.
    """
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)

    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    env.action_space.seed(seed)

    if full_obs:
        env = FullyObsWrapper(env)

    return env
