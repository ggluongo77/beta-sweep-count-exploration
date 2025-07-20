import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper  # importa il wrapper

def make_minigrid_env(env_name: str, seed: int = 0, render_mode: str = None, full_obs: bool = False):
    """
    Create and return a configured MiniGrid environment.

    Args:
        env_name (str): e.g. 'MiniGrid-ObstructedMaze-2Dlh-v0'.
        seed (int): random seed.
        render_mode (str): 'human' or 'rgb_array'.
        full_obs (bool): se True, passa al wrapper FullyObsWrapper per full observability.

    Returns:
        env: seeded, optionally wrapped environment.
    """
    # 1) instantiate, con render_mode se richiesto
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)

    # 2) seed
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    env.action_space.seed(seed)

    # 3) optional: full observability
    if full_obs:
        env = FullyObsWrapper(env)

    return env
