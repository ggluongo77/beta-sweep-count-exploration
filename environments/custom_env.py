from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register


class MultiGoalEnv(MiniGridEnv):
    """
     Custom MiniGrid environment with two goal objects: a nearby low-reward goal and a distant high-reward goal.

     The agent starts in the top-left corner of an empty 8x8 grid.
     One goal (reward = 0.1) is placed close to the agent, and another (reward = 1.0) is placed far away.
     This setup enables analysis of short-term versus long-term reward strategies and is useful
     for evaluating the impact of exploration bonuses.

     Parameters
     ----------
     size : int
        Grid size (default: 8).
     render_mode : Optional[str]
         Rendering mode used by Gym (e.g., "human").
     """
    def __init__(self, size=8, render_mode=None):
        self.size = size
        mission_space = MissionSpace(mission_func=lambda: "get the best reward")
        super().__init__(
            grid_size=size,
            max_steps=4 * size,
            mission_space=mission_space,
            render_mode = render_mode
        )

    def _gen_grid(self, width, height):
        """
            Generates the grid layout for the environment.

            - Surrounds the grid with walls.
            - Places the agent in the top-left corner.
            - Adds two goals:
                - A close goal at (3, 1) with low reward (0.1).
                - A far goal at the bottom-right corner with high reward (1.0).
            """
        self.grid = Grid(width, height)

        # Surround with walls
        self.grid.wall_rect(0, 0, width, height)

        # Place agent in top-left area
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # Facing right

        # Place close goal (low reward)
        close_goal = Goal()
        close_goal.reward = 0.1  # default reward, for clarity
        self.grid.set(3, 1, close_goal)
        self.close_goal_pos = (3, 1)

        # Place far goal (high reward)
        far_goal = Goal()
        far_goal.reward = 1.0
        self.grid.set(width - 2, height - 2, far_goal)
        self.far_goal_pos = (width - 2, height - 2)

        self.mission = "get the best reward"

    def step(self, action):
        """
           Executes one environment step given an action.

           Overrides the default reward function:
           - If the agent reaches a goal, the reward depends on that specific goal's reward value.
           - The episode terminates upon reaching either goal.
           - Otherwise, the reward is 0 and the episode continues.

           Parameters
           ----------
           action : int
               The action taken by the agent.

           Returns
           -------
           obs : Any
               Observation after the step.
           reward : float
               Reward received for the action.
           terminated : bool
               Whether the episode has ended due to reaching a goal.
           truncated : bool
               Whether the episode was truncated due to step limits.
           info : dict
               Additional debugging or environment-specific info.
           """
        obs, reward, terminated, truncated, info = super().step(action)

        # Check current cell and use its reward
        obj = self.grid.get(*self.agent_pos)
        if isinstance(obj, Goal):
            reward = obj.reward
            terminated = True
        else:
            reward = 0

        return obs, reward, terminated, truncated, info


# Register the environment with Gym
register(
    id='MiniGrid-MultiGoal-v0',
    entry_point='environments.custom_env:MultiGoalEnv',
)

