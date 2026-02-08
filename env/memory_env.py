import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MemoryManagementEnv(gym.Env):
    """
    A toy environment for studying memory management decisions.
    """

    def __init__(self, max_memory_size=5, episode_length=50):
        super().__init__()

        self.max_memory_size = max_memory_size
        self.episode_length = episode_length

        # Observation: [item_importance, memory_size, avg_memory_importance]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Actions: 0 = ignore, 1 = store, 2 = delete oldest
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.memory = []
        self.current_step = 0

        self.current_item_importance = np.random.rand()

        return self._get_obs(), {}

    def _get_obs(self):
        memory_size = len(self.memory)
        avg_importance = np.mean(self.memory) if self.memory else 0.0

        return np.array(
            [self.current_item_importance, memory_size / self.max_memory_size, avg_importance],
            dtype=np.float32
        )

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # Action logic
        if action == 1:  # STORE
            if len(self.memory) < self.max_memory_size:
                self.memory.append(self.current_item_importance)
                reward += self.current_item_importance
            else:
                reward -= 0.5  # overflow penalty

        elif action == 2:  # DELETE OLDEST
            if self.memory:
                removed = self.memory.pop(0)
                reward -= removed * 0.2  # penalty if removing useful info

        # IGNORE has no direct effect

        # Generate next item
        self.current_item_importance = np.random.rand()
        self.current_step += 1

        # Compact memory bonus
        reward += 0.1 * (1 - len(self.memory) / self.max_memory_size)

        if self.current_step >= self.episode_length:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}
