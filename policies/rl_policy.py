from stable_baselines3 import PPO


class RLMemoryPolicy:
    """
    PPO-based RL policy for memory management.
    """

    def __init__(self, env, total_timesteps=10000):
        self.env = env
        self.total_timesteps = total_timesteps
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            gamma=0.99,
        )

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps)

    def select_action(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
