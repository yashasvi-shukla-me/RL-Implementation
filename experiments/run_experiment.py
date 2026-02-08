import numpy as np

from RL_Demo.env.memory_env import MemoryManagementEnv
from RL_Demo.policies.rule_based import RuleBasedMemoryPolicy
from RL_Demo.policies.rl_policy import RLMemoryPolicy


def run_rule_based(env, episodes=50):
    policy = RuleBasedMemoryPolicy(max_memory_size=env.max_memory_size)
    rewards = []
    memory_sizes = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = policy.select_action(obs, env.memory)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        memory_sizes.append(len(env.memory))

    return np.mean(rewards), np.mean(memory_sizes)


def run_rl(env, episodes=50):
    policy = RLMemoryPolicy(env)
    policy.train()

    rewards = []
    memory_sizes = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = policy.select_action(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        memory_sizes.append(len(env.memory))

    return np.mean(rewards), np.mean(memory_sizes)


if __name__ == "__main__":
    env = MemoryManagementEnv()

    rb_reward, rb_mem = run_rule_based(env)
    rl_reward, rl_mem = run_rl(env)

    print("Rule-Based Policy:")
    print(f"Average Reward: {rb_reward:.2f}")
    print(f"Average Memory Size: {rb_mem:.2f}")

    print("\nRL-Based Policy (PPO):")
    print(f"Average Reward: {rl_reward:.2f}")
    print(f"Average Memory Size: {rl_mem:.2f}")
