import matplotlib.pyplot as plt

# Results from experiments
policies = ["Rule-Based", "RL-Based (PPO)"]
avg_rewards = [9.12, 16.94]
avg_memory_sizes = [4.66, 0.78]

# Plot 1: Average Reward Comparison
plt.figure()
plt.bar(policies, avg_rewards)
plt.ylabel("Average Episode Reward")
plt.title("Reward Comparison Between Memory Policies")
plt.tight_layout()
plt.savefig("RL_Demo/plots/reward_comparison.png")
plt.close()

# Plot 2: Average Memory Size Comparison
plt.figure()
plt.bar(policies, avg_memory_sizes)
plt.ylabel("Average Memory Size")
plt.title("Memory Usage Comparison Between Policies")
plt.tight_layout()
plt.savefig("RL_Demo/plots/memory_size_comparison.png")
plt.close()

print("Plots saved in RL_Demo/plots/ directory.")
