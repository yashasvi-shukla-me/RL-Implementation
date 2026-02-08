# RL-Based Memory Management Simulation

This repository contains a small, self-contained simulation used for the empirical validation in the paper:

**“Reinforcement Learning for Memory Management in Language Model Agents”**

The implementation is intentionally lightweight. Its purpose is not to reproduce large-scale language model environments, but to provide a controlled setting for examining how reinforcement learning (RL) policies differ from simple rule-based heuristics when making memory retention and deletion decisions.

---

## Purpose and Scope

Modern language model agents often rely on handcrafted rules to decide what information to store, retrieve, or discard. The goal of this simulation is to explore, in a simplified setting, whether an RL-based controller can learn memory management behaviors that differ meaningfully from such fixed heuristics.

This code should be viewed as:

- A **toy environment**
- A **sanity-check experiment**
- A **supporting empirical artifact**, not a full benchmark

It is **not** intended to claim generalization to complex environments such as ALFWorld or WebArena.

---

## Environment Overview

The environment models memory management as a sequential decision-making problem:

- At each timestep, the agent observes an incoming information item with a scalar importance score.
- The agent maintains a bounded external memory buffer.
- The agent selects one of three actions:
  - Store the current item
  - Ignore the item
  - Delete the oldest memory entry

The observation space includes:

- Importance of the current item
- Normalized memory size
- Average importance of items stored in memory

Rewards are designed to reflect common trade-offs in memory management, encouraging the retention of useful information while discouraging unnecessary memory growth.

---

## Policies Implemented

Two memory control strategies are compared:

1. **Rule-Based Heuristic**
   - Stores items only if their importance exceeds a fixed threshold
   - Reflects common handcrafted memory rules used in practice

2. **RL-Based Policy (PPO)**
   - Trained using Proximal Policy Optimization
   - Learns memory actions directly from reward feedback
   - Operates under the same observation and action space as the baseline

Both policies are evaluated under identical environmental conditions.

---

## Running the Experiments

Create and activate a virtual environment, then install dependencies:

```bash
pip install numpy matplotlib gymnasium stable-baselines3
```

Run the main experiment:

```bash
python -m RL_Demo.experiments.run_experiment
```

Generate plots:

```bash
python -m RL_Demo.experiments.plot_results
```

The resulting figures correspond to those reported in the paper.

### Results

In the reported experiments, the RL-based policy achieved:

- Higher average cumulative reward
- More compact memory usage
  These results support the qualitative claim that learned policies can become more selective than static heuristics, even in simplified settings.

### Limitations

This implementation intentionally abstracts away many aspects of real language model agents, including:

- Natural language inputs
- Retrieval noise
- Long-horizon task dependencies
  The results should therefore be interpreted as illustrative rather than definitive.

### Citation

If you use or reference this code, please cite the associated paper.

```bash
@misc{memoryrlcode2025,
author = {Shukla, Yashasvi},
title = {RL-Based Memory Management Simulation},
year = {2025},
url = {https://github.com/yashasvi-shukla-me/RL-Implementation}
}
```

### License

This code is provided for academic and research purposes.

YASHASVI SHUKLA (Yashasvi Shukla)

### Screenshots

<img width="1194" height="286" src="https://github.com/user-attachments/assets/411c7738-6d51-442f-ba7d-ba994b25fa9e" />
<img width="1469" height="878" src="https://github.com/user-attachments/assets/e67278a0-1744-42d4-8737-cc61e64faf9e" />
<img width="843" height="320"  src="https://github.com/user-attachments/assets/68fe91b4-3540-4e9c-866d-5adccccfab1c" />






