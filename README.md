# rl-benchmarking
# Deep RL Benchmarking — DQN, DDQN, and Rainbow

Benchmarking three deep reinforcement learning algorithms across 
CartPole-v1 and LunarLander-v3.

## Algorithms
- DQN (Mnih et al., 2013)
- Double DQN (van Hasselt et al., 2015)
- Rainbow (Hessel et al., 2018)

## Environments
- CartPole-v1 — solved at rolling avg ≥ 195
- LunarLander-v3 — solved at rolling avg ≥ 200

## Results
All 18 runs (3 algorithms × 3 seeds × 2 environments) 
successfully solved both environments.

## Requirements
python >= 3.8
gymnasium[box2d]
torch
numpy
matplotlib

## Usage
pip install -r requirements.txt
python train_cartpole.py
python train_lunarlander.py
