import torch
import numpy as np
import random

SEED = 7   # running for seeds 42, 123 and 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import gymnasium as gym
import matplotlib.pyplot as plt
from agents.dqn import DQNAgent


def train(num_episodes=4000):
    env   = gym.make("LunarLander-v3")

    agent = DQNAgent(
        state_size=8,
        action_size=4,
        hidden_size=128,
        lr=0.0005,
        epsilon_decay=0.9995,
        target_update=500,    # was 1000 — more frequent updates helps on LunarLander
        buffer_size=50000
    )

    rewards_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        # Decay epsilon 
        agent.epsilon = max(agent.epsilon_min,
                            agent.epsilon * agent.epsilon_decay)

        rewards_history.append(total_reward)

        # Print progress every 50 episodes
        if (episode + 1) % 100 == 0:
            avg = sum(rewards_history[-100:]) / 100
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward (last 50): {avg:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards_history

def plot_results(rewards, seed):
    # Compute rolling average over 50 episodes
    rolling = [sum(rewards[max(0,i-49):i+1]) /
               min(50, i+1) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    plt.plot(rolling, color='steelblue', linewidth=2, label='50-ep Rolling Avg')
    plt.axhline(y=200, color='red', linestyle='--', label='Solved (200)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'DQN on LunarLander-v3 (Seed {seed})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'experiments/results/dqn_lunarlander_seed{seed}.png', dpi=150)
    # Add inside plot_results(), after rolling avg calculation:
    np.save(f'experiments/results/dqn_rewards_lunar_seed{seed}.npy', np.array(rewards))
    plt.show()
    print("Plot saved to experiments/results/dqn_cartpole.png")

if __name__ == "__main__":
    rewards = train(num_episodes=4000)
    plot_results(rewards, SEED)