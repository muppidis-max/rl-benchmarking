import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from agents.rainbow import RainbowAgent

SEED = 123   # running for seeds 42, 123 and 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(num_episodes=800):
    env   = gym.make("CartPole-v1")
    agent = RainbowAgent(state_size=4, action_size=2)

    rewards_history = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        # n-step buffer flush
        while len(agent.n_step_buffer) > 0:
            agent.n_step_buffer.popleft()

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min,
                            agent.epsilon * agent.epsilon_decay)

        rewards_history.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg = sum(rewards_history[-50:]) / 50
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward (last 50): {avg:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards_history

def plot_results(rewards, seed):
    rolling = [sum(rewards[max(0,i-49):i+1]) /
               min(50, i+1) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    plt.plot(rolling, color='steelblue', linewidth=2, label='50-ep Rolling Avg')
    plt.axhline(y=195, color='red', linestyle='--', label='Solved (195)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Rainbow on CartPole-v1 (Seed {seed})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'experiments/results/rainbow_cartpole_seed{seed}.png', dpi=150)
    np.save(f'experiments/results/rainbow_rewards_seed{seed}.npy', np.array(rewards))
    plt.show()
    print(f"Plot saved.")

if __name__ == "__main__":
    rewards = train(num_episodes=800)
    plot_results(rewards, SEED)