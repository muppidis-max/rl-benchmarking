import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Dueling Network
class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        features   = self.feature(x)
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


# Prioritized Experience Replay
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_steps=6000):
        self.capacity       = capacity
        self.alpha          = alpha
        self.beta           = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_steps
        self.buffer         = []
        self.priorities     = np.zeros(capacity, dtype=np.float32)
        self.pos            = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        batch   = [self.buffer[i] for i in indices]

        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states),
            np.array(dones,       dtype=np.float32),
            indices,
            np.array(weights,     dtype=np.float32)
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)


# Rainbow Agent - epsilon-greedy exploration
class RainbowAgent:
    def __init__(self, state_size, action_size,
                 lr=0.001, epsilon_decay=0.997, target_update=500,
                 buffer_size=50000, beta_steps=6000, alpha=0.6):

        self.state_size  = state_size
        self.action_size = action_size

        self.gamma        = 0.99
        self.lr           = lr
        self.batch_size   = 64
        self.buffer_size  = buffer_size
        self.target_update= target_update
        self.n_steps      = 1

        self.epsilon      = 1.0
        self.epsilon_min  = 0.005           
        self.epsilon_decay= epsilon_decay

        self.online_net = DuelingNetwork(state_size, action_size)
        self.target_net = DuelingNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.memory = PrioritizedReplayBuffer(
            self.buffer_size,
            alpha=alpha,               
            beta_steps=beta_steps
        )
        self.n_step_buffer = deque(maxlen=self.n_steps)
        self.steps         = 0

    def _get_n_step_transition(self):
        n_reward = 0
        for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            n_reward += (self.gamma ** i) * r
            if d:
                next_state = ns
                done = d
                break
        else:
            next_state = self.n_step_buffer[-1][3]
            done       = self.n_step_buffer[-1][4]
        state  = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        return state, action, n_reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_steps:
            return
        self.memory.push(*self._get_n_step_transition())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)

        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)
        weights     = torch.FloatTensor(weights)

        # Double Q-learning target with n-step discount
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(1, keepdim=True)
            max_next_q   = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q     = rewards + (self.gamma ** self.n_steps) * max_next_q * (1 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        td_errors = (current_q - target_q).detach().abs().numpy()
        loss = (weights * nn.MSELoss(reduction='none')(current_q, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()