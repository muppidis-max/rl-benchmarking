import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64,
                 lr=0.001, epsilon_decay=0.997, target_update=500, buffer_size=10000):
        
        self.state_size  = state_size    
        self.action_size = action_size   

        # Hyperparameters
        self.gamma        = 0.99
        self.lr           = lr
        self.batch_size   = 64
        self.buffer_size  = buffer_size
        self.target_update= target_update
        self.epsilon      = 1.0
        self.epsilon_min  = 0.05
        self.epsilon_decay= epsilon_decay

        # Networks
        self.online_net = QNetwork(state_size, action_size, hidden_size)
        self.target_net = QNetwork(state_size, action_size, hidden_size)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.memory    = ReplayBuffer(self.buffer_size)
        self.steps     = 0

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

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # DQN target
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()