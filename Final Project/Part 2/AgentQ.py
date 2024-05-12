from BaseAgent import AbstractAgent

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

def observation_to_features(observation, action):
    """ Feature engineering helper function. """
    agent_pos = observation["agent"]["pos"]
    cookie_pos = observation["cookie"]["pos"]
    direction_to_cookie = agent_pos - cookie_pos
    velocity = observation["agent"]["vel"]
    time_left = observation["cookie"]["time"]

    action_vector = [0.0] * 3
    action_vector[action] = 1.0

    return torch.tensor([agent_pos, cookie_pos, direction_to_cookie, velocity, time_left] + action_vector, dtype=torch.float)


class Q_network(nn.Module):
    """ Deep Q-learning network. """
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 64)
        self.out = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.Tensor(x)

        # Pass through network
        out = F.tanh(self.fc1(x))
        out = F.tanh(self.fc2(out))
        return self.out(out)


class AgentQ(AbstractAgent):
    """ Agent using Q-learning following AbstractAgent interface. """
    def __init__(self, **kwargs):
        self.lr = kwargs["lr"] if "lr" in kwargs else 0.0001
        self.gamma = kwargs["gamma"] if "gamma" in kwargs else 0.95

        self.epsilon = 0.2
        self.actions = 3
        self.network = Q_network()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.cumulative_rewards = [0]
        
    def select_action(self, observation, epsilon_greedy=False):
        # Select best action optionally using epsilon_greedy
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.random.randint(self.actions)
        else:
            action = torch.argmax(torch.Tensor([self.network(observation_to_features(observation, a)) for a in range(self.actions)])).item()
        
        return action

    def learn(self, *args, **kwargs):
        env, timesteps = kwargs["env"], kwargs["timesteps"]
        observation, info = env.reset()

        for timestep in range(timesteps.int()):
            action = self.select_action(observation, epsilon_greedy=True)

            # Take the action and observe new state
            new_observation, reward, terminated, truncated, info = env.step(action)

            v_true = torch.tensor([reward]) + self.gamma * torch.max(torch.tensor([self.network(observation_to_features(new_observation, a)) for a in range(self.actions)]))
            v_pred = self.network(observation_to_features(observation, action))
            
            loss = torch.pow(v_true - v_pred, 2.0)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            observation = new_observation

            self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)

    def save(self, path: str = "") -> None:
        # Save the network's state_dict to disk
        return torch.save(self.network.state_dict(), path)

    def load(self, path: str = "") -> None:
        # Load a pretrained network state_dict from disk
        self.network.load_state_dict(torch.load(path))