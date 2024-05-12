from collections import deque

import numpy as np

# Standard UCB implementation
class UCB():
    def __init__(self, constant=1, actions=3):
        self.expected_reward = [ 1000 ] * actions
        self.action_counters = [ 0 ] * actions
        self.constant = constant
        self.actions = actions

    def sample(self, episode):
        # max(N_i, 1) prevents division by zero error
        confidence_bounds = [np.sqrt(2 * np.log(episode) / max(N_i, 1)) for N_i in self.action_counters]
        action = np.argmax(np.array(self.expected_reward) + self.constant * np.array(confidence_bounds))

        return action
        
    def update(self, action, reward):
        # Update actions and expected reward
        self.action_counters[action] += 1
        self.expected_reward[action] += (reward - self.expected_reward[action]) / self.action_counters[action]


# UCB implementation with sliding window average
class Sliding_UCB(UCB):
    def __init__(self, sliding_window_size, constant=1, actions=3):
        super().__init__(constant, actions)
        self.rewards = [deque() for _ in range(actions)]
        self.sliding_window_size = sliding_window_size
    
    def update(self, action, reward):
        # Update action count
        self.action_counters[action] += 1

        if self.action_counters[action] >= self.sliding_window_size:
            # Add and remove from sliding window (and stored expected value)
            self.rewards[action].append(reward)
            removed_reward = self.rewards[action].popleft()
            self.expected_reward[action] -= removed_reward / min(self.action_counters[action], self.sliding_window_size)
        
        self.expected_reward[action] += (reward - self.expected_reward[action]) / min(self.action_counters[action], self.sliding_window_size)
