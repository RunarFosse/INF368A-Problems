from abc import abstractmethod
import random

import numpy as np

class MAB():
    """ Multi Armed Bandit superclass """
    @abstractmethod
    def sample(self):
        """ Call to retrieve action """
        pass
    
    @abstractmethod
    def update(self):
        """ Call to update model """
        pass


class EpsilonGreedy(MAB):
    def __init__(self, epsilon=0.3, alpha=1, actions=3):
        super().__init__()

        self.expected_reward = [ 1000 ] * actions
        self.action_counters = [ 0 ] * actions
        self.epsilon = epsilon
        self.actions = actions

        # Optional Decaying Epsilon-greedy parameter (Default no decay)
        self.alpha = alpha

    def sample(self):
        e = random.random()
        if e < self.epsilon:
            action = random.randint(0, self.actions - 1)
        else:
            action = np.argmax(self.expected_reward)
        
        return action
        
    def update(self, action, reward):
        self.action_counters[action] += 1
        self.expected_reward[action] += (reward - self.expected_reward[action]) / self.action_counters[action]
        self.epsilon *= self.alpha


class UCB(MAB):
    def __init__(self, constant=1, actions=3):
        super().__init__()

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
        self.action_counters[action] += 1
        self.expected_reward[action] += (reward - self.expected_reward[action]) / self.action_counters[action]