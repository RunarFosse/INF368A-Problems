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
        
    def update(self, action, reward, timestep):
        self.action_counters[action] += 1
        self.expected_reward[action] += 1 / timestep * (reward - self.expected_reward[action])
        self.epsilon *= self.alpha