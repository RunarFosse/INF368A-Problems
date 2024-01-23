from bandit import Bandits_one
from mab import EpsilonGreedy

env = Bandits_one()

# Epsilon-greedy model
agent = EpsilonGreedy()
observation, reward, terminated, truncated, info = env.reset()
for timestep in range(1, 1001):
    action = agent.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    agent.update(action, reward, timestep)

    if terminated or truncated:
        observation, reward, terminated, truncated, info = env.reset()

# Decaying Epsilon-greedy model
agent = EpsilonGreedy(alpha=0.5)
observation, reward, terminated, truncated, info = env.reset()
for timestep in range(1, 1001):
    action = agent.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    agent.update(action, reward, timestep)
    print(action, end=" - ")

    if terminated or truncated:
        observation, reward, terminated, truncated, info = env.reset()