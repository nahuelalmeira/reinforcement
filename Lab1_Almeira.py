import numpy as np
import gym
from Lab1Functions import SARSA_run

random_state = np.random.RandomState(42)

alpha = 0.5
gamma = 1
epsilon = 0.35
tau = 25

episodes_to_run = 2000

env = gym.make("CliffWalking-v0")

avg_steps_per_eps, timesteps_eps, reward_eps, q = SARSA_run(env, episodes_to_run, alpha, gamma, epsilon, random_state)

print(timesteps_eps)
print(reward_eps)