import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
from worm import Worm
from brain import PolicyGradientAgent
from cbrain import C_Elegans_Agent
from worm_environment import WormSimulationEnv





env = WormSimulationEnv()
agent = C_Elegans_Agent(env.observation_space, env.action_space)

for episode in range(1000):
    observation = env._get_observation()
    total_reward = 0
    step_counter = 0  # Initialize step counter for the episode

    while step_counter < 5:  # Limit each episode to 10 steps
        action = agent.choose_action(observation)
        next_observation, reward, done, _ = env.step(action)
        agent.store_transition(observation, action, reward)
        observation = next_observation
        total_reward += reward
        step_counter += 1  # Increment step counter
        
        env.render()
        if done or step_counter == 3:  # Check if the episode is done or max steps reached
            agent.learn()
            break
        

env.close()

