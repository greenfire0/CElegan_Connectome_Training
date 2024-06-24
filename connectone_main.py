import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
from worm import Worm
from worm_environment import WormSimulationEnv
from disembodiedConnectome import move




env = WormSimulationEnv()

for episode in range(1000):
    observation = env._get_observation()
    print(observation[2])
    
    action = move(observation[2])
    next_observation, reward, done, _ = env.step(action)
    observation = next_observation
        
    env.render()

env.close()

