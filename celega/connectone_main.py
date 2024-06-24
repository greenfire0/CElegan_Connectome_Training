import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
from c_worm import *
from celegan_env import WormSimulationEnv
from disembodiedConnectome import move, createpostSynaptic

env = WormSimulationEnv()
createpostSynaptic()

for episode in range(1000):
    
    observation = env._get_observation()
    
    # Simulate the move function based on the observation distance
    movement = move(observation[0],env.worm.sees_food)

    # Since the move function does not return an action, we can step the environment directly
    # The observation, reward, done, and info are obtained from the environment's step method
    next_observation, reward, done, _ = env.step(movement)  # Assuming step can handle None action
    
    observation = next_observation
        
    env.render()
    
    if done:
        print(f"Episode {episode} finished.")
        break

env.close()
