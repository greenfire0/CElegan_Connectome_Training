import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
from c_worm import *
from celegan_env import WormSimulationEnv


#from disembodiedConnectome import move, createpostSynaptic
#from trained_connectome import move, createpostSynaptic 
from trained_connectome_with_bias import wormConnectone,all_neuron_names
training_interval = 25  # Train the agent every 25 steps
total_episodes = 1000
env = WormSimulationEnv()
wormdude = wormConnectone(weight_m=np.zeros(len(all_neuron_names)))
#createpostSynaptic()
#agent = PolicyGradientAgent(weight_matrix)
for episode in range(total_episodes):
    observation = env._get_observation()
    done = False
    cumulative_reward = 0.0
    
    for step in range(training_interval):
        # Simulate the move function based on the observation distance
        movement = wormdude.move(observation[0], env.worm.sees_food)
        
        # Take a step in the environment
        #next_observation, reward, done, _ = env.step(movement)
        
        # Store transition in agent's memory
        #agent.store_transition(next_observation, reward)
        
        # Accumulate reward
        #cumulative_reward += reward
        
        env.render()
        
        if done:
            break
        
        #observation = next_observation
    
    # Learn after every `training_interval` steps
    #agent.learn()
    
    print(f"Episode {episode + 1} finished. Cumulative Reward: {cumulative_reward}")

    if done:
        break

env.close()