import matplotlib.pyplot as plt
from gym import spaces
import gym
import numpy as np
import math
import random
from c_worm import Worm
from trained_connectome import wormConnectone

class WormSimulationEnv(gym.Env):
    def __init__(self, num_worms=1):
        self.dimx = 1600
        self.dimy = 1200
        self.num_worms = num_worms
        super(WormSimulationEnv, self).__init__()
        self.foodradius = 20
        self.fig, self.ax = plt.subplots()
        self.range = 200
        self.reset()

    def generate_circle_of_food(self, num_food=40, radius=200):
        for i in range(num_food):
            angle = i * (2 * math.pi / num_food)
            food_x = self.dimx-200 / 2 + radius * math.cos(angle)
            food_y = self.dimy-200 / 2 + radius * math.sin(angle)
            self.food.append([food_x, food_y])

    def generate_random_food(self, num_food=2):
        for _ in range(num_food):
            food_x = np.random.uniform(0, self.dimx)
            food_y = np.random.uniform(0, self.dimy)
            self.food.append([food_x, food_y])
    
    def generate_food_pattern(self):
        self.food.append([1000, 600])
        self.food.append([900, 1000])
        self.food.append([925, 1000])
        self.food.append([950, 1000])
    
    def reset(self):
        self.worms = [Worm(position=[self.dimx/2, self.dimy/2],range=self.range) for _ in range(self.num_worms)]
        self.food = []
        self.generate_circle_of_food()
        self.generate_food_pattern()
        #self.generate_random_food(20)
        return self._get_observations()


    def step(self, actions,worm_num,candidate):
        left_speed, right_speed,speed = actions
        self.worms[worm_num].update(left_speed=left_speed, right_speed=right_speed, food_positions=self.food)

        observations = self._get_observations()
        rewards = self._calculate_rewards(worm_num)
        self._check_eat_food(candidate)
        done = self._check_done()


        return observations, rewards, done, {}

    def _check_eat_food(self,candidate):
        for worm in self.worms:
            for food in self.food:
                if np.linalg.norm(np.array(worm.position) - np.array(food)) < self.foodradius:
                    if food == [1000, 600]:
                        candidate.move(10000,False,-10)
                        
                    self.food.remove(food)
                    break

    def render(self,worm_num=0, mode='human'):
        self.ax.clear()
        worm = self.worms[worm_num]
        self.ax.plot(worm.position[0], worm.position[1], 'ro')
        self.ax.plot([worm.position[0], worm.position[0] + 100 * np.cos(worm.facing_dir)],
                         [worm.position[1], worm.position[1] + 100 * np.sin(worm.facing_dir)], 'b-')


        worm = self.worms[worm_num]
        for f in self.food:
            if worm.is_food_close(f):
                    self.ax.plot(f[0], f[1], 'yo')
            else:
                    self.ax.plot(f[0], f[1], 'bo')

        self.ax.set_xlim(0, self.dimx)
        self.ax.set_ylim(0, self.dimy)
        plt.pause(0.01)

    def _get_observations(self):
        observations = []
        for worm in self.worms:
            distance_to_left_wall = worm.position[0]
            distance_to_right_wall = self.dimx - worm.position[0]
            distance_to_top_wall = worm.position[1]
            distance_to_bottom_wall = self.dimy - worm.position[1]

            min_distance_to_wall = min(distance_to_left_wall, distance_to_right_wall, distance_to_top_wall, distance_to_bottom_wall)

            observation = np.array([
                min_distance_to_wall,
                worm.position[0],
                worm.position[1],
                worm.facing_dir,
                worm.sees_food,
            ])
            observations.append(observation)
        return np.array(observations)

    def _calculate_rewards(self,worm_num):
        worm= self.worms[worm_num]
        wall_reward = 0
        if any(np.linalg.norm(np.array(worm.position) - np.array(food)) < self.foodradius for food in self.food):
                wall_reward += 20
                
        vision_radius = self.range  # The maximum distance for food reward gradient

        # Calculate the reward based on the distance to each food item
        for food in self.food:
            distance_to_food = np.linalg.norm(np.array(worm.position) - np.array(food))
            if distance_to_food < vision_radius:
                wall_reward += max(0, (vision_radius - distance_to_food) / vision_radius) / 30


        return wall_reward

    def _check_done(self):
        return len(self.food) == 0

    def close(self):
        plt.close()
