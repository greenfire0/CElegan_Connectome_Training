import matplotlib.pyplot as plt
from gym import spaces
import gym
import numpy as np
import math
import random
from c_worm import Worm
from trained_connectome_with_bias import wormConnectone

class WormSimulationEnv(gym.Env):
    def __init__(self, num_worms=1):
        self.dimx = 1600
        self.dimy = 1200
        self.num_worms = num_worms
        super(WormSimulationEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([[-1.0, -1.0]] * num_worms), 
                                       high=np.array([[1.0, 1.0]] * num_worms), 
                                       dtype=np.float32)  # Continuous action space for motor speeds
        self.observation_space = spaces.Box(low=np.array([[0, 0, 0, -np.pi, 0]] * num_worms, dtype=np.float64),
                                            high=np.array([[self.dimx, self.dimx, self.dimy/2, np.pi, 1]] * num_worms, dtype=np.float64),
                                            dtype=np.float64)  # (distance_to_wall, x, y, facingDir, seesFood)
        self.worms = [Worm(position=[self.dimx/2, self.dimy/2]) for _ in range(num_worms)]
        self.food = []
        self.foodradius = 20
        self.fig, self.ax = plt.subplots()
        self.generate_food_pattern()
        self.steps_without_food = 0
        self.max_steps_without_food = 25

    def generate_circle_of_food(self, num_food=40, radius=200):
        for i in range(num_food):
            angle = i * (2 * math.pi / num_food)
            food_x = self.dimx / 2 + radius * math.cos(angle)
            food_y = self.dimy / 2 + radius * math.sin(angle)
            self.food.append([food_x, food_y])

    def generate_random_food(self, num_food=2):
        for _ in range(num_food):
            food_x = np.random.uniform(0, self.dimx)
            food_y = np.random.uniform(0, self.dimy)
            self.food.append([food_x, food_y])
    
    def generate_food_pattern(self):
        self.food.append([1000, 600])
        self.food.append([100, 1000])
        self.food.append([125, 1000])
        self.food.append([150, 1000])
    
    def reset(self):
        self.worms = [Worm(position=[self.dimx/2, self.dimy/2]) for _ in range(self.num_worms)]
        self.food = []
        self.generate_circle_of_food()
        self.steps_without_food = 0
        return self._get_observations()

    def step(self, actions):
        for i, worm in enumerate(self.worms):
            left_speed, right_speed = actions[i]
            worm.update(left_speed=left_speed, right_speed=right_speed, food_positions=self.food)

        observations = self._get_observations()
        rewards = self._calculate_rewards()
        self._check_eat_food()
        done = self._check_done()

        self.steps_without_food += 1

        return observations, rewards, done, {}

    def _check_eat_food(self):
        for worm in self.worms:
            for food in self.food:
                if np.linalg.norm(np.array(worm.position) - np.array(food)) < self.foodradius:
                    if food == [1000, 600]:
                        #fireNeuron("BIAS")
                        pass
                    self.food.remove(food)
                    self.steps_without_food = 0
                    break

    def render(self, mode='human'):
        self.ax.clear()
        for worm in self.worms:
            self.ax.plot(worm.position[0], worm.position[1], 'ro')
            self.ax.plot([worm.position[0], worm.position[0] + 20 * np.cos(worm.facing_dir)],
                         [worm.position[1], worm.position[1] + 20 * np.sin(worm.facing_dir)], 'b-')

            vision_radius = 100
            vision_angle = np.pi / 4
            for angle_offset, color in zip([-vision_angle, 0, vision_angle], ['r', 'g', 'b']):
                vision_end_x = worm.position[0] + vision_radius * np.cos(worm.facing_dir + angle_offset)
                vision_end_y = worm.position[1] + vision_radius * np.sin(worm.facing_dir + angle_offset)
                self.ax.plot([worm.position[0], vision_end_x], [worm.position[1], vision_end_y], color + '-')

        for f in self.food:
            for worm in self.worms:
                if worm._is_food_in_vision(f):
                    self.ax.plot(f[0], f[1], 'yo')
                else:
                    self.ax.plot(f[0], f[1], 'bo')

        self.ax.set_xlim(0, self.dimx)
        self.ax.set_ylim(0, self.dimy)
        plt.pause(0.01)

    def _get_observation(self):
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

    def _calculate_rewards(self):
        rewards = []
        for worm in self.worms:
            wall_reward = 0
            if any(np.linalg.norm(np.array(worm.position) - np.array(food)) < self.foodradius for food in self.food):
                self.steps_without_food = 0
                rewards.append(5)
                continue

            if worm.sees_food:
                wall_reward += 0.5

            if self.steps_without_food <= 25:
                wall_reward += 2

            rewards.append(wall_reward)
        return np.array(rewards)

    def _check_done(self):
        return len(self.food) == 0

    def close(self):
        plt.close()
