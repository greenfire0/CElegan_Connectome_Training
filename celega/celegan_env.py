import matplotlib.pyplot as plt
from gym import spaces
import gym
import numpy as np
import math
import random
from c_worm import Worm


class WormSimulationEnv(gym.Env):
    def __init__(self):
        self.dimx = 1600
        self.dimy = 1200
        super(WormSimulationEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)  # Continuous action space for motor speeds
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -np.pi, 0], dtype=np.float64),
                                            high=np.array([self.dimx, self.dimx, self.dimy/2, np.pi, 1], dtype=np.float64),
                                            dtype=np.float64)  # (distance_to_wall, x, y, facingDir, seesFood)
        self.worm = Worm(position=[self.dimx/2, self.dimy/2])
        self.food = []
        self.foodradius = 20
        self.fig, self.ax = plt.subplots()
        self.generate_circle_of_food()
        
        self.steps_without_food = 0
        self.max_steps_without_food = 25

    def generate_circle_of_food(self, num_food=40, radius=200):
        for i in range(num_food):
            angle = i * (2 * math.pi / num_food)
            food_x = self.worm.position[0] + radius * math.cos(angle)
            food_y = self.worm.position[1] + radius * math.sin(angle)
            self.food.append([food_x, food_y])

    def generate_random_food(self, num_food=2):
        for _ in range(num_food):
            food_x = np.random.uniform(0, self.dimx)
            food_y = np.random.uniform(0, self.dimy)
            self.food.append([food_x, food_y])

    def reset(self):
        # Reset all variables and generate new food
        self.worm = Worm(position=[self.dimx/2, self.dimy/2])
        self.food = []
        self.generate_circle_of_food()
        self.steps_without_food = 0  # Reset steps without food counter
        return self._get_observation()

    def step(self, action):
        left_speed, right_speed = action[0],action[1]
        self.worm.update(left_speed=left_speed, right_speed=right_speed,food_positions=self.food)

        if random.random() < 0.005:
            self.generate_random_food()

        observation = self._get_observation()
        reward = self._calculate_reward()
        self._check_eat_food()
        done = self._check_done()

        if self.worm.sees_food:
            reward += 0.25  # Add reward for seeing food

        # Increment steps without food counter
        self.steps_without_food += 1

        return observation, reward, done, {}

    def _check_eat_food(self):
        for food in self.food:
            if np.linalg.norm(np.array(self.worm.position) - np.array(food)) < self.foodradius:
                self.food.remove(food)
                self.steps_without_food = 0  # Reset steps without food counter
                break

    def render(self, mode='human'):
        self.ax.clear()
        self.ax.plot(self.worm.position[0], self.worm.position[1], 'ro')
        self.ax.plot([self.worm.position[0], self.worm.position[0] + 20 * np.cos(self.worm.facing_dir)],
                     [self.worm.position[1], self.worm.position[1] + 20 * np.sin(self.worm.facing_dir)], 'b-')

        vision_radius = 100
        vision_angle = np.pi / 4
        for angle_offset, color in zip([-vision_angle, 0, vision_angle], ['r', 'g', 'b']):
            vision_end_x = self.worm.position[0] + vision_radius * np.cos(self.worm.facing_dir + angle_offset)
            vision_end_y = self.worm.position[1] + vision_radius * np.sin(self.worm.facing_dir + angle_offset)
            self.ax.plot([self.worm.position[0], vision_end_x], [self.worm.position[1], vision_end_y], color + '-')

        for f in self.food:
            if self.worm._is_food_in_vision(f):
                self.ax.plot(f[0], f[1], 'yo')
            else:
                self.ax.plot(f[0], f[1], 'bo')

        self.ax.set_xlim(0, self.dimx)
        self.ax.set_ylim(0, self.dimy)
        plt.pause(0.01)

    def _get_observation(self):
        # Calculate distance to nearest wall
        distance_to_left_wall = self.worm.position[0]
        distance_to_right_wall = self.dimx - self.worm.position[0]
        distance_to_top_wall = self.worm.position[1]
        distance_to_bottom_wall = self.dimy - self.worm.position[1]
        
        min_distance_to_wall = min(distance_to_left_wall, distance_to_right_wall, distance_to_top_wall, distance_to_bottom_wall)
        
        observation = np.array([
            min_distance_to_wall,
            self.worm.position[0],
            self.worm.position[1],
            self.worm.facing_dir,
            self.worm.sees_food,
        ])
        return observation

    def _calculate_reward(self):
        if any(np.linalg.norm(np.array(self.worm.position) - np.array(food)) < self.foodradius for food in self.food):
            self.steps_without_food = 0  # Reset steps without food counter
            return 5  # Reward for eating food

        # Apply penalty for consecutive steps without food with logarithmic scaling
        penalty = -1.1 * np.log(self.steps_without_food + 1)

        if 0 < self.worm.position[0] < self.dimx and 0 < self.worm.position[1] < self.dimy:
            wall_reward = 0.1
        elif not (0 < self.worm.position[0] < self.dimx) and not (0 < self.worm.position[1] < self.dimy):
            wall_reward = -5
        else:
            wall_reward = -0.1

        if self.worm.sees_food:
            wall_reward += 0.5

        if self.steps_without_food <= 25:
            wall_reward += 2

        return wall_reward + penalty

    def _check_done(self):
        return len(self.food) == 0

    def close(self):
        plt.close()
