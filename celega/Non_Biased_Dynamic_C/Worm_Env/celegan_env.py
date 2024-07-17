import matplotlib.pyplot as plt
import gym
import numpy as np
import math
from numba import njit
from Worm_Env.c_worm import Worm,is_food_close

class WormSimulationEnv(gym.Env):
    def __init__(self, num_worms=1):
        self.dimx = 1600
        self.dimy = 1200
        self.num_worms = num_worms
        super(WormSimulationEnv, self).__init__()
        self.foodradius = 20
        self.fig, self.ax = plt.subplots()
        self.range = 150
        self.reset(0, 40)

    @staticmethod
    @njit
    def generate_circle_of_food(num_food, radius, center_x, center_y):
        food = np.empty((num_food, 2))
        for i in range(num_food):
            angle = i * (2 * math.pi / num_food)
            food_x = center_x + radius * math.cos(angle)
            food_y = center_y + radius * math.sin(angle)
            food[i] = [food_x, food_y]
        return food

    @staticmethod
    @njit
    def calculate_rewards(worm_pos, food_positions, foodradius, vision_radius):
        reward = 0.0
        for f in food_positions:
            distance_to_food = np.linalg.norm(worm_pos - f)
            if distance_to_food < foodradius:
                reward += 30.0
            if distance_to_food < vision_radius:
                reward += max(0.0, (vision_radius - distance_to_food) / vision_radius) / 30.0
        return reward

    @staticmethod
    @njit
    def generate_food_pattern(pattern_type, num_food, dimx, dimy):
        food = []
        center_x = dimx / 2
        center_y = dimy / 2

        if pattern_type == 0:  # Random
            while len(food) < num_food:
                food_x = np.random.uniform(0, dimx)
                food_y = np.random.uniform(0, dimy)
                distance_from_center = np.sqrt((food_x - center_x) ** 2 + (food_y - center_y) ** 2)
                if distance_from_center >= 400:
                    food.append([food_x, food_y])
            
        elif pattern_type == 1:  # Grid
            grid_size = int(np.sqrt(num_food))
            spacing_x = dimx / (grid_size + 1)
            spacing_y = dimy / (grid_size + 1)
            for i in range(1, grid_size + 1):
                for j in range(1, grid_size + 1):
                    food.append([i * spacing_x, j * spacing_y])
                    
        elif pattern_type == 2:  # Clusters
            cluster_centers = [
                [dimx / 2, dimy / 4],
                [dimx / 4, dimy / 2],
                [dimx * 3 / 4, dimy / 2],
                [dimx / 2, dimy * 3 / 4]
            ]
            for center in cluster_centers:
                for _ in range(num_food // len(cluster_centers)):
                    offset_x = np.random.uniform(-50, 50)
                    offset_y = np.random.uniform(-50, 50)
                    food.append([center[0] + offset_x, center[1] + offset_y])
                    
        elif pattern_type == 3:  # Square
            side_length = min(dimx, dimy) / 2
            for i in range(num_food):
                side = i // (num_food // 4)
                position = (i % (num_food // 4)) / (num_food // 4 - 1)
                if side == 0:  # Top side
                    food_x = center_x - side_length / 2 + position * side_length
                    food_y = center_y - side_length / 2
                elif side == 1:  # Right side
                    food_x = center_x + side_length / 2
                    food_y = center_y - side_length / 2 + position * side_length
                elif side == 2:  # Bottom side
                    food_x = center_x + side_length / 2 - position * side_length
                    food_y = center_y + side_length / 2
                else:  # Left side
                    food_x = center_x - side_length / 2
                    food_y = center_y + side_length / 2 - position * side_length
                food.append([food_x, food_y])

        elif pattern_type == 4:  # Circle
            radius = min(dimx, dimy) / 4
            for i in range(num_food):
                angle = 2 * np.pi * i / num_food
                food_x = center_x + radius * np.cos(angle)
                food_y = center_y + radius * np.sin(angle)
                food.append([food_x, food_y])

        elif pattern_type == 5:  # Triangle
            top_vertex = (dimx / 2, dimy / 4)
            left_vertex = (dimx / 4, dimy * 3 / 4)
            right_vertex = (dimx * 3 / 4, dimy * 3 / 4)
            for i in range(num_food):
                p = i / (num_food - 1)
                if p <= 1/3:
                    ratio = p / (1/3)
                    x = top_vertex[0] + ratio * (left_vertex[0] - top_vertex[0])
                    y = top_vertex[1] + ratio * (left_vertex[1] - top_vertex[1])
                elif p <= 2/3:
                    ratio = (p - 1/3) / (1/3)
                    x = left_vertex[0] + ratio * (right_vertex[0] - left_vertex[0])
                    y = left_vertex[1] + ratio * (right_vertex[1] - left_vertex[1])
                else:
                    ratio = (p - 2/3) / (1/3)
                    x = right_vertex[0] + ratio * (top_vertex[0] - right_vertex[0])
                    y = right_vertex[1] + ratio * (top_vertex[1] - right_vertex[1])
                food.append([x, y])
        
        return np.array(food)

    def reset(self, pattern_type, num_food=40):
        self.worms = [Worm(position=[self.dimx/2, self.dimy/2], range=self.range) for _ in range(self.num_worms)]
        self.food = np.array(WormSimulationEnv.generate_food_pattern(pattern_type, num_food, self.dimx, self.dimy))
        return self._get_observations()

    def step(self, actions, worm_num, candidate):
        left_speed, right_speed = actions
        self.worms[worm_num].update(left_speed=left_speed, right_speed=right_speed, food_positions=self.food)

        observations = self._get_observations()
        
        worm_pos = self.worms[worm_num].position
        
        rewards = WormSimulationEnv.calculate_rewards(worm_pos, self.food, self.foodradius, self.range)
        
        self._check_eat_food(worm_pos)
        done = self._check_done()

        return observations, rewards, done

    def _check_eat_food(self, worm_pos):
        to_remove = [i for i, food in enumerate(self.food) if np.linalg.norm(worm_pos - food) < self.foodradius]
        self.food = np.delete(self.food, to_remove, axis=0)
        del to_remove

    def render(self, worm_num=0, mode='human'):
        self.ax.clear()
        worm = self.worms[worm_num]
        self.ax.plot(worm.position[0], worm.position[1], 'ro')
        self.ax.plot([worm.position[0], worm.position[0] + 100 * np.cos(worm.facing_dir)],
                     [worm.position[1], worm.position[1] + 100 * np.sin(worm.facing_dir)], 'b-')

        for f in self.food:
            if is_food_close(worm.position,f,150):
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



    def _check_done(self):
        return len(self.food) == 0

    def close(self):
        plt.close()