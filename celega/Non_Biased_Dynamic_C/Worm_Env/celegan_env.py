import matplotlib.pyplot as plt
import gym
import numpy as np
import math
from Worm_Env.c_worm import Worm

class WormSimulationEnv(gym.Env):
    def __init__(self, num_worms=1):
        self.dimx = 1600
        self.dimy = 1200
        self.num_worms = num_worms
        super(WormSimulationEnv, self).__init__()
        self.foodradius = 20
        self.fig, self.ax = plt.subplots()
        self.range = 150
        #self.reset()

    def generate_circle_of_food(self, num_food=40, radius=200):
        for i in range(num_food):
            angle = i * (2 * math.pi / num_food)
            food_x = self.dimx-200 / 2 + radius * math.cos(angle)
            food_y = self.dimy-200 / 2 + radius * math.sin(angle)
            self.food.append([food_x, food_y])

    def generate_random_food(self, num_food=2):
        center_x = self.dimx / 2
        center_y = self.dimy / 2

        while len(self.food) < num_food:
            food_x = np.random.uniform(0, self.dimx)
            food_y = np.random.uniform(0, self.dimy)

            # Calculate distance from the center
            distance_from_center = np.sqrt((food_x - center_x) ** 2 + (food_y - center_y) ** 2)

            if distance_from_center >= 400:
                self.food.append([food_x, food_y])

    def generate_food_pattern(self, pattern_type="random", num_food=10):
        self.food = []
        
        if pattern_type == "random":
            self.generate_random_food(num_food)
            
        elif pattern_type == "grid":
            grid_size = int(np.sqrt(num_food))
            spacing_x = self.dimx / (grid_size + 1)
            spacing_y = self.dimy / (grid_size + 1)
            for i in range(1, grid_size + 1):
                for j in range(1, grid_size + 1):
                    self.food.append([i * spacing_x, j * spacing_y])
                    
        elif pattern_type == "clusters":
            cluster_centers = [
                [self.dimx / 2, self.dimy / 4], 
                [self.dimx / 4, self.dimy / 2], 
                [self.dimx * 3 / 4, self.dimy / 2], 
                [self.dimx / 2, self.dimy * 3 / 4]
            ]
            for center in cluster_centers:
                for _ in range(num_food // len(cluster_centers)):
                    offset_x = np.random.uniform(-50, 50)
                    offset_y = np.random.uniform(-50, 50)
                    self.food.append([center[0] + offset_x, center[1] + offset_y])

        elif pattern_type == "square":
            center_x, center_y = self.dimx / 2, self.dimy / 2
            side_length = min(self.dimx, self.dimy) / 2
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
                self.food.append([food_x, food_y])

        elif pattern_type == "circle":
            center_x, center_y = self.dimx / 2, self.dimy / 2
            radius = min(self.dimx, self.dimy) / 4
            for i in range(num_food):
                angle = 2 * np.pi * i / num_food
                food_x = center_x + radius * np.cos(angle)
                food_y = center_y + radius * np.sin(angle)
                self.food.append([food_x, food_y])
                
        elif pattern_type == "triangle":
            top_vertex = (self.dimx / 2, self.dimy / 4)
            left_vertex = (self.dimx / 4, self.dimy * 3 / 4)
            right_vertex = (self.dimx * 3 / 4, self.dimy * 3 / 4)
            vertices = [top_vertex, left_vertex, right_vertex]
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
                self.food.append([x, y])

    def generate_food_pattern_old(self):
        self.food.append([1000, 600])
        self.food.append([900, 1000])
        self.food.append([925, 1000])
        self.food.append([950, 1000])
    
    def reset(self, pattern_type, num_food=40):
        self.worms = [Worm(position=[self.dimx/2, self.dimy/2],range=self.range) for _ in range(self.num_worms)]
        self.food = []
        #self.generate_circle_of_food()
        #self.generate_food_pattern()
        #self.generate_random_food(40)
        #print(pattern_type)
        self.generate_food_pattern(pattern_type, num_food)
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
                    #if food == [1000, 600]:

                        
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
                wall_reward += 30
                
        vision_radius = self.range  # The maximum distance for food reward gradient

        # Calculate the reward based on the distance to each food item
        for food in self.food:
            distance_to_food = np.linalg.norm(np.array(worm.position) - np.array(food))
            if distance_to_food < vision_radius:
                wall_reward += max(0, (vision_radius - distance_to_food) / vision_radius) / 30
        
        #print(wall_reward)
        return wall_reward

    def _check_done(self):
        return len(self.food) == 0

    def close(self):
        plt.close()
