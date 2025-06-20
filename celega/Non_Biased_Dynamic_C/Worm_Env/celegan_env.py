import matplotlib.pyplot as plt
import gym
import numpy as np
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
    def calculate_rewards_new(worm_pos, food_positions, foodradius, vision_radius):
        diff = food_positions - worm_pos
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        reward_food = 30 * np.sum(distances < foodradius)
        vision_mask = distances < vision_radius
        vision_rewards = np.sum(np.maximum(0.0, (vision_radius - distances[vision_mask]) / vision_radius)) / 30.0
        return reward_food + vision_rewards

    @staticmethod
    @njit
    def calculate_rewards2(worm_pos, food_positions, foodradius, vision_radius):
        reward = 0.0
        for f in food_positions:
            if np.linalg.norm(worm_pos - f) < foodradius:
                reward += 1
        return reward
    @staticmethod
    @njit
    def lasso_reg(candidate_weights, original, lambda_=0.1):
        num_differences = np.count_nonzero(candidate_weights != original)
        penalty = -lambda_ * np.power(num_differences, 1.3)
        return penalty

    @staticmethod
    def generate_food_pattern(pattern_type, num_food, dimx, dimy):
        food = []
        center_x = dimx / 2
        center_y = dimy / 2

        if pattern_type == 0:  # Circle
            radius = min(dimx, dimy) / 4
            for i in range(num_food):
                angle = 2 * np.pi * i / num_food
                food_x = center_x + radius * np.cos(angle)
                food_y = center_y + radius * np.sin(angle)
                food.append([food_x, food_y])

        elif pattern_type == 3:  # Triangle
            top_vertex = (dimx / 2, dimy* 3/ 4)
            left_vertex = (dimx / 4, dimy * 1 / 4)
            right_vertex = (dimx * 3 / 4, dimy * 1 / 4)
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
                    
        if pattern_type == 4:  # Square
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

        elif pattern_type == 5:  # Pentagon (5-sided Polygon)
            num_sides = 5
            radius = min(dimx, dimy) / 4
            for i in range(num_food):
                angle = (2 * np.pi * (i % num_sides)) / num_sides
                interp = (i // num_sides) / (num_food // num_sides)

                vertex_x = center_x + radius * np.cos(angle)
                vertex_y = center_y + radius * np.sin(angle)

                next_angle = (2 * np.pi * ((i % num_sides) + 1)) / num_sides
                next_vertex_x = center_x + radius * np.cos(next_angle)
                next_vertex_y = center_y + radius * np.sin(next_angle)

                food_x = vertex_x + interp * (next_vertex_x - vertex_x)
                food_y = vertex_y + interp * (next_vertex_y - vertex_y)
                food.append([food_x, food_y])

        elif pattern_type == 6:  # Hexagon (6-sided Polygon)
            num_sides = 6
            radius = min(dimx, dimy) / 4
            for i in range(num_food):
                angle = (2 * np.pi * (i % num_sides)) / num_sides
                interp = (i // num_sides) / (num_food // num_sides)

                vertex_x = center_x + radius * np.cos(angle)
                vertex_y = center_y + radius * np.sin(angle)

                next_angle = (2 * np.pi * ((i % num_sides) + 1)) / num_sides
                next_vertex_x = center_x + radius * np.cos(next_angle)
                next_vertex_y = center_y + radius * np.sin(next_angle)

                food_x = vertex_x + interp * (next_vertex_x - vertex_x)
                food_y = vertex_y + interp * (next_vertex_y - vertex_y)
                food.append([food_x, food_y])

        elif pattern_type == 7:  # Heptagon (7-sided Polygon)
            num_sides = 7
            radius = min(dimx, dimy) / 4
            for i in range(num_food):
                angle = (2 * np.pi * (i % num_sides)) / num_sides
                interp = (i // num_sides) / (num_food // num_sides)

                vertex_x = center_x + radius * np.cos(angle)
                vertex_y = center_y + radius * np.sin(angle)

                next_angle = (2 * np.pi * ((i % num_sides) + 1)) / num_sides
                next_vertex_x = center_x + radius * np.cos(next_angle)
                next_vertex_y = center_y + radius * np.sin(next_angle)

                food_x = vertex_x + interp * (next_vertex_x - vertex_x)
                food_y = vertex_y + interp * (next_vertex_y - vertex_y)
                food.append([food_x, food_y])

        elif pattern_type == 8:  # Octagon (8-sided Polygon)
            num_sides = 8
            radius = min(dimx, dimy) / 4
            for i in range(num_food):
                angle = (2 * np.pi * (i % num_sides)) / num_sides
                interp = (i // num_sides) / (num_food // num_sides)

                vertex_x = center_x + radius * np.cos(angle)
                vertex_y = center_y + radius * np.sin(angle)

                next_angle = (2 * np.pi * ((i % num_sides) + 1)) / num_sides
                next_vertex_x = center_x + radius * np.cos(next_angle)
                next_vertex_y = center_y + radius * np.sin(next_angle)

                food_x = vertex_x + interp * (next_vertex_x - vertex_x)
                food_y = vertex_y + interp * (next_vertex_y - vertex_y)
                food.append([food_x, food_y])

        elif pattern_type == 9:  # Nonagon (9-sided Polygon)
            num_sides = 9
            radius = min(dimx, dimy) / 4
            for i in range(num_food):
                angle = (2 * np.pi * (i % num_sides)) / num_sides
                interp = (i // num_sides) / (num_food // num_sides)

                vertex_x = center_x + radius * np.cos(angle)
                vertex_y = center_y + radius * np.sin(angle)

                next_angle = (2 * np.pi * ((i % num_sides) + 1)) / num_sides
                next_vertex_x = center_x + radius * np.cos(next_angle)
                next_vertex_y = center_y + radius * np.sin(next_angle)

                food_x = vertex_x + interp * (next_vertex_x - vertex_x)
                food_y = vertex_y + interp * (next_vertex_y - vertex_y)
                food.append([food_x, food_y])
        return np.array(food)

    def reset(self, pattern_type, num_food=36):
        self.worms = [Worm(position=[self.dimx/2, self.dimy/2], range=self.range) for _ in range(self.num_worms)]
        self.food = np.array(WormSimulationEnv.generate_food_pattern(pattern_type, num_food, self.dimx, self.dimy))
        return self._get_observations()

    def step(self, actions, worm_num, candidate):
        left_speed, right_speed = actions
        self.worms[worm_num].update(left_speed=left_speed, right_speed=right_speed, food_positions=self.food)

        observations = self._get_observations()
        
        worm_pos = self.worms[worm_num].position
        
        rewards = WormSimulationEnv.calculate_rewards_new(worm_pos, self.food, self.foodradius, self.range)
        self._check_eat_food(worm_pos)
        done = self._check_done()

        return observations, rewards, done

    def _check_eat_food(self, worm_pos):
        # Compute distances for all food positions at once.
        distances = np.linalg.norm(self.food - worm_pos, axis=1)
        # Keep only food items that are not eaten.
        self.food = self.food[distances >= self.foodradius]


    def render(self, worm_num=0, mode='human'):
        self.ax.clear()
        worm = self.worms[worm_num]
        self.ax.plot(worm.position[0], worm.position[1], 'ro')
        self.ax.plot([worm.position[0], worm.position[0] + 100 * np.cos(worm.facing_dir)],
                     [worm.position[1], worm.position[1] + 100 * np.sin(worm.facing_dir)], 'b-')

        for f in self.food:
            if is_food_close(worm.position,f,self.range):
                self.ax.plot(f[0], f[1], 'yo')
            else:    
                self.ax.plot(f[0], f[1], 'bo')

        self.ax.set_xlim(0, self.dimx)
        self.ax.set_ylim(0, self.dimy)
        plt.pause(0.01)

    def _get_observations(self):
        observations = []
        for worm in self.worms:
            min_distance_to_wall = min( worm.position[0], self.dimx - worm.position[0], worm.position[1], self.dimy - worm.position[1])

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