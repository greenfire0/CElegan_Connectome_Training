import numpy as np
from numba import njit, prange

@njit
def move(position, facing_dir, left_speed, right_speed):
    wheel_base = 10.0  # distance between the two wheels

    # Calculate the linear and angular velocities
    linear_velocity = (left_speed + right_speed) / 2.0
    angular_velocity = (right_speed - left_speed) / wheel_base

    # Calculate the new speed based on the accumulated speeds
    new_speed = abs(left_speed) + abs(right_speed)
    
    linear_velocity = min(max(new_speed, 75), 150) / 7

    # Update the facing direction
    facing_dir += angular_velocity

    # Ensure facing direction is within the range -π to π
    facing_dir = (facing_dir + np.pi) % (2 * np.pi) - np.pi
    position[0] += linear_velocity * np.cos(facing_dir)
    position[1] += linear_velocity * np.sin(facing_dir)

    # Ensure the position is within the specified bounds
    if position[0] < 0:
        position[0] = 0
    elif position[0] > 1600:
        position[0] = 1600

    if position[1] < 0:
        position[1] = 0
    elif position[1] > 1200:
        position[1] = 1200
    
    return position, facing_dir

@njit
def is_food_close(position, food, range):
    distance_to_food = np.linalg.norm(position - food)
    return distance_to_food <= range

@njit
def update(position, facing_dir, left_speed, right_speed, food_positions, range):
    position, facing_dir = move(position, facing_dir, left_speed, right_speed)
    sees_food = False
    for i in prange(len(food_positions)):
        if is_food_close(position, food_positions[i], range):
            sees_food = True
            break
    return position, facing_dir, sees_food

class Worm:
    def __init__(self, position, facing_dir=0, speed=20, range=200):
        self.position = np.array(position, dtype=np.float64)
        self.facing_dir = facing_dir
        self.speed = speed
        self.sees_food = False
        self.range = range

    def update(self, left_speed, right_speed, food_positions):
        food_positions = np.array(food_positions, dtype=np.float64)  # Ensure food_positions is a NumPy array
        self.position, self.facing_dir, self.sees_food = update(
            self.position, self.facing_dir, left_speed, right_speed, food_positions, self.range
        )

