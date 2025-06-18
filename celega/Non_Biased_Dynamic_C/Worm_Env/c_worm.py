import numpy as np
from numba import njit

@njit
def move(position, facing_dir, left_speed, right_speed):
    lv = min(max(abs(left_speed) + abs(right_speed), 75), 150) / 7.0
    facing_dir = (facing_dir + (right_speed - left_speed) / 10.0 + np.pi) % (2 * np.pi) - np.pi

    position[0] = max(0.0, min(1600.0, position[0] + lv * np.cos(facing_dir)))
    position[1] = max(0.0, min(1200.0, position[1] + lv * np.sin(facing_dir)))

    return position, facing_dir

@njit
def is_food_close(position, food_positions, range):
    distances = np.sqrt(np.sum((food_positions - position) ** 2, axis=1))
    return np.any(distances <= range)

@njit
def update(position, facing_dir, left_speed, right_speed, food_positions, range):
    position, facing_dir = move(position, facing_dir, left_speed, right_speed)
    sees_food = is_food_close(position, food_positions, range)
    return position, facing_dir, sees_food

class Worm:
    def __init__(self, position, facing_dir=0, speed=20, range=200):
        self.position = np.array(position, dtype=np.float32)
        self.facing_dir = facing_dir
        self.speed = speed
        self.sees_food = False
        self.range = range

    def update(self, left_speed, right_speed, food_positions):
        food_positions = np.array(food_positions, dtype=np.float32)  # Ensure food_positions is a NumPy array
        self.position, self.facing_dir, self.sees_food = update(
            self.position, self.facing_dir, left_speed, right_speed, food_positions, self.range
        )