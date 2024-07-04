import numpy as np

class Worm:
    def __init__(self, position, facing_dir=0, speed=20,range=200):
        self.position = np.array(position, dtype=np.float64)
        self.facing_dir = facing_dir
        self.speed = speed
        self.sees_food = False
        self.range = range

    def update(self, left_speed, right_speed, food_positions):
        self.move(left_speed, right_speed)
        self.sees_food = any(self.is_food_close(f) for f in food_positions)


    def move(self, left_speed, right_speed):
        # Calculate the new position and orientation based on the motor speeds
        wheel_base = 10.0  # distance between the two wheels

        # Calculate the linear and angular velocities
        linear_velocity = (left_speed + right_speed) / 2.0
        angular_velocity = (right_speed - left_speed) / wheel_base

        # Calculate the new speed based on the accumulated speeds
        new_speed = abs(left_speed) + abs(right_speed)
        
        linear_velocity = min(max(new_speed,75),150)/7

        # Update the facing direction
        self.facing_dir += angular_velocity

        # Ensure facing direction is within the range -π to π
        self.facing_dir = (self.facing_dir + np.pi) % (2 * np.pi) - np.pi
        self.position[0] += linear_velocity * np.cos(self.facing_dir)
        self.position[1] += linear_velocity * np.sin(self.facing_dir)

        # Ensure the position is within the specified bounds
        if self.position[0] < 0:
            self.position[0] = 0
        elif self.position[0] > 1600:
            self.position[0] = 1600

        if self.position[1] < 0:
            self.position[1] = 0
        elif self.position[1] > 1200:
            self.position[1] = 1200


    def is_food_close(self, food):
        distance_to_food = np.linalg.norm(np.array(self.position) - np.array(food))
        if distance_to_food > self.range:
                return False
        else:
            return True