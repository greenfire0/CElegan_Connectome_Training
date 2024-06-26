import numpy as np

class Worm:
    def __init__(self, position, facing_dir=0, speed=20):
        self.position = np.array(position, dtype=np.float64)
        self.facing_dir = facing_dir
        self.speed = speed
        self.sees_food = False

    def update(self, left_speed, right_speed, food_positions):
        self.move(left_speed, right_speed)
        self.sees_food = any(self._is_food_in_vision(f) for f in food_positions)


    def move(self, left_speed, right_speed):
        # Calculate the new position and orientation based on the motor speeds
        wheel_base = 10.0  # distance between the two wheels

        # Calculate the linear and angular velocities
        linear_velocity = (left_speed + right_speed) / 2.0
        angular_velocity = (right_speed - left_speed) / wheel_base

        # Calculate the new speed based on the accumulated speeds
        new_speed = abs(left_speed) + abs(right_speed)
        
        linear_velocity = min(max(new_speed,75),150)/15

        # Update the facing direction
        self.facing_dir += angular_velocity

        # Ensure facing direction is within the range -π to π
        self.facing_dir = (self.facing_dir + np.pi) % (2 * np.pi) - np.pi



        # Update the position based on the adjusted linear velocity
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


    def _is_food_in_vision(self, food):
        vision_radius = 500  # Radius of the vision cone
        vision_angle = np.pi / 4  # Angle of the vision cone

        # Define the vectors representing the edges of the vision cone
        v1 = np.array([np.cos(self.facing_dir - vision_angle), np.sin(self.facing_dir - vision_angle)])
        v2 = np.array([np.cos(self.facing_dir + vision_angle), np.sin(self.facing_dir + vision_angle)])

        # Vector from the worm to the food
        v_food = np.array([food[0] - self.position[0], food[1] - self.position[1]])
        
        # Check if the food is within the vision radius
        distance_to_food = np.linalg.norm(v_food)
        if distance_to_food > vision_radius:
            return False
        
        # Normalize the vector to food to get the direction
        v_food_normalized = v_food / distance_to_food
        
        # Check if the food is within the vision cone
        is_within_cone = np.dot(v1, v_food_normalized) >= 0 and np.dot(v2, v_food_normalized) >= 0
        return is_within_cone