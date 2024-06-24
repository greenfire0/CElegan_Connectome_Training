import numpy as np

class Worm:
    def __init__(self, position, facing_dir=0, speed=20):
        self.position = np.array(position, dtype=np.float64)
        self.facing_dir = facing_dir
        self.speed = speed
        self.sees_food = False

    def update(self, action, food_positions):
        if action == 0:  # Turn left
            self.facing_dir += np.pi / 18  # 10 degrees
        elif action == 1:  # Turn right
            self.facing_dir -= np.pi / 18
        self.move()
        self.sees_food = any(self._is_food_in_vision(f) for f in food_positions)


    def move(self):
        next_x = self.position[0] + self.speed * np.cos(self.facing_dir)
        next_y = self.position[1] + self.speed * np.sin(self.facing_dir)

        # Adjust the position if the worm hits a wall
        if next_x < 0:
            next_x = 0
        elif next_x > 1600:
            next_x = 1600

        if next_y < 0:
            next_y = 0
        elif next_y > 1200:
            next_y = 1200

        self.position[0] = next_x
        self.position[1] = next_y


    def _is_food_in_vision(self, food):
        vision_radius = 110  # Radius of the vision cone
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