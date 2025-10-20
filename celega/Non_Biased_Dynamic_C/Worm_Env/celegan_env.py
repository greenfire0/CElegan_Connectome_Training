import matplotlib.pyplot as plt
import gymnasium as gym
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
            elif distance_to_food < vision_radius:
                reward += max(0.0, (vision_radius - distance_to_food) / vision_radius) 
        return reward
    
    @staticmethod
    @njit
    def calculate_rewards_new(worm_pos, food_positions, foodradius, vision_radius): ## the /30 might need to be removed
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
        
        rewards = WormSimulationEnv.calculate_rewards2(worm_pos, self.food, self.foodradius, self.range)
        self._check_eat_food(worm_pos)
        done = self._check_done()

        return observations, rewards, done

    def _check_eat_food(self, worm_pos):
        # Compute distances for all food positions at once.
        distances = np.linalg.norm(self.food - worm_pos, axis=1)
        # Keep only food items that are not eaten.
        self.food = self.food[distances >= self.foodradius]


    def render(self, worm_num=0, mode="human"):
        self.ax.clear()
        worm = self.worms[worm_num]

        # worm body + heading
        self.ax.plot(*worm.position, "ro")
        self.ax.plot(
            [worm.position[0], worm.position[0] + 100 * np.cos(worm.facing_dir)],
            [worm.position[1], worm.position[1] + 100 * np.sin(worm.facing_dir)],
            "b-",
        )

        # vectorised proximity check (no Numba)
        if self.food.size:
            dists = np.linalg.norm(self.food - worm.position, axis=1)
            close = dists < self.range
            self.ax.plot(*self.food[~close].T, "bo")
            self.ax.plot(*self.food[close].T,  "yo")

        self.ax.set_xlim(0, self.dimx)
        self.ax.set_ylim(0, self.dimy)
        self.ax.set_aspect("equal", adjustable="box")
        plt.pause(0.001)     # small delay is enough


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

class ChemotaxisPeakEnv(gym.Env):
    """
    API-compatible with WormSimulationEnv:
      - reset(pattern_type, num_food=...)
      - step(actions, worm_num, candidate) -> (obs, reward: float, done: bool)
      - render(worm_num=0, mode="human")
      - close()
      - lasso_reg(candidate_weights, original, lambda_=0.1)  [@staticmethod]
    
    Key differences in task design:
      - No 'consumption' or removal of targets. We model a continuous chemo field
        (Gaussian peak). Reward is dense each step:
           reward = k_step * dC + k_abs * C_now - k_wall * wall_proximity
        where C_now is concentration at current position and dC is the one-step
        improvement (ascent). This tests gradient following (and optionally tracking)
        rather than discrete pickup.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 num_worms=1,
                 sigma=220.0,         # width of Gaussian (controls gradient steepness)
                 k_step=1.0,          # weight for ascent (delta concentration)
                 k_abs=0.1,           # weight for being near the peak (absolute concentration)
                 k_wall=0.002,        # penalty weight near walls
                 drift_std=0.0        # std of per-step source drift (0 = static peak)
                 ):
        super(ChemotaxisPeakEnv, self).__init__()
        # arena geometry (match your existing env)
        self.dimx = 1600
        self.dimy = 1200
        self.num_worms = int(num_worms)

        # keep these names so your code/tools still work if referenced
        self.foodradius = 20      # not used for "consumption", kept for compatibility
        self.range = 150          # also used in wall proximity shaping
        self.wall_margin = 100

        # gradient + reward config
        self.sigma = float(sigma)
        self.k_step = float(k_step)
        self.k_abs = float(k_abs)
        self.k_wall = float(k_wall)
        self.drift_std = float(drift_std)

        # state
        self.worms = []
        self.food = np.zeros((0, 2), dtype=float)  # we still call it "food" for API consistency
        self.prev_C = None

        # plotting (kept the same pattern as your env)
        self.fig, self.ax = plt.subplots()

    # ---------- Public API (unchanged signatures) ----------

    def reset(self, pattern_type=None, num_food=1):
        """
        Keep signature identical; pattern_type/num_food are accepted but ignored.
        Spawns a single chemo source (peak) and centers the worm(s).
        """
        self.worms = [Worm(position=[self.dimx / 2, self.dimy / 2], range=self.range)
                      for _ in range(self.num_worms)]
        self.food = np.array([self._sample_peak()], dtype=float)  # shape (1, 2)

        # initialize previous concentration per worm for the ascent term
        self.prev_C = np.zeros(self.num_worms, dtype=float)
        for i, w in enumerate(self.worms):
            self.prev_C[i] = self._concentration(np.asarray(w.position, dtype=float), self.food[0])

        return self._get_observations()

    def step(self, actions, worm_num, candidate):
        """
        Same signature as WormSimulationEnv.step.
        Returns: (observations, reward: float, done: bool)
        """
        left_speed, right_speed = actions

        # Pass the current peak as "food_positions" so the worm's internal sensory
        # logic (e.g., sees_food) continues to work without any code changes.
        self.worms[worm_num].update(
            left_speed=left_speed,
            right_speed=right_speed,
            food_positions=self.food
        )

        # Optional: drift the source slightly each step ⇒ tracking behavior required
        if self.drift_std > 0.0:
            self.food[0] += np.random.normal(scale=self.drift_std, size=2)
            self.food[0, 0] = float(np.clip(self.food[0, 0], 0.0, self.dimx))
            self.food[0, 1] = float(np.clip(self.food[0, 1], 0.0, self.dimy))

        obs = self._get_observations()

        # Dense reward: climb the gradient and remain near the peak; avoid walls
        worm_pos = np.asarray(self.worms[worm_num].position, dtype=float)
        C_now = self._concentration(worm_pos, self.food[0])
        dC = C_now - self.prev_C[worm_num]
        self.prev_C[worm_num] = C_now

        # wall proximity penalty in [0, 1]: 0 far, 1 at/beyond margin
        mdw = min(worm_pos[0],
                  self.dimx - worm_pos[0],
                  worm_pos[1],
                  self.dimy - worm_pos[1])
        wall_prox = max(0.0, (self.range - mdw) / self.range)

        reward = float(self.k_step * dC + self.k_abs * C_now - self.k_wall * wall_prox)

        # This task is continuous; let your outer loop set the horizon (e.g., 250 steps)
        done = False
        return obs, reward, done

    def render(self, worm_num=0, mode="human"):
        self.ax.clear()
        worm = self.worms[worm_num]

        # worm body + heading
        self.ax.plot(*worm.position, "ro")
        self.ax.plot(
            [worm.position[0], worm.position[0] + 100 * np.cos(worm.facing_dir)],
            [worm.position[1], worm.position[1] + 100 * np.sin(worm.facing_dir)],
            "b-",
        )

        # draw the chemo peak position (red if far, yellow if within 'range')
        if self.food.size:
            d = np.linalg.norm(self.food - worm.position, axis=1)
            close = d < self.range
            if np.any(~close):
                self.ax.plot(*self.food[~close].T, "ro")
            if np.any(close):
                self.ax.plot(*self.food[close].T, "yo")

        self.ax.set_xlim(0, self.dimx)
        self.ax.set_ylim(0, self.dimy)
        self.ax.set_aspect("equal", adjustable="box")
        plt.pause(0.001)

    def close(self):
        plt.close()

    # ---------- Helpers (internal) ----------

    def _get_observations(self):
        """
        Keep exact structure as your original env:
        [min_distance_to_wall, x, y, facing_dir, sees_food]
        """
        observations = []
        for worm in self.worms:
            mdw = min(
                worm.position[0],
                self.dimx - worm.position[0],
                worm.position[1],
                self.dimy - worm.position[1],
            )
            observations.append(np.array([
                float(mdw),
                float(worm.position[0]),
                float(worm.position[1]),
                float(worm.facing_dir),
                float(worm.sees_food),
            ], dtype=float))
        return np.array(observations, dtype=float)

    def _sample_peak(self):
        """
        Deterministic spawn in the top-right corner, kept away from walls
        by the same safety margin used elsewhere.
        """
        margin = max(self.range, self.wall_margin) + 20.0
        x = self.dimx - margin
        y = self.dimy - margin
        return np.array([x, y], dtype=float)


    def _concentration(self, pos_xy, src_xy):
        """
        Gaussian chemo field centered at src_xy with width sigma.
        Returns a scalar in (0, 1].
        """
        dx = pos_xy[0] - src_xy[0]
        dy = pos_xy[1] - src_xy[1]
        d2 = dx * dx + dy * dy
        return float(np.exp(-d2 / (2.0 * self.sigma * self.sigma)))

    # ---------- Static penalty (unchanged signature) ----------

    @staticmethod
    @njit
    def lasso_reg(candidate_weights, original, lambda_=0.1):
        """
        Same as in your WormSimulationEnv: -lambda * |W != W0|^1.3
        Returns a plain Python float (Numba-compatible).
        """
        num_differences = np.count_nonzero(candidate_weights != original)
        penalty = -lambda_ * np.power(num_differences, 1.3)
        return penalty