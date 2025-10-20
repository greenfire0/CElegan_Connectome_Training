import numpy as np
import ray
from Worm_Env.connectome2 import WormConnectome
from Worm_Env.weight_dict import muscles, muscleList, mLeft, mRight, all_neuron_names
from Algorithms.algo_utils import evaluate_fitness_ray  # (not used here; we keep your internal remote)
from matplotlib import pyplot as plt
from util.write_read_txt import read_arrays_from_csv_pandas
import random
import matplotlib.cm as cm
import os
from Worm_Env.weight_dict import dict as dict2
from matplotlib.cm import ScalarMappable


class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[4], total_episodes=10, training_interval=25, genome=None, matrix_shape=3689):
        # keep API: we’ll still accept population_size but follow your current behavior of using 1
        self.population_size = 1
        print(pattern)
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = []

    def initialize_population(self, csv_list, folder: str):
        base_dir = os.path.dirname(__file__)
        full_folder = os.path.join(base_dir, folder)

        # build once to keep side effects identical to your version
        _ = []
        for sub_dict in dict2.values():
            _.extend(sub_dict.values())

        # "Before training"
        arr = read_arrays_from_csv_pandas(os.path.join(full_folder, "Evolutionary_algorithm.csv"))
        self.population.append(WormConnectome(np.array(arr[0], dtype=float), all_neuron_names))

        # Trained variants
        for csv_path in csv_list:
            arr = read_arrays_from_csv_pandas(os.path.join(full_folder, csv_path))
            self.population.append(WormConnectome(np.array(arr[-1], dtype=float), all_neuron_names))

    @staticmethod
    @ray.remote
    def evaluate_fitness_ray(candidate_weights, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes):
        """
        Keep the exact signature + behavior. For ChemotaxisPeakEnv the reward is a scalar float each step,
        so sum_rewards becomes a 'chemotaxis score'. We also return the trajectory for plotting.
        """
        sum_rewards = 0.0
        trajectory = []
        candidate = WormConnectome(weight_matrix=candidate_weights, all_neuron_names=nur_name)

        for a in prob_type:
            env.reset(a)  # chemotaxis env ignores pattern_type but keeps signature
            for _ in range(episodes):
                observation = env._get_observations()
                for _ in range(interval):
                    movement = candidate.move(
                        observation[0][0],  # min distance to wall
                        env.worms[0].sees_food,
                        mLeft, mRight, muscleList, muscles
                    )
                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    trajectory.append([observation[0][1], observation[0][2]])
                    observation = next_observation
                    sum_rewards += float(reward)
        return float(sum_rewards), trajectory

    def generate_random_color(self):
        return "#%06x" % random.randint(0, 0xFFFFFF)

    def run(
        self,
        env,
        gen,
        csv_files=[
            "Evolutionary_algorithm.csv",
            "ES_worms.csv",
            "Random_50_nomad.csv",
            "Hybrid_nomad.csv",
            "pure_nomad.csv",
        ],
        batch_size=10,
        jitter_strength=10,
    ):
        """
        Same 6-panel layout; now visualizes the chemotaxis peak + gradient background
        if the env exposes `env.food` (peak center) and `env.sigma`.
        """
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        axs = axs.flatten()

        # Build population like before
        self.initialize_population(csv_list=csv_files, folder="data_chemotaxis")

        # Evaluate all candidates (parallel)
        results = ray.get([
            self.evaluate_fitness_ray.remote(
                worm.weight_matrix,
                all_neuron_names,
                env,
                self.food_patterns,
                mLeft,
                mRight,
                muscleList,
                muscles,
                self.training_interval,
                self.total_episodes,
            )
            for worm in self.population
        ])

        titles = [
            "Before Training",
            "Evolutionary Algorithm",
            "OPENAI_ES Algorithm",
            "cfNOMAD",
            "mENOMAD",
            "rENOMAD",
        ]

        # helper: draw a faint chemo field background if attributes exist
        def _draw_chemo_field(ax_local, env_local):
            try:
                if not hasattr(env_local, "sigma") or env_local.food.size == 0:
                    return
                # coarse grid for speed
                nx, ny = 80, 60
                X = np.linspace(0, 1600, nx)
                Y = np.linspace(0, 1200, ny)
                XX, YY = np.meshgrid(X, Y)
                px, py = env_local.food[0]  # center of the Gaussian
                sigma = float(getattr(env_local, "sigma", 220.0))
                C = np.exp(-((XX - px) ** 2 + (YY - py) ** 2) / (2.0 * sigma * sigma))
                ax_local.imshow(
                    C,
                    extent=[0, 1600, 0, 1200],
                    origin="lower",
                    cmap="Greens",
                    alpha=0.25,
                    interpolation="bilinear",
                )
            except Exception:
                # if anything goes wrong, silently skip (keeps API/flow intact)
                pass

        total_steps = int(self.total_episodes * self.training_interval)

        for idx, (fitness, traj) in enumerate(results):
            if idx >= 6:
                break

            ax = axs[idx]
            ax.set_title(titles[idx], fontsize=28)

            # y labels left column only
            if idx % 3 == 0:
                ax.set_ylabel("Y Position", fontsize=26)
                ax.tick_params(axis="y", labelsize=26)
            else:
                ax.set_yticks([])

            # x labels bottom row only
            if idx // 3 == 1:
                ax.set_xlabel("X Position", fontsize=26)
                ax.tick_params(axis="x", labelsize=26)
            else:
                ax.set_xticks([])

            ax.set_xlim(0, 1600)
            ax.set_ylim(0, 1200)

            # Reset once to fetch current peak location; keeps the API the same
            env.reset(self.food_patterns[0])

            # draw chemotaxis field (if available)
            _draw_chemo_field(ax, env)

            # draw the current peak position (one red dot)
            if hasattr(env, "food") and getattr(env, "food", np.zeros((0, 2))).size:
                fx, fy = env.food[0]
                ax.plot([fx], [fy], "ro", markersize=6)

            # trajectory (time-colored)
            traj = np.asarray(traj)
            if len(traj) > 1:
                colors = cm.viridis(np.linspace(0, 1, len(traj)))
                for j in range(1, len(traj)):
                    ax.plot(traj[j - 1:j + 1, 0], traj[j - 1:j + 1, 1], color=colors[j], lw=4)

            # chemotaxis metric label (sum of per-step rewards)
            ax.text(
                0.03,
                0.12,
                f"Chemotaxis Score: {fitness:.1f}",
                transform=ax.transAxes,
                fontsize=26,
                va="top",
                ha="left",
                bbox=dict(facecolor="wheat", alpha=0.5, boxstyle="round,pad=0.2"),
            )

        # time colorbar (0..total_steps)
        sm = ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=total_steps))
        sm.set_array([])

        fig.tight_layout(rect=[0, 0, 1, 0.90])

        cbar_ax = fig.add_axes([0.20, 0.94, 0.60, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=20, length=0)
        cbar_ax.set_title("Time", fontsize=28, pad=10, fontweight="bold")

        fig.savefig("fig_pos_over_time_chemotaxis.svg", dpi=300)
        plt.show()
        ray.shutdown()
