import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import  muscles, muscleList, mLeft, mRight, all_neuron_names
from Algorithms.algo_utils import evaluate_fitness_ray
from matplotlib import pyplot as plt
from util.write_read_txt import read_arrays_from_csv_pandas
import random
import matplotlib.cm as cm  # Import for color mapping
import os
from Worm_Env.weight_dict import dict as dict2
from matplotlib.cm import ScalarMappable          #  ADD THIS

class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[4], total_episodes=10, training_interval=25, genome=None, matrix_shape=3689):
        self.population_size = 1 #population_size
        print(pattern)
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = []

    def initialize_population(self, csv_list,folder:str):  
        base_dir = os.path.dirname(__file__)
        full_folder = os.path.join(base_dir, folder)
        values_list = []
        for sub_dict in dict2.values():
            values_list.extend(sub_dict.values())
 
        self.population.append(WormConnectome(np.array(values_list,  dtype=float), all_neuron_names))
        for csv_path in csv_list:
                arr = read_arrays_from_csv_pandas(os.path.join(full_folder, csv_path))
                    # after/best (last row)
                self.population.append(
                        WormConnectome(np.array(arr[-1], dtype=float), all_neuron_names)
                    )



    @staticmethod
    @ray.remote
    def evaluate_fitness_ray(candidate_weights, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes):
        sum_rewards = 0
        trajectory = []  # To store worm positions over time
        candidate = WormConnectome(weight_matrix=candidate_weights, all_neuron_names=nur_name)
        
        for a in prob_type:
            env.reset(a)
            for _ in range(episodes):  # total_episodes
                observation = env._get_observations()
                for _ in range(interval):  # training_interval
                    movement = candidate.move(observation[0][0], env.worms[0].sees_food, mLeft, mRight, muscleList, muscles)

                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    trajectory.append([observation[0][1],observation[0][2]])  # Capture worm's position
                    observation = next_observation
                    sum_rewards += reward
        return sum_rewards, trajectory


    def generate_random_color(self):
        return '#%06x' % random.randint(0, 0xFFFFFF)

    def run(
        self,
        env,
        gen,
        csv_files=[
            "Evolutionary_algorithm.csv",
            "ES_worms.csv",
            "Random_50_nomad.csv",

            "Hybrid_nomad.csv",
            "Pure_nomad.csv",
        ],
        batch_size=10,
        jitter_strength=10,
    ):
        """
        Draw 6 panels (2 rows × 3 cols)

            ┌───────────────┬───────────────┬───────────────┐
            │ Evo – before  │ Evo – after   │ OPENAI-ES     │
            ├───────────────┼───────────────┼───────────────┤
            │ Hybrid NOMAD  │ Pure NOMAD    │ Random NOMAD  │
            └───────────────┴───────────────┴───────────────┘
        """

        # ── 0) create figure & axes grid ───────────────────────────── #
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.flatten()        # easier indexing

        # ── 1) build population (your original logic) ─────────────── #
        self.initialize_population(csv_list=csv_files, folder="data_full_pentagon")

        # ── 2) run fitness evaluations in parallel ────────────────── #
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

        # ── 3) panel → title mapping  (6 panels) ──────────────────── #
        titles = [
            "Before Training",
            "Evolutionary Algorithm",
            "OPENAI_ES Algorithm",
            "Random NOMAD",
            "Hybrid NOMAD",
            "Pure NOMAD",
            
        ]

        # ── 4) draw each trajectory  ──────────────────────────────── #
        for idx, (fitness, traj) in enumerate(results):
            if idx >= 6:          # ignore any extra worms
                break

            ax = axs[idx]
            ax.set_title(titles[idx], fontsize=24)

            # only left-most column shows Y-ticks
            if idx % 3 == 0:
                ax.set_ylabel("Y Position",fontsize=22)
                ax.tick_params(axis="y", labelsize=22)
            else:
                ax.set_yticks([])

            # only bottom row shows X-ticks
            if idx // 3 == 1:
                ax.set_xlabel("X Position",fontsize=22)
                ax.tick_params(axis="x", labelsize=22)
            else:
                ax.set_xticks([])

            ax.set_xlim(0, 1600)
            ax.set_ylim(0, 1200)

            # food pattern
            env.reset(self.food_patterns[0])
            ax.plot(*zip(*env.food), "ro")

            # trajectory
            traj = np.asarray(traj)
            if len(traj) > 1:
                colors = cm.viridis(np.linspace(0, 1, len(traj)))
                for j in range(1, len(traj)):
                    ax.plot(
                        traj[j-1:j+1, 0],
                        traj[j-1:j+1, 1],
                        color=colors[j],
                        lw=4,
                    )

            # fitness box
            ax.text(
                0.03, 0.12, f"Food Sources Eaten: {fitness:.1f}",
                transform=ax.transAxes,
                fontsize=22,
                va="top", ha="left",
                bbox=dict(facecolor="wheat", alpha=0.5, boxstyle="round,pad=0.2"),
            )
        sm = ScalarMappable(
            cmap=cm.viridis,
            norm=plt.Normalize(vmin=0, vmax=250)   # <-- new
        )        
        sm.set_array([])

        # reserve 10 % of the figure’s height at the top:
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        # make the rows sit a bit closer together
        # overall title
        fig.suptitle(
            "Worm Movement Trajectories Before and After Training",
            fontsize=28, y=0.98                     # y just below the top edge
        )

        # add an axes for the colour-bar (left, bottom, width, height)
        cbar_ax = fig.add_axes([0.20, 0.92, 0.60, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=14, length=0)

        # ── 5) save/show ────────────────────────────────────────────── #
        fig.savefig("fig_pos_over_time.svg")
        plt.show()
        ray.shutdown()