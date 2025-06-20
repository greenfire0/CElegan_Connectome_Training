import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import muscles, muscleList, mLeft, mRight, all_neuron_names
from matplotlib import pyplot as plt
from util.write_read_txt import read_arrays_from_csv_pandas
import matplotlib.cm as cm  # For color mapping

class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[4], total_episodes=10, training_interval=25, genome=None, matrix_shape=3689):
        """
        Note: population_size is overridden to 1 in the example, 
        but we will store exactly 4 specific connectomes (indexes) from arrays.csv.
        """
        self.population_size = 1
        self.food_patterns = pattern
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.population = []

    def initialize_population_from_quartiles(self, all_genomes):
        """
        Takes a list of all weight matrices from arrays.csv, 
        and picks 4 specific indices: 0, 1/4, 3/4, and last index.
        """
        n = len(all_genomes)
        if n == 0:
            raise ValueError("No genomes found in arrays.csv")

        idx_before = 0
        idx_q1 = n // 4
        idx_q3 = (3 * n) // 4
        idx_final = n - 1

        selected_indices = [idx_before, idx_q1, idx_q3, idx_final]
        for idx in selected_indices:
            self.population.append(
                WormConnectome(weight_matrix=np.array(all_genomes[idx], dtype=float),
                               all_neuron_names=all_neuron_names)
            )

    @staticmethod
    @ray.remote
    def evaluate_fitness_ray(candidate_weights, nur_name, env, prob_type, mLeft, mRight, muscleList, muscles, interval, episodes):
        sum_rewards = 0
        trajectory = []
        candidate = WormConnectome(weight_matrix=candidate_weights, all_neuron_names=nur_name)
        
        # Evaluate on each food pattern in `prob_type`
        for pat in prob_type:
            env.reset(pat)
            for _ in range(episodes):
                observation = env._get_observations()
                for _ in range(interval):
                    movement = candidate.move(observation[0][0], env.worms[0].sees_food, 
                                              mLeft, mRight, muscleList, muscles)
                    next_observation, reward, _ = env.step(movement, 0, candidate)
                    trajectory.append([observation[0][1], observation[0][2]])
                    observation = next_observation
                    sum_rewards += reward
        return sum_rewards, trajectory

    def run_single_csv_quartiles(self, env, arrays_csv="Evo_nomad.csv"):
        """
        - Reads in `arrays.csv` (a single file).
        - Identifies four points in the evolutionary timeline:
            1) First genome
            2) Genome at 1/4 of the file
            3) Genome at 3/4 of the file
            4) Final genome
        - Plots them in a 2×2 grid:
            Top-left: Before Training 
            Top-right: 1st Quartile
            Bottom-left: 3rd Quartile
            Bottom-right: Final Generation
        """

        # Load all genomes from the single CSV
        all_genomes = read_arrays_from_csv_pandas(arrays_csv)
        
        # Create the 4-worm population from quartiles
        self.initialize_population_from_quartiles(all_genomes)

        # Evaluate each worm
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
                self.total_episodes
            )
            for worm in self.population
        ])

        # Create a figure with 2 rows and 2 columns
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))
        axs = axs.flatten()

        # Titles for our 4 subplots
        plot_titles = [
            "Before Training",
            "First Quartile Generation",
            "Third Quartile Generation",
            "Final Generation"
        ]

        # Plot each worm's trajectory
        for i, (fitness, trajectory) in enumerate(results):
            ax = axs[i]
            ax.set_title(plot_titles[i], fontsize=18, pad=20)
            
            # Label logic: only put x-label on bottom row, y-label on left column
            if i % 2 == 0:  # left column
                ax.set_ylabel('Y Position', fontsize=14)
            else:
                ax.set_ylabel('')
            if i >= 2:  # bottom row
                ax.set_xlabel('X Position', fontsize=14)
            else:
                ax.set_xlabel('')
            
            # For cleaner subplots, hide unneeded tick labels
            if i % 2 != 0:  # right column
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            if i < 2:  # top row
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            # Set axis limits
            ax.set_xlim(0, 1600)
            ax.set_ylim(0, 1200)
            ax.set_aspect('equal')

            # Reset env in order to plot the same food layout (optional: pick a single pattern)
            env.reset(self.food_patterns[0] if self.food_patterns else 0)
            for f in env.food:
                ax.plot(f[0], f[1], 'ro')  # Plot food as red dots

            # Convert trajectory to NumPy array for easier slicing
            trajectory = np.array(trajectory)
            if len(trajectory) < 2:
                print(f"Not enough trajectory data for worm {i}.")
                continue
            
            # Make a colormap
            cmap = cm.get_cmap('viridis')
            num_points = len(trajectory)
            colors = cmap(np.linspace(0, 1, num_points))

            # Plot trajectory with gradient color
            for step in range(1, num_points):
                x_vals = [trajectory[step-1][0], trajectory[step][0]]
                y_vals = [trajectory[step-1][1], trajectory[step][1]]
                ax.plot(x_vals, y_vals, color=colors[step], linewidth=5, alpha=0.8)

            # Show the total "reward" or "food eaten"
            ax.text(
                0.05, 0.95, f'Food Eaten: {fitness}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        # Clean up layout
        plt.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.95])
        plt.suptitle("Worm Trajectories at Quartile Checkpoints", fontsize=22, y=0.98)

        # Save and show
        plt.savefig("quartiles_trajectories.png")
        plt.show()

        # Shutdown Ray
        ray.shutdown()

###############################################################################
# Example usage (in whatever main script you have):
#
# env = ...  # Your Worm environment
# gd = Genetic_Dyn_Algorithm(population_size=1, pattern=[4], total_episodes=10, training_interval=25)
# gd.run_single_csv_quartiles(env, arrays_csv="arrays.csv")
###############################################################################
