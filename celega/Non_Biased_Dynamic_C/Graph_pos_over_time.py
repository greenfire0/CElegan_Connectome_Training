import numpy as np
import ray
from Worm_Env.trained_connectome import WormConnectome
from graphing import graph
from Worm_Env.weight_dict import  muscles, muscleList, mLeft, mRight, all_neuron_names
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
import os
from util.write_read_txt import read_arrays_from_csv_pandas
import random
import matplotlib.cm as cm  # Import for color mapping

class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[5], total_episodes=10, training_interval=25, genome=None, matrix_shape=3689):
        self.population_size = 1 #population_size
        print(pattern)
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = genome
        self.food_patterns = pattern
        self.population = []

    def initialize_population(self, genomes=None,genome2=None,genome3=None):    
        self.population.append(WormConnectome(weight_matrix=np.array(genomes[0], dtype=float), all_neuron_names=all_neuron_names))
    
        self.population.append(WormConnectome(weight_matrix=np.array(genomes[-1], dtype=float), all_neuron_names=all_neuron_names))
        self.population.append(WormConnectome(weight_matrix=np.array(genome2[-1], dtype=float), all_neuron_names=all_neuron_names))
        self.population.append(WormConnectome(weight_matrix=np.array(genome3[-1], dtype=float), all_neuron_names=all_neuron_names))



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

    def run(self, env, gen, csv_files=["tri_evo5.csv", "15_tri_nomad.csv800.csv", "250-tri-NO_gen52-61.csv"], batch_size=10, jitter_strength=10):
        """
        Generates a 2x2 grid of plots:
        - First row: Two plots from "tri_evo5.csv" (before and after training)
        - Second row: 
            - First column: "15_tri_nomad.csv800.csv" (NOMAD hybrid)
            - Second column: "250-tri-NO_gen52-61.csv" (pure NOMAD)
        """
        # Initialize Ray
        ray.init(
            ignore_reinit_error=True,
            object_store_memory=14 * 1024 * 1024 * 1024,
            num_cpus=16,
        )

        # Create a figure with 2 rows and 2 columns
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))  # Increased figure size for clarity

        # Flatten axs for easy indexing
        axs = axs.flatten()

        # Define CSV files for each subplot
        # Ensure the list has exactly three CSV files as per user request
        if len(csv_files) != 3:
            raise ValueError("Please provide exactly three CSV files.")

        # Initialize population with four connectomes: two from tri_evo5.csv, one from tri_nomad800.csv, one from pure NOMAD
        genomes_evo = read_arrays_from_csv_pandas(csv_files[0])  # tri_evo5.csv
        genomes_nomad_hybrid = read_arrays_from_csv_pandas(csv_files[1])  # tri_nomad800.csv
        genomes_pure_nomad = read_arrays_from_csv_pandas(csv_files[2])  # 250-tri-NO_gen52-61.csv

        self.initialize_population(genomes_evo, genomes_nomad_hybrid, genomes_pure_nomad)

        # Evaluate fitness and get trajectories
        results = ray.get([
            self.evaluate_fitness_ray.remote(
                candidate.weight_matrix, 
                all_neuron_names, 
                env, 
                self.food_patterns, 
                mLeft, 
                mRight, 
                muscleList, 
                muscles, 
                self.training_interval, 
                self.total_episodes
            ) for worm_num, candidate in enumerate(self.population)
        ])

        # Define mapping of plots to subplots
        plot_mappings = {
            0: {"title": "Evolutionary Algorithm - Before Training", "csv": "tri_evo5.csv"},
            1: {"title": "Evolutionary Algorithm", "csv": "tri_evo5.csv"},  # Removed " - After Training"
            2: {"title": "NOMAD Hybrid Algorithm", "csv": "15_tri_nomad.csv800.csv"},
            3: {"title": "Pure NOMAD Algorithm", "csv": "250-tri-NO_gen52-61.csv"}
        }

        for worm_num, (fitness, trajectory) in enumerate(results):
            if worm_num not in plot_mappings:
                continue  # Skip if not mapped

            plot_info = plot_mappings[worm_num]
            ax = axs[worm_num]

            # Set plot title
            ax.set_title(plot_info["title"], fontsize=18, pad=20)  # Added padding to prevent overlap

            # Set labels only for specific plots to avoid duplication
            if worm_num % 2 == 0:
                ax.set_ylabel('Y Position', fontsize=14)
            else:
                ax.set_ylabel('')
            
            if worm_num >= 2:
                ax.set_xlabel('X Position', fontsize=14)
            else:
                ax.set_xlabel('')

            # Remove ticks for plots without labels
            if worm_num % 2 != 0:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            if worm_num < 2:
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            # Set axis limits and aspect
            ax.set_xlim(0, 1600)
            ax.set_ylim(0, 1200)
            ax.set_aspect('equal')

            # Plot food locations
            env.reset(5)  # Assuming '5' corresponds to a specific food pattern for all plots
            for f in env.food:
                ax.plot(f[0], f[1], 'ro')  # Red 'ro' markers for food

            # Convert trajectory to NumPy array for easy manipulation
            trajectory = np.array(trajectory)
            num_points = len(trajectory)

            if num_points < 2:
                print(f"Worm {worm_num} in {plot_info.get('csv')} has insufficient trajectory data.")
                continue

            # Choose color map based on worm number to differentiate trajectories
            cmap = cm.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, num_points))

            # Plot the trajectory
            for i in range(1, num_points):
                x_values = [trajectory[i-1][0], trajectory[i][0]]  # x coordinates of the segment
                y_values = [trajectory[i-1][1], trajectory[i][1]]  # y coordinates of the segment

                # Plot the line segment connecting these points with color based on progression
                ax.plot(x_values, y_values, color=colors[i], linewidth=5, alpha=0.8)

            # Display fitness in the subplot
            ax.text(
                0.05, 0.95, f'Food Eaten: {fitness}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                ha='left', va='top'
            )

        # Adjust layout and add a super title
        plt.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.95])  # Increased padding to prevent overlap
        plt.subplots_adjust(top=0.92)  # Further adjust top to accommodate super title
        plt.suptitle('Worm Movement Trajectories Before and After Training', fontsize=22, y=0.98)  # Adjusted y-position

        # Save the plot to a file
        plt.savefig("figofpos1.png")
        plt.show()  # Show the plot (optional)

        # Shutdown Ray
        ray.shutdown()