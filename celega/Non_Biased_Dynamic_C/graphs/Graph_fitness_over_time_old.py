import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import muscles, muscleList, mLeft, mRight, all_neuron_names
from matplotlib import pyplot as plt
import os
from util.write_read_txt import read_arrays_from_csv_pandas
import random
from Algorithms.algo_utils import evaluate_fitness_ray
from collections import defaultdict
class Genetic_Dyn_Algorithm:
    def __init__(self, population_size, pattern=[5], total_episodes=10, training_interval=25, genome=None, matrix_shape=3689):
        self.population_size = population_size
        print(pattern)
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = np.array(genome, dtype=float) if genome is not None else np.zeros(matrix_shape)
        self.food_patterns = pattern
        self.population = []

    def initialize_population(self, genomes=None):
        if genomes is None:
            raise ValueError("Genomes must be provided to initialize the population.")
        if len(genomes) > 400:
            genomes = genomes[:400]  # Limiting to first 400 genomes if necessary
        for g in genomes:
            self.population.append(WormConnectome(weight_matrix=np.array(g, dtype=float), all_neuron_names=all_neuron_names))



    def calculate_euclidean_distance(self, candidate_weights):
        candidate_weights = np.array(candidate_weights, dtype=float)
        if candidate_weights.shape != self.original_genome.shape:
            raise ValueError("Shape of candidate_weights and original_genome must be the same.")
        distance = np.linalg.norm(candidate_weights - self.original_genome)
        return distance


    def count_changes(self, candidate_weights, atol=1e-6):
        """Element-wise count of weights that differ from the
        original genome by more than atol."""
        return np.count_nonzero(
            np.abs(np.asarray(candidate_weights, float) - self.original_genome) > atol
        )

    def run(
        self,
        env,
        batch_size: int = 10,
        jitter_strength: float = 0.0,
    ):
        folder: str = "data_full_pentagon"
        # ── plotting setup ──
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20), sharex=True)
        ax1.set_title("Fitness – Pentagon Food Pattern")
        ax1.set_ylabel("Food targets consumed at End of Training")

        ax2.set_title("Euclidean distance")
        ax2.set_ylabel("L2 distance")

        ax3.set_title("Number of weight changes")
        ax3.set_ylabel("Changed synapses")
        ax3.set_xlabel("Generation")

        # dynamic metric storage
        metrics = {
            "fitness":  defaultdict(list),
            "distance": defaultdict(list),
            "changes":  defaultdict(list),
        }

        # colour → label mapping
        label_map = {
            "royalblue":  "OPENAI ES",
            "forestgreen": "Evolutionary algorithm",
            "crimson":     "Large-diff search",
            "darkorange":  "Random 50 search",
            "purple":      "Pure Nomad",
            "black":       "NOMAD Hybrid",
            "red":  "CMA-ES"
        }
        colour_axes = {"fitness": ax1, "distance": ax2, "changes": ax3}

        base_dir = os.path.dirname(__file__)
        full_folder = os.path.join(base_dir, folder)

        # ── gather metrics ──
        for filename in os.listdir(full_folder):
            self.population.clear()
            genomes = read_arrays_from_csv_pandas(
                os.path.join(full_folder, filename)
            )
            self.initialize_population(genomes)

            fitness, dist, changes = [], [], []
            batches = [
                self.population[i:i + batch_size]
                for i in range(0, len(self.population), batch_size)
            ]

            for batch in batches:
                # fitness (parallel)
                fitness.extend(ray.get([
                    evaluate_fitness_ray.remote(
                        c.weight_matrix, all_neuron_names, env,
                        self.food_patterns, mLeft, mRight,
                        muscleList, muscles,
                        self.training_interval, self.total_episodes
                    ) for c in batch
                ]))
                # distance & changes
                #d_batch = [self.calculate_euclidean_distance(c.weight_matrix)
                #    for c in batch]
                dist.extend([
                    self.calculate_euclidean_distance(c.weight_matrix) + jitter_strength
                    for c in batch
                ])
                changes.extend([self.count_changes(c.weight_matrix) for c in batch])
                #print(f"{filename:25s}  "
                #    f"L2 mean={np.mean(d_batch):7.3f}  std={np.std(d_batch):.3e}  "
                #    f"Δw unique={len(set(changes))}")
            # ── decide colour bucket ──
            fname = filename.lower()
            if "hybrid" in fname:
                colour = "black"
            elif "cmaes" in fname:
                colour = "red"
            elif "evolutionary" in fname:
                colour = "forestgreen"
            elif "random" in fname:
                colour = "darkorange"
            elif "pure" in fname:
                colour = "purple"
            elif len(dist) > 90 and dist[90] > 500:
                colour = "crimson"
            else:
                colour = "royalblue"

            # store
            metrics["fitness"][colour].append(fitness)
            metrics["distance"][colour].append(dist)
            metrics["changes"][colour].append(changes)
        print(len(metrics["changes"]["royalblue"]))
        # ── helper to stack, trim & compute mean±sd ──
        def mean_sd(runs):
            min_len = min(map(len, runs))
            stack = np.asarray([r[:min_len] for r in runs], float)
            return stack.mean(0), stack.std(0)

        # ── draw curves + shaded bands ──
        for metric in ("fitness", "distance", "changes"):
            for colour, runs in metrics[metric].items():
                if not runs:
                    continue
                mean, sd = mean_sd(runs)
                x = np.arange(len(mean))
                ax = colour_axes[metric]
                ax.plot(x, mean, lw=2, color=colour,
                        label=f"{label_map.get(colour,'unknown')} (mean)")
                ax.fill_between(x, mean - sd, mean + sd,
                                color=colour, alpha=0.15)

        for a in (ax1, ax2, ax3):
            a.set_xscale("log")
        ax1.legend(fontsize=10, ncol=2)

        ax3.set_yscale("log")
        ax2.set_yscale("log")

        plt.tight_layout()
        plt.savefig("fig7.svg")
        ray.shutdown()