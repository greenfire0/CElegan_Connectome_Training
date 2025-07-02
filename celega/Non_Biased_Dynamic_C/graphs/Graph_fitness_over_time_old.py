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


    def count_changes(self, candidate_weights, atol=1e-6):
        """Element-wise count of weights that differ from the
        original genome by more than atol."""
        return np.count_nonzero(
            np.abs(np.asarray(candidate_weights, float) - self.original_genome) > atol
        )
    
    def calculate_euclidean_distance(self, candidate_weights):
        candidate_weights = np.array(candidate_weights, dtype=float)
        if candidate_weights.shape != self.original_genome.shape:
            raise ValueError("Shape of candidate_weights and original_genome must be the same.")
        distance = np.linalg.norm(candidate_weights - self.original_genome)
        return distance


    def run(self, env, batch_size: int = 10, jitter_strength: float = 0.0):
        folder: str = "data_full_pentagon"

        # ── plotting setup ──
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20), sharex=True)
        ax1.set_title("Fitness – Pentagon Food Pattern", fontsize=20)
        ax1.set_ylabel("Food Targets Consumed at End of Training", fontsize=18)
        ax2.set_title("Euclidean distance", fontsize=20)
        ax2.set_ylabel("L2 distance", fontsize=18)
        ax3.set_title("Number of weight changes", fontsize=20)
        ax3.set_ylabel("Changed synapses", fontsize=18)
        ax3.set_xlabel("Time (minutes)", fontsize=18)  # Updated label for time axis
        for ax in (ax1, ax2, ax3):
            ax.tick_params(axis="both", labelsize=16)

        metrics = {"fitness": defaultdict(list),
                   "distance": defaultdict(list),
                   "changes": defaultdict(list)}

        label_map = {
            "royalblue":  "OPENAI ES",
            "forestgreen": "Evolutionary",
            "crimson":     "Large-diff search",
            "darkorange":  "Random 50 search",
            "purple":      "Pure Nomad",
            "black":       "NOMAD Hybrid",
            "red":         "CMA-ES",
        }
        colour_axes = {"fitness": ax1, "distance": ax2, "changes": ax3}

        base_dir   = os.path.dirname(__file__)
        full_folder = os.path.join(base_dir, folder)

        # ── gather metrics ──
        for filename in os.listdir(full_folder):
            self.population.clear()
            genomes = read_arrays_from_csv_pandas(os.path.join(full_folder, filename))
            self.initialize_population(genomes)

            fitness, dist, changes = [], [], []
            batches = [
                self.population[i:i + batch_size]
                for i in range(0, len(self.population), batch_size)
            ]

            for batch in batches:
                fitness.extend(
                    ray.get([
                        evaluate_fitness_ray.remote(
                            c.weight_matrix, all_neuron_names, env,
                            self.food_patterns, mLeft, mRight,
                            muscleList, muscles,
                            self.training_interval, self.total_episodes
                        ) for c in batch
                    ])
                )
                dist.extend([
                    self.calculate_euclidean_distance(c.weight_matrix) + jitter_strength
                    for c in batch
                ])
                changes.extend([self.count_changes(c.weight_matrix) for c in batch])

            # colour bucket
            fname = filename.lower()
            colour = (
                "black"      if "hybrid"      in fname else
                "red"        if "cmaes"       in fname else
                "forestgreen"if "evolutionary"in fname else
                "darkorange" if "random"      in fname else
                "purple"     if "pure"        in fname else
                "crimson"    if len(dist) > 90 and dist[90] > 500 else
                "royalblue"
            )

            metrics["fitness"][colour].append(fitness)
            metrics["distance"][colour].append(dist)
            metrics["changes"][colour].append(changes)

        # ── common timeline ──
        max_len = max(
            len(r)
            for metric in metrics.values()
            for runs   in metric.values()
            for r      in runs
        )

        def resample(run, tgt):
            """Linear-interpolate run to length tgt."""
            if len(run) == tgt:
                return np.asarray(run, float)
            x_old = np.arange(len(run))
            x_new = np.linspace(0, len(run) - 1, tgt)
            return np.interp(x_new, x_old, run).astype(float)

        # Map generations to a 0–20‑minute timeline
        x = np.linspace(0, 20, max_len)  # Updated x-axis for time in minutes

        # ── draw curves ──
        for metric in ("fitness", "distance", "changes"):
            for colour, runs in metrics[metric].items():
                if not runs:
                    continue
                up = np.vstack([resample(r, max_len) for r in runs])
                mean, sd = up.mean(0), up.std(0)
                ax = colour_axes[metric]
                ax.plot(x, mean, lw=2, color=colour,
                        label=f"{label_map.get(colour,'unknown')} (mean)")
                ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15)

        # ── axes / legend ──
        ax2.set_yscale("log")
        ax3.set_yscale("log")
        ax1.set_ylim([0, 36])

        legend_order = [
            "Pure Nomad", "NOMAD Hybrid", "Evolutionary",
            "OPENAI ES", "Large-diff search", "Random 50 search", "CMA-ES",
        ]
        h, l = ax1.get_legend_handles_labels()
        m = {lab.replace(" (mean)", ""): (handle, lab) for handle, lab in zip(h, l)}
        ax1.legend([m[o][0] for o in legend_order if o in m],
                   [m[o][1] for o in legend_order if o in m],
                   fontsize=14, ncol=2, loc="lower right")

        plt.tight_layout()
        plt.savefig("fig7.svg")
                # ------  Compute the “x” values for the paper  ------


        # Map the plot colours back to human-readable algorithm names.
        LABEL = {
            "royalblue":  "OpenAI-ES",
            "forestgreen":"Evolutionary",
            "crimson":    "Large-∆ Search",
            "darkorange": "Random-50 Search",
            "purple":     "Pure NOMAD",
            "black":      "NOMAD Hybrid",
            "red":        "CMA-ES",
        }

        summary = {}

        for colour, runs in metrics["fitness"].items():
            if not runs:            # skip groups with no data
                continue
            # Stack every run, grab the final time-point for each, then summarise.
            last_scores = np.vstack(runs)[:, -1].astype(float)
            summary[LABEL[colour]] = {
                "mean" : last_scores.mean(),
                "std"  : last_scores.std(ddof=1),
                "min"  : last_scores.min(),
                "max"  : last_scores.max(),
                "n"    : len(last_scores),
            }

        # Helpful percent-gain examples
        try:
            evo_mean   = summary["Evolutionary"]["mean"]
            hybrid_mean = summary["NOMAD Hybrid"]["mean"]
            pure_mean   = summary["Pure NOMAD"]["mean"]

            summary["Hybrid vs Evolutionary % gain"] = 100 * (hybrid_mean - evo_mean) / evo_mean
            summary["Pure vs Evolutionary % gain"]   = 100 * (pure_mean   - evo_mean) / evo_mean
        except KeyError:
            pass   # one of the algorithms wasn’t present this run

        # ------  Print nicely for copy-paste  ------
        print("\n=====  Final-generation food-targets consumed  =====")
        for key, vals in summary.items():
            if isinstance(vals, dict):
                print(f"{key:20s}:  {vals['mean']:.2f} ± {vals['std']:.2f}  "
                    f"(min {vals['min']:.1f}, max {vals['max']:.1f}, n = {vals['n']})")
            else:   # percent gains
                print(f"{key:20s}:  {vals:.1f}%")


        summary = {}

        # iterate over every colour that *actually* has distance data
        for colour, dist_runs in metrics["distance"].items():
            change_runs = metrics["changes"].get(colour, [])
            if not dist_runs or not change_runs:          # skip if either metric missing
                continue

            # Stack each run, then take the final generation’s value for every run
            last_dist    = np.vstack(dist_runs)[:,   -1].astype(float)
            last_changes = np.vstack(change_runs)[:, -1].astype(float)

            label = LABEL.get(colour, colour)             # fall back to raw colour if unknown
            summary[label] = {
                "dist_mean"   : last_dist.mean(),
                "dist_std"    : last_dist.std(ddof=1),
                "dist_min"    : last_dist.min(),
                "dist_max"    : last_dist.max(),
                "change_mean" : last_changes.mean(),
                "change_std"  : last_changes.std(ddof=1),
                "change_min"  : last_changes.min(),
                "change_max"  : last_changes.max(),
                "n"           : len(last_dist),
            }

        print("\n=====  Final-generation metrics =====")
        for name, v in summary.items():
            print(f"{name:15s}:  "
                f"L2 = {v['dist_mean']:.2f} ± {v['dist_std']:.2f}"
                f"  (min {v['dist_min']:.1f}, max {v['dist_max']:.1f}) | "
                f"changes = {v['change_mean']:.1f} ± {v['change_std']:.1f}"
                f"  (min {v['change_min']:.0f}, max {v['change_max']:.0f}, n={v['n']})")


        ray.shutdown()
