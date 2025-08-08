import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import (
    muscles,
    muscleList,
    mLeft,
    mRight,
    all_neuron_names,
)
from matplotlib import pyplot as plt
import os
from util.write_read_txt import read_arrays_from_csv_pandas
from collections import defaultdict
from Algorithms.algo_utils import evaluate_fitness_ray


class Genetic_Dyn_Algorithm:
    """Dynamic‑training genetic search over C. elegans connectome weights."""

    # ─────────────────────────────────────────────────────────────────────────────
    #  Construction helpers
    # ─────────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        population_size: int,
        pattern=[5],
        total_episodes: int = 10,
        training_interval: int = 25,
        genome=None,
        matrix_shape: int = 3689,
    ) -> None:
        self.population_size = population_size
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.original_genome = (
            np.array(genome, dtype=float) if genome is not None else np.zeros(matrix_shape)
        )
        self.food_patterns = pattern
        self.population = []

    # ─────────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────────
    def initialize_population(self, genomes=None):
        if genomes is None:
            raise ValueError("Genomes must be provided to initialize the population.")
        for g in genomes:
            self.population.append(
                WormConnectome(weight_matrix=np.array(g, dtype=float), all_neuron_names=all_neuron_names)
            )

    def count_changes(self, candidate_weights, atol: float = 1e-6) -> int:
        """Count synapses whose weight differs from the original genome by more than *atol*."""
        return np.count_nonzero(
            np.abs(np.asarray(candidate_weights, float) - self.original_genome) > atol
        )

    def get_colour(self, fname: str, dist):
        fname = fname.lower()
        if "evolutionary" in fname:
            return "forestgreen"
        if "hybrid" in fname and "nomad" in fname:
            return "black"
        if "cmaes" in fname:
            return "red"
        if "random" in fname:
            return "darkorange"
        if "pure" in fname:
            return "purple"
        if len(dist) > 90 and dist[90] > 500:
            return "crimson"
        return "royalblue"

    def calculate_euclidean_distance(self, candidate_weights):
        candidate_weights = np.array(candidate_weights, dtype=float)
        if candidate_weights.shape != self.original_genome.shape:
            raise ValueError("Shape mismatch between candidate_weights and original_genome.")
        return np.linalg.norm(candidate_weights - self.original_genome)

    # ─────────────────────────────────────────────────────────────────────────────
    #  Main routine
    # ─────────────────────────────────────────────────────────────────────────────
    def run(self, env, batch_size: int = 10, jitter_strength: float = 0.0):
        folder: str = "data_new_pentagon"

        # ── plotting setup: horizontal 1 × 3 layout ──
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(30, 10), sharex=True, constrained_layout=True
        )
        ax1.set_ylabel("Food Targets", fontsize=28)
        ax1.set_title("Task Performance", fontsize=30, pad=18)
        ax2.set_title("Distance to Original Connectome", fontsize=30, pad=18)
        ax2.set_ylabel("L2 distance", fontsize=28)
        ax3.set_title("Number of weight changes", fontsize=30, pad=18)
        ax3.set_ylabel("Changed synapses", fontsize=28)
        for ax in (ax1, ax2, ax3):
            ax.set_xlabel("Time (minutes)", fontsize=28)
            ax.tick_params(axis="both", labelsize=24)

        # metrics container
        metrics = {k: defaultdict(list) for k in ("fitness", "distance", "changes")}
        colour_axes = {"fitness": ax1, "distance": ax2, "changes": ax3}
        label_map = {
            "royalblue": "OPENAI-ES",
            "crimson": "Large-diff search",
            "darkorange": "Rand. Mutation NOMAD",
            "purple": "rENOMAD",
            "black": "mENOMAD",
            "forestgreen": "Evolutionary",
        }

        base_dir = os.path.dirname(__file__)
        full_folder = os.path.join(base_dir, folder)

        # ── BEST‑WORM BOOK‑KEEPING ────────────────────────────────────────────
        best_pure_score, best_pure_file, best_pure_idx = -np.inf, None, None
        best_hybrid_score, best_hybrid_file, best_hybrid_idx = -np.inf, None, None

        # ── gather metrics over every CSV in the folder ──────────────────────
        for filename in os.listdir(full_folder)[:20]:
            self.population.clear()
            genomes = read_arrays_from_csv_pandas(os.path.join(full_folder, filename))
            self.initialize_population(genomes)

            fitness, dist, changes = [], [], []
            batches = [
                self.population[i : i + batch_size]
                for i in range(0, len(self.population), batch_size)
            ]

            for batch in batches:
                # expensive roll‑outs – parallelised with Ray
                fitness.extend(
                    ray.get(
                        [
                            evaluate_fitness_ray.remote(
                                c.weight_matrix,
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
                            for c in batch
                        ]
                    )
                )
                # cheap post‑processing
                dist.extend(
                    [
                        self.calculate_euclidean_distance(c.weight_matrix) + jitter_strength
                        for c in batch
                    ]
                )
                changes.extend([self.count_changes(c.weight_matrix) for c in batch])

            # ── update best‑pure / best‑hybrid trackers ──
            local_best = max(fitness)
            local_idx = fitness.index(local_best)
            fname_lower = filename.lower()
            if "pure" in fname_lower and local_best > best_pure_score:
                best_pure_score, best_pure_file, best_pure_idx = local_best, filename, local_idx
            if "hybrid" in fname_lower and local_best > best_hybrid_score:
                best_hybrid_score, best_hybrid_file, best_hybrid_idx = (
                    local_best,
                    filename,
                    local_idx,
                )

            # assign colour bucket & stash series
            colour = self.get_colour(filename, dist)
            metrics["fitness"][colour].append(fitness)
            metrics["distance"][colour].append(dist)
            metrics["changes"][colour].append(changes)

        # ── PLOTTING ──────────────────────────────────────────────────────────
        max_len = max(
            len(r) for metric in metrics.values() for runs in metric.values() for r in runs
        )

        def resample(run, tgt):
            if len(run) == tgt:
                return np.asarray(run, float)
            x_old = np.arange(len(run))
            x_new = np.linspace(0, len(run) - 1, tgt)
            return np.interp(x_new, x_old, run).astype(float)

        x = np.linspace(0, 20, max_len)
        excluded = {"gold", "teal"}
        for metric in ("fitness", "distance", "changes"):
            for colour, runs in metrics[metric].items():
                if colour in excluded or not runs:
                    continue
                up = np.vstack([resample(r, max_len) for r in runs])
                mean, sd = up.mean(0), up.std(0)
                ax = colour_axes[metric]
                ax.plot(x, mean, lw=2, color=colour, label=f"{label_map.get(colour, 'unknown')} (mean)")
                ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15)

        ax2.set_yscale("log")
        ax3.set_yscale("log")
        ax1.set_ylim([0, 36])

        legend_order = [
            "mENOMAD",
            "rENOMAD",
            "Evolutionary",
            "OPENAI-ES",
            "Large-diff search",
            "Rand. Mutation NOMAD",
        ]
        h, l = ax1.get_legend_handles_labels()
        m = {lab.replace(" (mean)", ""): (handle, lab) for handle, lab in zip(h, l)}
        ax2.legend(
            [m[o][0] for o in legend_order if o in m],     # handles
            [m[o][1] for o in legend_order if o in m],     # labels
            fontsize=18,
            ncol=1,
            loc='upper left',                              # anchor point of the legend box
            bbox_to_anchor=(0.5, 0.2),                     # x, y coordinates in axes fraction
        )

        for i, ax in enumerate((ax1, ax2, ax3)):
            ax.text(
                -0.12,
                1.04,
                f"{chr(97 + i)})",
                transform=ax.transAxes,
                fontsize=28,
                fontweight="bold",
                va="top",
                ha="left",
            )

        plt.savefig("fig7.svg")

        # ── REPORT BEST PURE / HYBRID WORMS ──────────────────────────────────
        print("\n===  Best Pure NOMAD worm  ===")
        if best_pure_file is not None:
            print(f"File : {best_pure_file}")
            print(f"Index: {best_pure_idx}")
            print(f"Score: {best_pure_score:.2f}")
        else:
            print("No \"pure\" file found.")

        print("\n===  Best Hybrid NOMAD worm ===")
        if best_hybrid_file is not None:
            print(f"File : {best_hybrid_file}")
            print(f"Index: {best_hybrid_idx}")
            print(f"Score: {best_hybrid_score:.2f}")
        else:
            print("No \"hybrid\" file found.")

        # ── teardown ──
        ray.shutdown()
