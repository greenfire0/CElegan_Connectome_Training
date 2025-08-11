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
        if "pure_nomad_random" in fname:
            return "crimson"
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
            1, 3, figsize=(30, 9), sharex=True, constrained_layout=True
        )
        ax1.set_ylabel("Food Targets Consumed", fontsize=30)
        ax1.set_title("Task Performance", fontsize=32, pad=18)
        ax2.set_title("Distance to Original Connectome", fontsize=32, pad=18)
        ax2.set_ylabel("L2 distance", fontsize=30)
        ax3.set_title("Number of Weight Changes", fontsize=32, pad=18)
        ax3.set_ylabel("Changed synapses", fontsize=30)
        for ax in (ax1, ax2, ax3):
            ax.set_xlabel("Time (minutes)", fontsize=30)
            ax.tick_params(axis="both", labelsize=30)

        # metrics container
        metrics = {k: defaultdict(list) for k in ("fitness", "distance", "changes")}
        colour_axes = {"fitness": ax1, "distance": ax2, "changes": ax3}
        label_map = {
            "royalblue": "OPENAI-ES",
            "crimson": "Large-L2 search",
            "darkorange": "cfNOMAD",
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
        for filename in os.listdir(full_folder):
            if ("pure_nomad_random" in filename.lower()):
                continue
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
            "Large-L2 search",
            "cfNOMAD",
        ]
        h, l = ax1.get_legend_handles_labels()
        m = {lab.replace(" (mean)", ""): (handle, lab) for handle, lab in zip(h, l)}
        ax1.legend(
            [m[o][0] for o in legend_order if o in m],     # handles
            [m[o][1] for o in legend_order if o in m],     # labels
            fontsize=26,
            ncol=1,
            loc='upper left',                              # anchor point of the legend box
            bbox_to_anchor=(0.25, 0.41),       #41               # x, y coordinates in axes fraction
        )

        for i, ax in enumerate((ax1, ax2, ax3)):
            ax.text(
                -0.13,
                1.09,
                f"({chr(97 + i)})",
                transform=ax.transAxes,
                fontsize=34,
                fontweight="bold",
                va="top",
                ha="left",
            )

        
        plt.savefig("fig7.png",dpi=300)


        LABEL = {
            "royalblue": "OPENAI-ES",
            "crimson": "crimson",
            "darkorange": "cfNOMAD",
            "purple": "rENOMAD",
            "black": "mENOMAD",
            "forestgreen": "Evolutionary",
        }
        summary = {}

        # ── Final‑generation food‑target metrics ──
        for colour, runs in metrics["fitness"].items():
            if not runs:
                continue  # skip groups with no data
            last_scores = np.vstack(runs)[:, -1].astype(float)
            summary[LABEL.get(colour, colour)] = {
                "mean": last_scores.mean(),
                "std": last_scores.std(ddof=1),
                "min": last_scores.min(),
                "max": last_scores.max(),
                "n": len(last_scores),
            }
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
        try:
                    evo_mean = summary["Evolutionary"]["mean"]
                    hybrid_mean = summary["NOMAD Hybrid"]["mean"]
                    pure_mean = summary["Pure NOMAD"]["mean"]
                    summary["Hybrid vs Evolutionary % gain"] = 100 * (hybrid_mean - evo_mean) / evo_mean
                    summary["Pure vs Evolutionary % gain"] = 100 * (pure_mean - evo_mean) / evo_mean
        except KeyError:
                    pass  # one of the algorithms wasn’t present this run

        print("\n=====  Final-generation food-targets consumed  =====")
        for key, vals in summary.items():
            if isinstance(vals, dict):
                print(
                    f"{key:20s}:  {vals['mean']:.2f} ± {vals['std']:.2f}  "
                    f"(min {vals['min']:.1f}, max {vals['max']:.1f}, n = {vals['n']})"
                )
            else:  # percent gains
                print(f"{key:20s}:  {vals:.1f}%")

        summary = {}

        # ── Final‑generation distance / change metrics ──
        for colour, dist_runs in metrics["distance"].items():
            change_runs = metrics["changes"].get(colour, [])
            if not dist_runs or not change_runs:
                continue  # skip if either metric missing

            last_dist = np.vstack(dist_runs)[:, -1].astype(float)
            last_changes = np.vstack(change_runs)[:, -1].astype(float)

            label = LABEL.get(colour, colour)
            summary[label] = {
                "dist_mean": last_dist.mean(),
                "dist_std": last_dist.std(ddof=1),
                "dist_min": last_dist.min(),
                "dist_max": last_dist.max(),
                "change_mean": last_changes.mean(),
                "change_std": last_changes.std(ddof=1),
                "change_min": last_changes.min(),
                "change_max": last_changes.max(),
                "n": len(last_dist),
            }

        print("\n=====  Final-generation metrics =====")
        for name, v in summary.items():
            print(
                f"{name:20s}:  "
                f"L2 = {v['dist_mean']:.2f} ± {v['dist_std']:.2f} "
                f" (min {v['dist_min']:.1f}, max {v['dist_max']:.1f}) | "
                f"changes = {v['change_mean']:.1f} ± {v['change_std']:.1f} "
                f" (min {v['change_min']:.0f}, max {v['change_max']:.0f}, n={v['n']})"
            )
        # ── teardown ──
        ray.shutdown()


