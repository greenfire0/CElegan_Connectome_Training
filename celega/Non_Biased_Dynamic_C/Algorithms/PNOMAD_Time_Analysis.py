import time
from collections import defaultdict
import numpy as np
import ray
from tqdm import tqdm
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import (
    muscles, muscleList, mLeft, mRight, all_neuron_names
)
from Algorithms.algo_utils import (
    initialize_population, select_parents, crossover,
    evaluate_fitness_static, evaluate_fitness_nomad
)
from util.snip import write_worm_to_csv
import numpy.typing as npt


class Genetic_Dyn_Algorithm:
    """
    Genetic algorithm with fine-grained timing.

    After a run() you’ll see:

    ┌────────────────────────┐
    │ total wall-clock  1253 │
    │ ├─ fitness eval   930  │  <- time blocked on ray.get
    │ │  ├─ pure fitness 698 │  <- inside evaluate_fitness_static
    │ │  └─ NOMAD / OH  232  │
    │ ├─ selection       12  │
    │ ├─ crossover       24  │
    │ ├─ unpack results  27  │
    │ ├─ csv write        7  │
    │ └─ other driver   253  │
    └────────────────────────┘
    """

    def __init__(
        self,
        population_size: int,
        genome: npt.NDArray[np.float64],
        pattern: list[int] = [5],
        total_episodes: int = 0,
        training_interval: int = 250,
        matrix_shape: int = 3689,
        indicies: list[int] | None = None,
    ):
        self.population_size = population_size
        self.indicies = indicies or []
        self.matrix_shape = matrix_shape
        self.total_episodes = total_episodes
        self.training_interval = training_interval
        self.food_patterns = pattern
        assert genome.size == matrix_shape
        self.population = initialize_population(population_size, genome)

    # ──────────────────────────────────────────────────────────────────────────
    # RUN
    # ──────────────────────────────────────────────────────────────────────────
    def run(
        self,
        env,
        generations: int = 50,
        batch_size: int = 32,
        filename: str = "Pure_nomad",
    ):
        # master stop-watch buckets
        wall = defaultdict(float)   # wall["fitness"], wall["csv"], …
        wall["fit_wall"]      = 0.0   # wall time spent in fitness (slowest task)
        wall["nomad_wall"]    = 0.0   # wall time spent in PyNomad
        wall["overhead_wall"] = 0.0   # wall time spent in Ray / misc

        overall_start = time.perf_counter()

        try:
            with tqdm(range(generations), desc="Gen", unit="Gen") as pbar:
                for gen in pbar:
                    # ═════════════════ FITNESS EVALUATIONS ═══════════════════
                    t0 = time.perf_counter()
                    futures = [
                        evaluate_fitness_nomad.remote(
                            evaluate_fitness_static,
                            candidate_weights=cand.weight_matrix,
                            nur_name=all_neuron_names,
                            env=env,
                            prob_type=self.food_patterns,
                            mLeft=mLeft,
                            mRight=mRight,
                            muscleList=muscleList,
                            muscles=muscles,
                            interval=self.training_interval,
                            episodes=self.total_episodes,
                            ind=np.random.choice(self.matrix_shape, 49, False),
                            bounds=4,
                            bb_eval=250,
                            verify=False,
                        )
                        for cand in self.population
                    ]
                    results = ray.get(futures)
                    wall["fitness_block"] += time.perf_counter() - t0
                    # ── split task wall vs cpu
                    # ── aggregate CPU, and track the *slowest* task for wall breakdown
                    slow_wall = slow_fit = slow_nomad = slow_ovh = 10000000

                    for _, _, cpu_sec, wall_task, cpu_nomad in results:   # 5-tuple
                        ovh = max(wall_task - cpu_sec - cpu_nomad, 0.0)

                        wall["fitness_cpu"]   += cpu_sec
                        wall["nomad_cpu"]     += cpu_nomad
                        wall["nomad_overhead"] += ovh

                        if wall_task < slow_wall:          # keep the task that set the block time
                            slow_wall, slow_fit, slow_nomad, slow_ovh = (
                                wall_task,
                                cpu_sec,
                                cpu_nomad,
                                ovh,
                            )

                    # add that task’s components to per-gen wall totals
                    wall["fit_wall"]      += slow_fit
                    wall["nomad_wall"]    += slow_nomad
                    wall["overhead_wall"] += slow_ovh

                    # ═════════════════ UNPACK / STORE RESULTS ════════════════
                    t0 = time.perf_counter()
                    fitnesses = []
                    for a,result in enumerate(results):
                            self.population[a].weight_matrix[result[0][0]] = np.copy(result[0][1])
                            fitnesses.append(np.max([(result[1]),0]))
                    wall["unpack"] += time.perf_counter() - t0

                    # ═════════════════ SELECTION ════════════════════════════

                    # ═════════════════ CROSSOVER / MUTATE ═══════════════════
                    best_index = np.argmax(fitnesses)
                    best_fitness = fitnesses[best_index]
                    best_weights = np.copy(self.population[best_index].weight_matrix)
                    best_worm = WormConnectome(weight_matrix=best_weights, all_neuron_names=all_neuron_names)
                    t0 = time.perf_counter()
                    self.population = select_parents(
                        self.population, fitnesses, self.population_size // 2
                    )
                    wall["selection"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    self.population.extend(
                        crossover(
                            self.population,
                            fitnesses,
                            self.population_size - len(self.population),
                            self.matrix_shape,
                        )
                    )
                    wall["crossover"] += time.perf_counter() - t0

                

                    # ═════════════════ CSV WRITE ════════════════════════════
                    t0 = time.perf_counter()
                    write_worm_to_csv(filename, best_worm, max_rows=generations)
                    wall["csv_write"] += time.perf_counter() - t0

                    pbar.set_description(f"Gen {gen+1}")
                    pbar.set_postfix(best=f"{best_fitness:.2f}")            

        finally:
            ray.shutdown()

        wall_total = time.perf_counter() - overall_start
        wall["other_driver"] = (
            wall_total
            - wall["fitness_block"]
            - wall["selection"]
            - wall["crossover"]
            - wall["unpack"]
            - wall["csv_write"]
        )

        # ──────────────────── REPORT ────────────────────────────────
        def pct(x):  # pretty %
            return f"{100*x/wall_total:5.1f}%"
        def pct_inner(x):
            return f"{100*x/wall['fitness_block']:5.1f}%"

        print("\n======= TIMING BREAKDOWN (wall-clock) =======")
        print(f"{'total':18}: {wall_total:8.2f} s (100%)")
        print(f"  fitness blocked      : {wall['fitness_block']:8.2f} s {pct(wall['fitness_block'])}")
        print(f"    ├─ pure fitness cpu: {wall['fitness_cpu']:8.2f} s")
        print(f"    ├─ PyNomad cpu      : {wall['nomad_cpu']:8.2f} s")
        print(f"    └─ nomad / overhead: {wall['nomad_overhead']:8.2f} s")
        print(f"    ├─ pure fitness wall: {wall['fit_wall']:8.2f} s {pct_inner(wall['fit_wall'])}")
        print(f"    ├─ PyNomad wall     : {wall['nomad_wall']:8.2f} s {pct_inner(wall['nomad_wall'])}")
        print(f"    └─ other overhead   : {wall['overhead_wall']:8.2f} s {pct_inner(wall['overhead_wall'])}")
        print(f"  selection            : {wall['selection']:8.2f} s {pct(wall['selection'])}")
        print(f"  crossover            : {wall['crossover']:8.2f} s {pct(wall['crossover'])}")
        print(f"  unpack results       : {wall['unpack']:8.2f} s {pct(wall['unpack'])}")
        print(f"  csv write            : {wall['csv_write']:8.2f} s {pct(wall['csv_write'])}")
        print(f"  other driver code    : {wall['other_driver']:8.2f} s {pct(wall['other_driver'])}\n")

        # optional: parallel efficiency for the pure-fitness slice
        if wall["fitness_cpu"]:
            par = wall["fitness_cpu"] / max(wall["fitness_block"], 1e-6)
            print(f"effective parallelism (fitness): ×{par:4.1f}")

        return best_worm.weight_matrix
