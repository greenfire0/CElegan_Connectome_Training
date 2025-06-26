import numpy as np
import jax, jax.numpy as jnp
import optax
from evosax.algorithms import Open_ES            # ← 100 % correct class
import ray
from tqdm import tqdm
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import (muscles, muscleList,
                                  mLeft, mRight, all_neuron_names)
from Algorithms.algo_utils import evaluate_fitness_static
from util.snip import write_worm_to_csv
import multiprocessing as mp
mp.set_start_method("spawn", force=True)


# --------------------------------------------------------------------- #
@ray.remote(num_cpus=1)
def _fit_worker(weights, env, prob_type, interval, episodes):
    """Ray wrapper around your deterministic fitness."""
    return evaluate_fitness_static(
        weights, all_neuron_names, env,
        prob_type, mLeft, mRight, muscleList, muscles,
        interval, episodes
    )


# --------------------------------------------------------------------- #
def train_openai_es(
        env,
        init_genome: np.ndarray,
        generations: int = 300,
        pop_size: int = 512,          # ↑ bigger population
        sigma: float = 0.20,          # ↑ wider exploration radius
        lr: float   = 0.05,           # ↑ larger initial LR (will decay)
        prob_type: list[int] = [5],
        interval: int = 250,
        episodes: int = 0,
        csv_log: str = "ES_worms"):


    # ---- schedules -------------------------------------------------------
    lr_schedule  = optax.exponential_decay(
        init_value=lr,
        transition_steps=200,         # start decaying after 200 gens
        decay_rate=0.3,
    )
    std_schedule = optax.exponential_decay(
        init_value=sigma,
        transition_steps=150,         # decay σ slightly sooner
        decay_rate=0.3,
    )

    # ---- build optimiser -------------------------------------------------
    solution = jnp.asarray(init_genome, jnp.float32)
    es = Open_ES(
        population_size=pop_size,
        solution=solution,
        use_antithetic_sampling=True,
        optimizer=optax.adam(lr_schedule),
        std_schedule=std_schedule,
    )
    params = es.default_params

    key = jax.random.PRNGKey(np.random.randint(0, 2**32-1))
    key, sub = jax.random.split(key)
    state = es.init(sub, solution, params)        # ESState dataclass

    best_reward = -np.inf
    best_theta  = np.asarray(solution)

    # ---- evolution loop --------------------------------------------------
    for g in tqdm(range(1, generations + 1)):
        key, k_ask, k_tell = jax.random.split(key, 3)

        # ask ----------------------------------------------------------------
        population, state = es.ask(k_ask, state, params)    # JAX (pop, n)
        pop_np = np.asarray(population)

        # evaluate -----------------------------------------------------------
        futs = [_fit_worker.remote(p, env, prob_type, interval, episodes)
                for p in pop_np]
        rewards = np.asarray(ray.get(futs), dtype=np.float32)

        # tell  (evosax minimises => pass negative rewards as “costs”) -------
        costs = -rewards
        state, _ = es.tell(k_tell, population, costs, state, params)

        # bookkeeping --------------------------------------------------------
        gen_best_idx = rewards.argmax()
        gen_best     = rewards[gen_best_idx]
        best_reward = float(gen_best)
        best_theta  = pop_np[gen_best_idx].copy()
        write_worm_to_csv(csv_log,
                              WormConnectome(weight_matrix=best_theta,
                                             all_neuron_names=all_neuron_names),max_rows=generations)

        if g % 100 == 0:
            print(f"Gen {g:4d} | best so far: {best_reward:.3f}")

    ray.shutdown()
    return best_theta