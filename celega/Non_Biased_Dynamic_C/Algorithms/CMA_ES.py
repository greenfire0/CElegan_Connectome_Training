from __future__ import annotations
"""CMA‑ES trainer for the *C. elegans* connectome.

Key fixes (2025‑06‑26):
  1. **Latent mean starts at user’s `init_genome`** (centred search).
  2. **Pass *positive* rewards to `es.tell`** when using rank shaping.
  3. Keeps fallbacks for missing `sigma` in recent *evosax*.
"""

import multiprocessing as mp
from typing import Callable, Optional, Sequence, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import ray
from evosax.algorithms import CMA_ES
from tqdm import tqdm

from Algorithms.algo_utils import evaluate_fitness_static
from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import (
    all_neuron_names,
    mLeft,
    mRight,
    muscleList,
    muscles,
)
from util.snip import write_worm_to_csv

mp.set_start_method("spawn", force=True)

# ──────────────────────────────────────────────────────────────────────────────
@ray.remote(num_cpus=1)
def _fit_worker(
    weights: np.ndarray,
    env,
    prob_type: Sequence[int],
    interval: int,
    episodes: int,
) -> float:
    """Deterministic fitness wrapper for Ray."""

    return float(
        evaluate_fitness_static(
            weights,
            all_neuron_names,
            env,
            prob_type,
            mLeft,
            mRight,
            muscleList,
            muscles,
            interval,
            episodes,
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# CMA‑ES loop
# ──────────────────────────────────────────────────────────────────────────────

def train_cma_es(
    env,
    init_genome: np.ndarray,
    *,
    generations: int = 300,
    pop_size: Optional[int] = None,
    prob_type: Optional[Sequence[int]] = None,
    interval: int = 250,
    episodes: int = 0,
    csv_log: str = "CMAES_worms",
    search_scale: float = 0.25,
    fitness_shaping_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    metrics_fn: Optional[
        Callable[[jax.Array, jax.Array, jax.Array, Any, Any], Dict[str, Any]]
    ] = None,
) -> np.ndarray:
    """Evolve locomotion with CMA‑ES using a latent search space."""

    # — defaults —
    if prob_type is None:
        prob_type = [5]

    if pop_size is None:
        pop_size = int(8 + 6 * np.log(init_genome.size))

    # — fitness shaping (rank) —
    if fitness_shaping_fn is None:

        def _rank(_pop, f, *_):  # type: ignore
            return jnp.argsort(jnp.argsort(-f)).astype(jnp.float32)

        fitness_shaping_fn = _rank  # type: ignore
    elif fitness_shaping_fn.__code__.co_argcount == 1:  # type: ignore
        user_fn = fitness_shaping_fn  # type: ignore

        def _wrap(_pop, f, *_):  # type: ignore
            return user_fn(f)

        fitness_shaping_fn = _wrap  # type: ignore

    # — metrics fallback —
    if metrics_fn is None:

        def metrics_fn(*args, **kwargs):  # type: ignore
            return {}

    # — latent strategy initialised around given genome —
    latent_mean = jnp.asarray(init_genome, jnp.float32) / search_scale
    es = CMA_ES(pop_size, latent_mean, fitness_shaping_fn, metrics_fn)
    params = es.default_params

    # RNG & state
    key = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))
    key, sub = jax.random.split(key)
    state = es.init(sub, latent_mean, params)

    best_reward: float = -jnp.inf
    best_theta: np.ndarray = np.asarray(init_genome)

    for g in tqdm(range(1, generations + 1)):
        key, k_ask, k_tell = jax.random.split(key, 3)
        latent_pop, state = es.ask(k_ask, state, params)
        phenotypes = np.asarray(latent_pop) * search_scale

        # fitness
        futs = [
            _fit_worker.remote(p, env, prob_type, interval, episodes)
            for p in phenotypes
        ]
        rewards = np.asarray(ray.get(futs), dtype=np.float32)

        # tell with *positive* rewards (rank shaping already converts)
        state, _ = es.tell(k_tell, latent_pop, rewards, state, params)

        # best tracking
        idx_best = int(rewards.argmax())
        if rewards[idx_best] > best_reward:
            best_reward = float(rewards[idx_best])
            best_theta = phenotypes[idx_best].copy()

        write_worm_to_csv(
            csv_log,
            WormConnectome(best_theta, all_neuron_names),
            max_rows=generations,
        )

        if g % 100 == 0:
            cur_sig = getattr(state, "sigma", jnp.nan)
            print(f"Gen {g:4d} | best: {best_reward:.3f} | σ={float(cur_sig):.4f}")

        # restart if σ collapses
        cur_sig = getattr(state, "sigma", None)
        if cur_sig is not None and cur_sig < 1e-3:
            key, sub = jax.random.split(key)
            state = es.init(sub, latent_mean, params)
            print("[restart] σ reset – continuing …")

    ray.shutdown()
    return best_theta
