import os
import numpy as np
import ray
from Worm_Env.celegan_env import WormSimulationEnv
from Worm_Env.weight_dict import muscles, muscleList, mLeft, mRight, all_neuron_names
from util.write_read_txt import read_arrays_from_csv_pandas
from matplotlib import pyplot as plt
from graphs.Graph_path_over_gen import Genetic_Dyn_Algorithm

def evaluate_candidate(candidate_weights, env, food_pattern, training_interval, total_episodes):
    """
    Evaluates a candidate genome using your provided evaluate_fitness_ray method.
    Returns the fitness (food eaten).
    """
    # Call the remote evaluation method with the food pattern set to the ngon value.
    result = ray.get(
        Genetic_Dyn_Algorithm.evaluate_fitness_ray.remote(
            candidate_weights,
            all_neuron_names,
            env,
            [food_pattern],
            mLeft,
            mRight,
            muscleList,
            muscles,
            training_interval,
            total_episodes
        )
    )
    return result[0]  # The fitness value (food eaten)

def plot_ngon_performance(csv_files, training_interval, total_episodes=10):
    TITLE_FSIZE   = 24   # subplot titles
    LABEL_FSIZE   = 22   # axis labels
    TICK_FSIZE    = 22   # tick labels
    SUPTITLE_FSIZE = 28  # big title

    
    ngon_list = []
    start_perf = []
    end_perf = []


    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found, skipping.")
            continue
        
        genomes = read_arrays_from_csv_pandas(os.path.join("24hr",csv_file))
        if len(genomes) == 0:
            print(f"No genomes in {csv_file}, skipping.")
            continue

        try:
            base = os.path.basename(csv_file)
            ngon = int(base.replace("array", "").replace(".csv", ""))
        except Exception as e:
            print(f"Error extracting ngon from {csv_file}: {e}")
            continue
        
        ngon_list.append(ngon)

        # Use the first genome (start candidate) and the last genome (end candidate)
        start_genome = np.array(genomes[0], dtype=float)
        end_genome = np.array(genomes[-1], dtype=float)

        # Create separate environments for evaluation
        env_start = WormSimulationEnv()
        env_end = WormSimulationEnv()

        perf_start = evaluate_candidate(start_genome, env_start, food_pattern=ngon, 
                                        training_interval=training_interval, total_episodes=total_episodes)
        perf_end = evaluate_candidate(end_genome, env_end, food_pattern=ngon, 
                                      training_interval=training_interval, total_episodes=total_episodes)

        start_perf.append(perf_start)
        end_perf.append(perf_end)

    # Order the results by ngon value
    sorted_indices = sorted(range(len(ngon_list)), key=lambda i: ngon_list[i])
    sorted_ngon = [ngon_list[i] for i in sorted_indices]
    sorted_start = [start_perf[i] for i in sorted_indices]
    sorted_end = [end_perf[i] for i in sorted_indices]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # top bar-chart
    axs[0].bar(sorted_ngon, sorted_start)
    axs[0].set_title("Performance at Start of Training", fontsize=TITLE_FSIZE)
    axs[0].set_ylabel("Fitness (Food Eaten)", fontsize=LABEL_FSIZE)
    axs[0].tick_params(axis="both", labelsize=TICK_FSIZE)

    # bottom bar-chart
    axs[1].bar(sorted_ngon, sorted_end)
    axs[1].set_title("Performance at End of Training", fontsize=TITLE_FSIZE)
    axs[1].set_xlabel("Ngon (Food Pattern)", fontsize=LABEL_FSIZE)
    axs[1].set_ylabel("Fitness (Food Eaten)", fontsize=LABEL_FSIZE)
    axs[1].tick_params(axis="both", labelsize=TICK_FSIZE)

    plt.suptitle("Ngon Performance Comparison", fontsize=SUPTITLE_FSIZE, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.92])   # leave space for the suptitle
    plt.savefig("ngon_performance_barcharts.png")
    ray.shutdown()