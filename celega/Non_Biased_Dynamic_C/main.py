import os
import ray
import time
import numpy as np
import multiprocessing
import numpy.typing as npt
from Worm_Env.weight_dict import dict
from graphs.graph_ngon_performance import plot_ngon_performance
from graphs.chemotax_graph import Genetic_Dyn_Algorithm
from Worm_Env.celegan_env import ChemotaxisPeakEnv
from util.main_utils import run_genetic_algorithm,polygon_test,clean_environment,\
    graph_quartiles,graph_aggregates,calculate_worm_suffering_index, run_openai_es,\
        run_cma_es,graph_image_ngons,\
        test_last_generations,graph_training_results,graph_trained_population,graph_video_ngons
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["DISABLE_TQDM"] = "1"

# =========================================
# Configuration and Global Parameters
# =========================================
config = {
    "population_size": 64,
    "generations": 14*2, ### 21 min for es worm
    "training_interval": 250,
    "total_episodes": 1,
    "food_patterns": [5],
    "path": "/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C",
    "clean_env": 0,
    "freeze_indicies": 0, ## this all needs documentation
    "run_gen": 1,
    "worm_suffering_index": 0,
    "graphing": 0,
    "graph_best": 0,
    "graphing_agg": 0,
    "test_last_ten": 0,
    "testing_mode": 0,
    "graph_quartiles": 0,
    "polygon_test":0,
    "graph_ngon_performance": 0,
    "graph_video_ngons": 0,


    # More descriptive name in ga_variant:
    # "graph_positions_over_time", "graph_path_quartile_evolution",
    # "graph_fitness_over_time", "graph_fitness_over_time_legacy",
    # "pure_nomad_algorithm", "random_nomad_algorithm",
    # "graph_fitness_over_time_legacy", "nomad_evolutionary_algorithm"
    # EVO_NOMAD, OPENAI_ES, CMA_ES
    "ga_variant": "graph_fitness_over_time_legacy", ## evo nomad = bad 
    
    ##change order before graphing bigger text
}


frozen_indices = []
values_list = []
for sub_dict in dict.values():
    values_list.extend(sub_dict.values())
connectome_weights:npt.NDArray[np.float64] = np.array(values_list)
length = len(values_list)
#from graphs.heatmap_nomad import main
#main()
#from graphs.graph_video import GeneticDynVideo
from graphs.graph_video import GeneticDynVideo
GeneticDynVideo(patterns=[5], episodes=1, steps_per_episode=250).save_cover_frame()
#from graphs.fig5 import run
#run()
quit()
env = ChemotaxisPeakEnv(num_worms=1, sigma=220.0, drift_std=0.0)  # same API as before
ga = Genetic_Dyn_Algorithm(population_size=1, pattern=[0], total_episodes=10, training_interval=25)
ga.run(env, gen=0)
quit(); exit()
def main(config):
    # Clean environment if requested
    if config["clean_env"]:
        clean_environment()

    # Run genetic algorithm if requested
    if config["run_gen"]:
        if config.get("ga_variant") == "OPENAI_ES":
            config.update({
                "population_size": 512,
                "generations": 250,
            })
            run_openai_es(config, connectome_weights, length)
        elif config.get("ga_variant") == "CMA_ES":
            config.update({
                "population_size": None,
            })
            run_cma_es(config, connectome_weights, length)
        else:
            run_genetic_algorithm(config,connectome_weights,length)

    # Compute worm suffering index if requested
    if config["worm_suffering_index"]:
        calculate_worm_suffering_index(config,connectome_weights,length)

    # Test the last ten generations if requested
    if config["test_last_ten"]:
        test_last_generations(config,connectome_weights,length)

    # Graph results if requested
    if config["graphing"]:
        graph_training_results(config,connectome_weights)

    # Graph best worms if requested
    if config["graph_best"]:
        graph_trained_population(config,connectome_weights)

    # Graph aggregate data if requested
    if config["graphing_agg"]:
        graph_aggregates(config,connectome_weights)

    # NEW: plot quartiles from arrays.csv if requested
    if config.get("graph_quartiles", 0):
        graph_quartiles(config,connectome_weights,length)
    if config.get("graph_ngon_performance", 0):
        csv_files = [f"array{i}.csv" for i in range(3, 10)]
        plot_ngon_performance(csv_files, training_interval=config["training_interval"],
                              total_episodes=config["total_episodes"])
    if config.get("graph_video_ngons", 0):
       # graph_video_ngons(config,)
        graph_image_ngons(config)
    if config.get("polygon_test", 0):
        polygon_test(config,connectome_weights,length)
    # Additional testing mode logic
    if config["testing_mode"]:
        pass


if __name__ == "__main__":
    ray.init(
            ignore_reinit_error=True,
            object_store_memory=16 * 1024 * 1024 * 1024,
            num_cpus=11,
    )

    num_cpus=multiprocessing.cpu_count()
    #print(f"using {num_cpus} cpu's")

    algos = ["random_nomad_algorithm","OPENAI_ES","standard_evolutionary_algorithm",
             "nomad_evolutionary_algorithm","pure_nomad_algorithm"]
    for a in (algos):
        for _ in range (1):

            config.update({
                    "ga_variant": a,
                    "population_size": 64,
            })
            if a == "pure_nomad_algorithm":
                config.update({"generations": 14})
            if a == "OPENAI_ES":
                config.update({"generations": 250})
            if a == "random_nomad_algorithm":
                config.update({"generations": 100})
            if a == "standard_evolutionary_algorithm":
                config.update({"generations": 1300})
            if a == "nomad_evolutionary_algorithm":
                config.update({"generations": 190})
            if a == "CMA_ES":
                config.update({"generations": 190})
            t0 = time.perf_counter()      
            main(config)
            dt = time.perf_counter() - t0             
            print(f"[{config['ga_variant']}] {config['generations']} gens finished in {dt/60:.1f} min ({dt:.1f} s)")
