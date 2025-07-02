import os
import ray
import time
import numpy as np
import multiprocessing
import numpy.typing as npt
from Worm_Env.weight_dict import dict
from graphs.graph_ngon_performance import plot_ngon_performance
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
    "generations": 14, ### 21 min for es worm
    "training_interval": 250,
    "total_episodes": 1,
    "food_patterns": [5],
    "path": "/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C",
    "clean_env": 0,
    "freeze_indicies": 0, ## this all needs documentation
    "run_gen": 0,
    "worm_suffering_index": 0,
    "graphing": 0,
    "graph_best": 0,
    "graphing_agg": 0,
    "test_last_ten": 0,
    "testing_mode": 0,
    "graph_quartiles": 0,
    "polygon_test":0,
    "graph_ngon_performance": 0,
    "graph_video_ngons": 1,


    # More descriptive name in ga_variant:
    # "graph_positions_over_time", "graph_path_quartile_evolution",
    # "graph_fitness_over_time", "graph_fitness_over_time_legacy",
    # "pure_nomad_algorithm", "random_nomad_algorithm",
    # "graph_fitness_over_time_legacy", "nomad_evolutionary_algorithm"
    # EVO_NOMAD, OPENAI_ES, CMA_ES
    "ga_variant": "graph_positions_over_time", ## evo nomad = bad 
    
    ##change order before graphing bigger text
}


frozen_indices = []
values_list = []
for sub_dict in dict.values():
    values_list.extend(sub_dict.values())
connectome_weights:npt.NDArray[np.float64] = np.array(values_list)
length = len(values_list)



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
    main(config)



    """
    num_cpus=multiprocessing.cpu_count()
    print(f"using {num_cpus} cpu's")
    ray.init(
            ignore_reinit_error=True,
            object_store_memory=15 * 1024 * 1024 * 1024,
            num_cpus=num_cpus,
        )
    algos = ["standard_evolutionary_algorithm"]
    for a in (algos):
        if a == "pure_nomad_algorithm":  b = 20
        else: b= 30
        for _ in range (b):

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
            """