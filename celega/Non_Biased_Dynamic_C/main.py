import os
import numpy as np
import numpy.typing as npt
from Worm_Env.weight_dict import dict
from graphs.graph_ngon_performance import plot_ngon_performance
from util.main_utils import run_genetic_algorithm,polygon_test,clean_environment,\
    graph_quartiles,graph_aggregates,calculate_worm_suffering_index, run_openai_es,\
        test_last_generations,graph_training_results,graph_trained_population,graph_video_ngons
os.environ["RAY_DEDUP_LOGS"] = "0"

# =========================================
# Configuration and Global Parameters
# =========================================
config = {
    "population_size": 64,
    "generations": 0,
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
    "graph_quartiles": 1,
    "polygon_test":0,
    "graph_ngon_performance": 0,
    "graph_video_ngons": 0,


    # More descriptive name in ga_variant:
    # "graph_positions_over_time", "graph_path_quartile_evolution",
    # "graph_fitness_over_time", "graph_fitness_over_time_legacy",
    # "pure_nomad_algorithm", "random_nomad_algorithm",
    # "standard_evolutionary_algorithm", "nomad_evolutionary_algorithm"
    # EVO_NOMAD, OPENAI_ES
    "ga_variant": "OPENAI_ES",
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
                "generations": 200,
            })
            run_openai_es(config, connectome_weights, length)
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
        graph_video_ngons(config,)
    if config.get("polygon_test", 0):
        polygon_test(config,connectome_weights,length)
    # Additional testing mode logic
    if config["testing_mode"]:
        pass


if __name__ == "__main__":
    main(config)
