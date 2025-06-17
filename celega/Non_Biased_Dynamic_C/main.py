import os
import numpy as np
import numpy.typing as npt
from Worm_Env.celegan_env import WormSimulationEnv

# Genetic Algorithm Variants
from genetic_dynamic.Genetic_Dynamic_TRAINING import Genetic_Dyn_Algorithm as GD_EA
from genetic_dynamic.Genetic_Dynamic_TRAINING_nomad import Genetic_Dyn_Algorithm as GD_EA_Nomad
from genetic_dynamic.Genetic_Dynamic_train_god import Genetic_Dyn_Algorithm as GD_PureNomad
from genetic_dynamic.Random_TRAINING_nomad import Genetic_Dyn_Algorithm as GD_RandomNomad

# Graphs
from graphs.graph_ngon_performance import plot_ngon_performance
from graphs.Graph_pos_over_time import Genetic_Dyn_Algorithm as GD_Pos
from graphs.Graph_fitness_over_time import Genetic_Dyn_Algorithm as GD_Graph
from graphs.Graph_path_over_gen import Genetic_Dyn_Algorithm as GD_PathGen
from graphs.wpi import search_connection_impacts, graph_wsi, calc_simular
from graphs.graphing import graph, graph2, graph_results, graph_trained_worms, graph_agg
from graphs.Graph_fitness_over_time_old import Genetic_Dyn_Algorithm as GD_Graph_Old
   
from Worm_Env.weight_dict import dict
from util.dist_dict_calc import dist_calc
from util.movie import compile_images_to_video
from util.findmotor_ind import find_motor_ind, get_indicies_to_change
from util.read_from_xls import combine_neuron_data
from util.write_read_txt import read_last_array_from_csv, read_arrays_from_csv_pandas, delete_arrays_csv_if_exists
from util.main_utils import run_genetic_algorithm,polygon_test,clean_environment,\
    graph_quartiles,graph_aggregates,select_ga_class,calculate_worm_suffering_index,\
        test_last_generations,graph_training_results,graph_trained_population,graph_video_ngons
os.environ["RAY_DEDUP_LOGS"] = "0"

# =========================================
# Configuration and Global Parameters
# =========================================
config = {
    "population_size": 64,
    "generations": 10,
    "training_interval": 250,
    "total_episodes": 1,
    "food_patterns": [5],
    "path": "/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C",

    # Execution Flags
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
    # "standard_evolutionary_algorithm", "nomad_evolutionary_algorithm"
    "ga_variant": "pure_nomad_algorithm",

    # Turn on quartile plotting from arrays.csv if desired

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
