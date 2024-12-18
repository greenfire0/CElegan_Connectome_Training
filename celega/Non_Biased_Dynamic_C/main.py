import os
import numpy as np

from Worm_Env.celegan_env import WormSimulationEnv

# Genetic Algorithm Variants
from Genetic_Dynamic_TRAINING import Genetic_Dyn_Algorithm as GD_EA
# ^ Evolutionary algorithm training (standard evolutionary approach)

from Genetic_Dynamic_TRAINING_nomad import Genetic_Dyn_Algorithm as GD_EA_Nomad
# ^ Evolutionary with NOMAD (Nomad-based optimization integrated into evolutionary)

from Graph_fitness_over_time import Genetic_Dyn_Algorithm as GD_Graph
# ^ Graphing version: Outputs a single fitness-over-time graph

from Graph_fitness_over_time_old import Genetic_Dyn_Algorithm as GD_Graph_Old
# ^ Older graphing version: Outputs fitness-over-time and Euclidean distance from original connectome

from Genetic_Dynamic_train_god import Genetic_Dyn_Algorithm as GD_PureNomad
# ^ "Train God" - the best method, referred to as pure NOMAD approach


from Random_TRAINING_nomad import Genetic_Dyn_Algorithm as GD_RandomNomad
# ^ Random Training NOMAD: NOMAD approach with 2 randomly selected mutations every 4 gens

from Graph_pos_over_time import Genetic_Dyn_Algorithm as GD_Pos
# ^ Graph position over time variant (Tracks positional changes over generations)

from wpi import search_connection_impacts, graph_wsi, calc_simular
from Worm_Env.weight_dict import dict
from graphing import graph, graph2, graph_results, graph_trained_worms, graph_agg
from util.dist_dict_calc import dist_calc
from util.movie import compile_images_to_video
from util.findmotor_ind import find_motor_ind, get_indicies_to_change
from util.read_from_xls import combine_neuron_data
from util.write_read_txt import read_last_array_from_csv, read_arrays_from_csv_pandas, delete_arrays_csv_if_exists

# =========================================
# Configuration and Global Parameters
# =========================================
config = {
    "population_size": 64,
    "generations": 100,
    "training_interval": 250,
    "total_episodes": 1,  # Unless environment changes, this can be 1
    "food_patterns":  [3],
    "path": "/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C",
    
    # Execution Flags
    "clean_env": 0,
    "freeze_indicies": 0,
    "run_gen": 0,
    "worm_suffering_index": 0,
    "graphing": 0,
    "graph_best": 0,
    "graphing_agg": 0,
    "test_last_ten":1,
    "testing_mode": 0,
    
    # Default GA Variant to run if run_gen = 1
    "ga_variant": "pure_nomad"  
    # Could be: 'ea', 'ea_nomad', 'graph', 'graph_old', 'pure_nomad', 'wpi', 'random_nomad', 'pos'
    # Adjust this as you like, or even select based on other config flags.
}


frozen_indices = []
values_list = []


for sub_dict in dict.values():
    values_list.extend(sub_dict.values())
values_list=np.array(values_list)
length = (len(values_list))

def select_ga_class(config):
    """
    Selects which Genetic_Dyn_Algorithm variant to use based on the config.
    If worm_suffering_index is set, we choose GD_WPI, otherwise we use the 
    default from 'ga_variant'.
    """

    variant_map = {

        "graph": GD_Graph,
        "pos": GD_Pos,
        "graph_old": GD_Graph_Old,

        "pure_nomad": GD_PureNomad,
        "random_nomad": GD_RandomNomad,
        "ea": GD_EA,
        "ea_nomad": GD_EA_Nomad,

    }
    return variant_map.get(config["ga_variant"], GD_PureNomad)


def clean_environment():
    """Cleans the environment by deleting arrays.CSV if it exist."""
    print("Clearing Environment...")
    delete_arrays_csv_if_exists()


def run_genetic_algorithm(config):
    """Runs the genetic algorithm to find the best weight matrix."""
    print("Running Genetic Algorithm...")
    env = WormSimulationEnv()

    GA_Class = select_ga_class(config)
    ga = GA_Class(
        config["population_size"],
        config["food_patterns"],
        config["total_episodes"],
        config["training_interval"],
        values_list,
        matrix_shape=length
    )
    best_weight_matrix = ga.run(env, config["generations"])
    print("Best weight matrix found:", best_weight_matrix)


def calculate_worm_suffering_index(config):
    """
    Calculates and compares connection impacts on the worm's behavior under 
    different food patterns, then calculates similarity.
    """
    env = WormSimulationEnv()
    ci_3 = search_connection_impacts(
        original_genome=values_list,
        matrix_shape=length,
        env=env,
        prob_type=[3],
        interval=config["training_interval"],
        episodes=config["total_episodes"]
    )
    ci_5 = search_connection_impacts(
        original_genome=values_list,
        matrix_shape=length,
        env=env,
        prob_type=[5],
        interval=config["training_interval"],
        episodes=config["total_episodes"]
    )
    ci_diff = ci_3 - ci_5
    calc_simular(ci_3, ci_5)


def test_last_generations(config):
    """Tests and prints the last generations of the genetic algorithm results."""
    env = WormSimulationEnv()
    ga_instance = GD_Graph(
    population_size=config["population_size"],
    pattern=config["food_patterns"],
    total_episodes=config["total_episodes"],
    training_interval=config["training_interval"],
    genome=values_list,
    matrix_shape=length)
    ga_instance.run_and_print_last_generations(env, '24hr')


def graph_training_results(config):
    """Graphs results of a single worm's training."""
    csv_name = "5"
    graph_results(config["path"], csv_name, values_list)


def graph_trained_population(config):
    """Graphs the results of trained worms at the end of training."""
    graph_trained_worms(base_path=config["path"], values_list=values_list)


def graph_aggregates(config):
    """Graphs aggregate results from multiple runs."""
    graph_agg(base_path=config["path"], values_list=values_list)


# =========================================
# Main Execution Flow
# =========================================

def main(config):
    # Clean environment if requested
    if config["clean_env"]:
        clean_environment()

    # Run genetic algorithm if requested
    if config["run_gen"]:
        run_genetic_algorithm(config)

    # Compute worm suffering index if requested
    if config["worm_suffering_index"]:
        calculate_worm_suffering_index(config)

    # Test the last ten generations if requested
    if config["test_last_ten"]:
        test_last_generations(config)

    # Graph results if requested
    if config["graphing"]:
        graph_training_results(config)

    # Graph best worms if requested
    if config["graph_best"]:
        graph_trained_population(config)

    # Graph aggregate data if requested
    if config["graphing_agg"]:
        graph_aggregates(config)

    # Additional testing mode logic
    if config["testing_mode"]:
        # Add any testing logic here
        pass


if __name__ == "__main__":
    main(config)