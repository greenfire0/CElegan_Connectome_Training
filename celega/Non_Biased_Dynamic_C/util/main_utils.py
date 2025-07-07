import numpy as np
from typing import Dict
from Worm_Env.celegan_env import WormSimulationEnv
import ray
import multiprocessing
# Genetic Algorithm Variants
from Algorithms.Evolutionary_Algorithm import Genetic_Dyn_Algorithm as GD_EA
from Algorithms.Hybrid_NOMAD import Genetic_Dyn_Algorithm as GD_EA_Nomad
from Algorithms.PNOMAD_Time_Analysis import Genetic_Dyn_Algorithm as GD_PureNomad
from Algorithms.RandParam_50_NOMAD import Genetic_Dyn_Algorithm as GD_RandomNomad
from Algorithms.EVO_NOMAD import Genetic_Dyn_Algorithm as GD_EVO
from Algorithms.OaI_es import train_openai_es
from Algorithms.CMA_ES import train_cma_es
# Graphs
from graphs.Graph_pos_over_time import Genetic_Dyn_Algorithm as GD_Pos
from graphs.Graph_fitness_over_time import Genetic_Dyn_Algorithm as GD_Graph
from graphs.Graph_path_over_gen import Genetic_Dyn_Algorithm as GD_PathGen
from graphs.wpi import search_connection_impacts, calc_simular
from graphs.graphing import graph_results, graph_trained_worms, graph_agg
from graphs.Graph_fitness_over_time_old import Genetic_Dyn_Algorithm as GD_Graph_Old
from graphs.graph_video_ngons import Genetic_Dyn_Video
import numpy.typing as npt
from util.write_read_txt import delete_arrays_csv_if_exists



def select_ga_class(config:Dict):
    """
    Selects which Genetic_Dyn_Algorithm variant to use based on the config.
    Now the dictionary keys are more descriptive.
    """
    variant_map = {
        "EVO_NOMAD":                      GD_EVO,
        "graph_positions_over_time":      GD_Pos,      # old "pos"
        "graph_path_quartile_evolution":  GD_PathGen,  # new "path"
        "graph_fitness_over_time":        GD_Graph,    # old "graph"
        "graph_fitness_over_time_legacy": GD_Graph_Old,# old "graph_old"
        "pure_nomad_algorithm":           GD_PureNomad,# old "pure_nomad"
        "random_nomad_algorithm":         GD_RandomNomad,  # old "random_nomad"
        "standard_evolutionary_algorithm":GD_EA,       # old "ea"
        "nomad_evolutionary_algorithm":   GD_EA_Nomad, # old "ea_nomad"
    }
    return variant_map.get(config["ga_variant"], GD_PureNomad)


def calculate_worm_suffering_index(config:Dict,values_list:npt.NDArray[np.float64],length:int):
    """Calculates a 'worm suffering index' by searching for connection impacts, etc."""
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
    #ci_diff = ci_3 - ci_5 was used in the past for something now not used
    calc_simular(ci_3, ci_5)


def test_last_generations(config:Dict,values_list:npt.NDArray[np.float64],length:int):
    """Tests and prints the last generations from arrays.csv (if you store them)."""
    env = WormSimulationEnv()
    ga_instance = GD_Graph(
        population_size=config["population_size"],
        pattern=config["food_patterns"],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"],
        genome=values_list,
        matrix_shape=length
    )
    ga_instance.run_and_print_last_generations(env, '24hr')


def graph_training_results(config:Dict,values_list:npt.NDArray[np.float64]):
    """Graphs results of a single worm's training."""
    csv_name = "5"
    graph_results(config["path"], csv_name, values_list)


def graph_trained_population(config:Dict,values_list:npt.NDArray[np.float64]):
    """Graphs the results of trained worms at the end of training."""
    graph_trained_worms(base_path=config["path"], values_list=values_list)


def graph_aggregates(config:Dict,values_list:npt.NDArray[np.float64]):
    """Graphs aggregate results from multiple runs."""
    graph_agg(base_path=config["path"], values_list=values_list)


# NEW: Function to graph quartiles from arrays.csv using Graph_path_over_gen
def graph_quartiles(config:Dict,values_list:npt.NDArray[np.float64],length:int):
    """
    Produces a 2x2 plot of:
      - 0th generation
      - 1/4 generation
      - 3/4 generation
      - final generation
    by calling a method like 'run_single_csv_quartiles' on the GD_PathGen class.
    """
    print("Plotting quartiles from arrays.csv ...")
    env = WormSimulationEnv()

    # We specifically use the 'graph_path_quartile_evolution' class here
    quartile_ga = GD_PathGen(
        population_size=1,  # We only care about the 4 special connectomes
        pattern=config["food_patterns"],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"],
        genome=values_list,
        matrix_shape=length
    )

    # Ensure 'run_single_csv_quartiles' is defined in Graph_path_over_gen.Genetic_Dyn_Algorithm
    quartile_ga.run_single_csv_quartiles(env, arrays_csv="ES_worms.csv")



def clean_environment():
    """Cleans the environment by deleting arrays.csv if it exists."""
    print("Clearing Environment...")
    delete_arrays_csv_if_exists()


def run_genetic_algorithm(config:Dict,values_list:npt.NDArray[np.float64],shape:int):
    """Runs the genetic algorithm to find the best weight matrix."""
    print("Running Genetic Algorithm...")
    env = WormSimulationEnv()

    GA_Class = select_ga_class(config)
    ga = GA_Class(
        population_size=config["population_size"],
        pattern=config["food_patterns"],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"],
        genome=values_list,
        matrix_shape=shape
    )
    best_weight_matrix = ga.run(env, config["generations"])
    print("Best weight matrix found:", best_weight_matrix)



def polygon_test(config:Dict,values_list:npt.NDArray[np.float64],length:int):
    """
    Runs the pure nomad algorithm with food patterns 6 to 10.
    Each pattern runs for 40 generations before switching to the next.
    """
    print("Starting Polygon Test Experiment...")

    env = WormSimulationEnv()
    GA_Class = GD_PureNomad 

    for food_pattern in range(6, 11):  # Iterate from 6 to 10 (inclusive)
        print(f"Running with food pattern: {food_pattern}")

        ga = GA_Class(
            population_size=config["population_size"],
            pattern=[food_pattern],  # Change food pattern dynamically
            total_episodes=config["total_episodes"],
            training_interval=config["training_interval"],
            genome=values_list,
            matrix_shape=length
        )
        best_weight_matrix = ga.run(env=env, generations=config["generations"],batch_size=32,\
                                    filename=str(config["ga_variant"])+str(food_pattern))  # Run for 40 generations

        print(f"Completed training for food pattern {food_pattern}")
        print("Best weight matrix found:", best_weight_matrix)


def graph_video_ngons(config):
    env = WormSimulationEnv()
    gd_video = Genetic_Dyn_Video(
        population_size=1,
        pattern=[3, 4, 5, 6, 7, 8],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"],
    )
    gd_video.run_video_simulation(env, output_video=config.get("video_output", "food_collection_video.mp4"))

def graph_image_ngons(config):
    env = WormSimulationEnv()
    sim = Genetic_Dyn_Video(
        population_size=1,
        pattern=[3, 4, 5, 6, 7, 8],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"],
    )
    sim.run_image_simulation()


def run_openai_es(config: Dict,
                  genome: np.ndarray,
                  genome_len: int):
    """
    Launch OpenAI-ES search seeded with the real connectome + random worms.
    Writes every improvement to ES_worms.csv and saves the champion to best_es.npy.
    """
    # WormSimulationEnv currently takes only num_worms
    env = WormSimulationEnv(num_worms=1)

    best = train_openai_es(
        env=env,
        init_genome=genome,
        generations=config["generations"],
        pop_size=config["population_size"],
        sigma=config.get("sigma", 0.1),
        lr=config.get("lr", 0.02),
        prob_type=config["food_patterns"],
        interval=config["training_interval"],
        episodes=config["total_episodes"],
        csv_log="ES_worms",
    )

    print(f"OpenAI-ES finished → {best}")

def run_cma_es(config: Dict,
                  genome: np.ndarray,
                  genome_len: int):
    """
    Launch OpenAI-ES search seeded with the real connectome + random worms.
    Writes every improvement to ES_worms.csv and saves the champion to best_es.npy.
    """
    # WormSimulationEnv currently takes only num_worms
    env = WormSimulationEnv(num_worms=1)

    best = train_cma_es(
        env=env,
        init_genome=genome,
        generations=config["generations"],
        pop_size=config["population_size"],
        prob_type=config["food_patterns"],
        interval=config["training_interval"],
        episodes=config["total_episodes"],
        csv_log="CMAES_worms",
    )

    print(f"OpenAI-ES finished → {best}")