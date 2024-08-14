# Caenorhabditis Elegans Connectome Training

## Overview

This project focuses on training *Caenorhabditis elegans* (C. elegans) connectomes using an innovative hybrid approach. By combining the NOMAD (Nonlinear Optimization with the Mesh Adaptive Direct Search) algorithm with traditional evolutionary algorithms, the goal is to enhance C. elegans connectomes for specific tasks and evaluate the changes during training.

## Key Features

- **Hybrid Training Approach**: Utilizes both NOMAD and evolutionary algorithms for optimizing C. elegans connectomes.
- **Visualization**: Includes tools to visualize the training process, results, and state of trained connectomes.
- **Simulation Environment**: Features a custom environment to simulate and train C. elegans connectomes.

## Installation

**Clone the Repository**

   ```bash
   git clone https://github.com/greenfire0/C.-Elegan-bias-Exploration.git
   cd C.-Elegan-bias-Exploration
   pip install -r requirements.txt
 ```
## Usage

Configuration: Adjust parameters such as population size, type of genetic algorithm, number of generations, training intervals, and food patterns according to your needs.

Running the Algorithm: Execute the genetic algorithm training process and evaluate the performance make sure run_gen is set to 1 in main.py. Graphing of the simulations is off by default, if you want to see the worm see run_from_genes.py. All algorithms will produce a csv with the weight matrix of the best connectome for each generation in training.

Graphing: Generate and view graphs for individual worms, trained connectomes, and aggregated results, some file shuffling is required. See also Graph_fitness_over_time.py.


## Contact
Email: Mileschurchland@gmail.com



This software is licensed under the MIT License for non-commercial use. 
