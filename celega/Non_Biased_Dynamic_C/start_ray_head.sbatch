#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=ray_head.out
#SBATCH --error=ray_head.err
#SBATCH --time=14:00:00  # Adjust the time limit as needed
#SBATCH --partition=normal  # Adjust the partition as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Adjust the number of CPUs as needed
#SBATCH --mail-type=END
#SBATCH --mail-user=miles.churchland@upf.edu

module load  nomad/v.4.4.0-Miniconda3-4.9.2
source activate nomad_env  # Activate your conda environment if needed

python main.py