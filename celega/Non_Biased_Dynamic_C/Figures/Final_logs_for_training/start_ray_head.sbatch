#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=ray_head.out
#SBATCH --error=ray_head.err
#SBATCH --time=25:00:00  # Adjust the time limit as needed
#SBATCH --nodes=1
#SBATCH --mem=80G 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Adjust the number of CPUs as needed
#SBATCH --mail-type=END
#SBATCH --mail-user=miles.churchland@upf.edu

module load nomad/v.4.4.0-Miniconda3-23.9.0-0
source activate nomad_env  # Activate your conda environment if needed


# Run your Python script
python main.py
