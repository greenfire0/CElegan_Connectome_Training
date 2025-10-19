#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=ray_head.out
#SBATCH --error=ray_head.err
#SBATCH --time=02:00:00  # Adjust the time limit as needed
#SBATCH --nodes=1
#SBATCH --mem=32G 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Adjust the number of CPUs as needed
#SBATCH --mail-type=END
#SBATCH --mail-user=milesc@reed.edu

conda run -n cel python main.py