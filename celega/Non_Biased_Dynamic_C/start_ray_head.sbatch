#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=ray_head.out
#SBATCH --error=ray_head.err
#SBATCH --time=4:00:00  # Adjust the time limit as needed
#SBATCH --partition=normal  # Adjust the partition as needed
#SBATCH --nodes=1
#SBATCH --mem=80G 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Adjust the number of CPUs as needed
#SBATCH --mail-type=END
#SBATCH --mail-user=miles.churchland@upf.edu

module load nomad/v.4.4.0-Miniconda3-4.9.2
source activate nomad_env  # Activate your conda environment if needed

# Set Ray memory limits
export RAY_MEMORY_LIMIT=50000000000  # Set the memory limit to 50G in bytes

# Start the Ray head node
ray start --head --node-ip-address=$(hostname) --port=6379 \
          --memory=$RAY_MEMORY_LIMIT --object-store-memory=$RAY_MEMORY_LIMIT

# Run your Python script
python main.py