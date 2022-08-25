#!/bin/bash -l

# Important: this should be run from the base conda environment

# Slurm sbatch options
#SBATCH -n 8

# Loading the required module
source /etc/profile
module load anaconda/2022b
source activate pmp

# Run the script
mpirun -n 8 python examples/srr/srr.py > output.txt
