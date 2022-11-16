#!/bin/bash -l

# Slurm sbatch options
#SBATCH -n 8
#SBATCH -N 1

# Loading the required module
source /etc/profile
module load anaconda/2022b
source activate pmp

# Run the script
mpirun -n 8 python examples/square_gratings/square_grating_3D.py
