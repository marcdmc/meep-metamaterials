#!/bin/bash -l

# Slurm sbatch options
#SBATCH -n 8

# Loading the required module
source /etc/profile
module load anaconda/2022b
source activate pmp

# Run the script
mpirun -n 8 python examples/mpb/Tutorial/point_defect.py > examples/mpb/Tutorial/output.txt