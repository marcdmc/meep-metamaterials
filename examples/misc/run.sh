#!/bin/bash -l

# Slurm sbatch options
#SBATCH -n 8

# Loading the required module
source /etc/profile
module load anaconda/2022b
source activate pmp

# Run the script
mpirun -n 8 python examples/misc/spirals/spirals.py > examples/misc/output2.txt
# mpirun -n 8 python examples/misc/spirals/spirals.py > examples/misc/spiral/output.txt