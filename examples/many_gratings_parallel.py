import meep as mp
# import meep_retrieval as mr
import numpy as np
from meep.materials import Au, Ag
from mpi4py import MPI

import os, sys
sys.path.append('../metamaterials')
from metamaterials import meep_retrieval as mr

## Simulation definition
# Frequency range
lambda_min = 1
lambda_max = 4
nfreqs = 100
pol = mp.Ex # Polarization
t = 0.020 # Thickness of metal layer
dir = 'results' # Directory to save results

## Range of parameters to sweep
fact = np.linspace(0.4, 0.8, 8) # Size factor

## Parallelization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_per_rank = len(fact) // size
lower_bound = rank * num_per_rank
upper_bound = (rank + 1) * num_per_rank
print("This is processor ", rank, "and I am summing numbers from", lower_bound," to ", upper_bound - 1, flush=True)

for a in np.linspace(0.75, 2.5, 10):
    for f in fact[lower_bound:upper_bound]:
        d = f*a
        print('a = %.2f, d = %.2f' % (a, d))
        print('!!!!!!!!!!!!!!!!!!!!!!')
        geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, -t/2), material=Ag)]
        sim = mr.Simulation(resolution=60, cell=a, fmin=1/lambda_max, fmax=1/lambda_min, nfreqs=nfreqs, pol=pol, directory=dir)
        sim.set_geometry(geometry, Ag)
        sim.run()
        [S11, S21] = sim.get_s_params(plot='wl', plot_title='square a = %.2f, d = %.2f' % (a, d))

comm.Barrier()