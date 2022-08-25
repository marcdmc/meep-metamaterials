"""Run single iteration of square grating"""
import meep as mp
import numpy as np
from meep.materials import Au, Ag
import time
import os, sys
sys.path.append('../metamaterials')
from metamaterials import meep_retrieval as mr

lambda_min = 1
lambda_max = 4
nfreqs = 100
pol = mp.Ex
t = 0.020 # Thickness of metal
dir = 'results/basic'

a = 2.3
d = 0.45*a


tic = time.perf_counter()

geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, -t/2), material=Ag)]
sim = mr.Simulation(resolution=60, cell=a, fmin=1/lambda_max, fmax=1/lambda_min, nfreqs=nfreqs, pol=pol, directory=dir)
sim.set_geometry(geometry, Ag)
sim.run()
[S11, S21] = sim.get_s_params(plot='wl', plot_title='square a = %.2f, d = %.2f' % (a, d))

toc = time.perf_counter()
total_time = toc-tic
print("time for code block")
print(total_time)

mp.all_wait()

# Open file and append time
with open('time.txt', 'a') as f:
    f.write(str(total_time) + '\n')
