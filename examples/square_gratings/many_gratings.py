import meep as mp
import numpy as np
from meep.materials import Au, Ag

import os, sys
sys.path.append('../metamaterials')
from metamaterials import meep_retrieval as mr

lambda_min = 1
lambda_max = 4
nfreqs = 100
pol = mp.Ex
t = 0.020 # Thickness of metal
dir = 'results/test2'
index = 3

for a in np.linspace(2, 4, 10):
    for fact in np.linspace(0.4, 0.8, 8):
        d = fact*a
        geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, -t/2), material=mp.Medium(index=index))]
        sim = mr.Simulation(resolution=60, cell=a, fmin=1/lambda_max, fmax=1/lambda_min, nfreqs=nfreqs, pol=pol, directory=dir)
        sim.set_geometry(geometry, mp.Medium(index=index))
        sim.run()
        filename = 'a_%.2f_d_%.2f_n_%d.txt' % (a, d, index)
        [S11, S21] = sim.get_s_params(do_plot=True, plot='wl', plot_title='Square grating w a = %.2f, d = %.2f and n=%.2f' % (a, d, index), filename=filename)
