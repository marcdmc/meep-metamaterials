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
t = 0.020 # Thickness of metal
dir = 'results/many_gratings_mark2/shifted'

for a in np.linspace(0.5, 2, 10):
    for fact in np.linspace(.1, .9, 10):
        d = fact*a
        metal = Ag
        geometry = [
            mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(-a/2, -a/2, -t/2), material=metal),
            mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(-a/2, a/2, -t/2), material=metal),
            mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(a/2, 0, -t/2), material=metal),
            mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(a/2, a, -t/2), material=metal),
        ]
        sim = mr.Simulation(resolution=60, cell=2*a, fmin=1/lambda_max, fmax=1/lambda_min, nfreqs=nfreqs, pol=pol, directory=dir)
        sim.set_geometry(geometry, Ag)
        sim.run()
        [S11, S21] = sim.get_s_params(plot='wl', plot_title='shifted a = %.2f, d = %.2f' % (a, d))
