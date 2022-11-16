import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import sys, os

from meep.materials import Ag

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from meep_metamaterials import metamaterials as mm

l = .320
w = .090
a = .450
gap = .070
t = .020

geometry = [
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, -t/2), material=Ag),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(l/2-w/2, 0, -t/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(-l/2+w/2, 0, -t/2), material=Ag),
    mp.Block(size=mp.Vector3(gap, w, t), center=mp.Vector3(0, l/2-w/2, -t/2), material=mp.Medium(epsilon=1)), # Gap
]

freqs = np.linspace(0.3, 0.8, 50)
sep = 0.05

for layers in [4,5,6,7]:
    sim = mm.MetamaterialSimulation(period=a, geometry=geometry, freq_range=freqs, resolution=100, layers=layers, period_z=sep)

    sim.plot2D(plane='xz', save=True, filename='plot2D_'+ str(layers) +'.png')

    sim.run()

    [S11, S21] = sim.get_s_parameters()

    # Save S11 and S21 to file
    np.savetxt('examples/srr/tmp/S11_'+str(layers)+'_layers.txt', S11)
    np.savetxt('examples/srr/tmp/S21_'+str(layers)+'_layers.txt', S21)

    sim.sim.reset_meep()