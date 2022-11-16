import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import sys, os

from meep.materials import Ag

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from meep_metamaterials import metamaterials as mm

pol = mp.Hx

l = .320
w = .090
a = .450
gap = .070
t = .020

sep = 0.05 # For two layers

geometry = [
    mp.Block(size=mp.Vector3(t, w, l), center=mp.Vector3(0, l/2-w/2, 0), material=Ag),
    mp.Block(size=mp.Vector3(t, w, l), center=mp.Vector3(0, -l/2+w/2, 0), material=Ag),
    mp.Block(size=mp.Vector3(t, l, w), center=mp.Vector3(0, 0, l/2-w/2), material=Ag),
    mp.Block(size=mp.Vector3(t, l, w), center=mp.Vector3(0, 0, -l/2+w/2), material=Ag),
    mp.Block(size=mp.Vector3(t, w, gap), center=mp.Vector3(0, l/2-w/2, 0), material=mp.Medium(epsilon=1)), # Gap

    mp.Block(size=mp.Vector3(t, w, l), center=mp.Vector3(0.05, l/2-w/2, 0), material=Ag),
    mp.Block(size=mp.Vector3(t, w, l), center=mp.Vector3(0.05, -l/2+w/2, 0), material=Ag),
    mp.Block(size=mp.Vector3(t, l, w), center=mp.Vector3(0.05, 0, l/2-w/2), material=Ag),
    mp.Block(size=mp.Vector3(t, l, w), center=mp.Vector3(0.05, 0, -l/2+w/2), material=Ag),
    mp.Block(size=mp.Vector3(t, w, gap), center=mp.Vector3(0.05, l/2-w/2, 0), material=mp.Medium(epsilon=1)), # Gap

    mp.Block(size=mp.Vector3(t, w, l), center=mp.Vector3(-0.05, l/2-w/2, 0), material=Ag),
    mp.Block(size=mp.Vector3(t, w, l), center=mp.Vector3(-0.05, -l/2+w/2, 0), material=Ag),
    mp.Block(size=mp.Vector3(t, l, w), center=mp.Vector3(-0.05, 0, l/2-w/2), material=Ag),
    mp.Block(size=mp.Vector3(t, l, w), center=mp.Vector3(-0.05, 0, -l/2+w/2), material=Ag),
    mp.Block(size=mp.Vector3(t, w, gap), center=mp.Vector3(-0.05, l/2-w/2, 0), material=mp.Medium(epsilon=1)), # Gap
]

freqs = np.linspace(0.3, 0.7, 50)
layers = 1

sim = mm.MetamaterialSimulation(period=a, geometry=geometry, freq_range=freqs, resolution=100, layers=layers, pol=pol)

sim.plot2D(plane='xz', save=True, filename='plotH.png')

sim.run()

[S11, S21] = sim.get_s_parameters()

# Save S11 and S21 to file
np.savetxt('examples/srr/tmp/H/s11tipleH.txt', S11)
np.savetxt('examples/srr/tmp/H/s21tipleH.txt', S21)

sim.sim.reset_meep()