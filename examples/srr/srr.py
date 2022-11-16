import numpy as np
import matplotlib.pyplot as plt
import sys, os

import meep as mp
from meep.materials import Au

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from meep_metamaterials import metamaterials as mm
from meep_metamaterials.retrieval import retrieval

resolution = 100  # pixels/um

pol = mp.Ex # Incident polarization

# Dimensions in um
l = .320
w = .090
a = .450 # Lattice constant
gap = .070
t = .020 # Thickness

geometry = [
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, 0), material=Au),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, 0), material=Au),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(l/2-w/2, 0, 0), material=Au),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(-l/2+w/2, 0, 0), material=Au),
    mp.Block(size=mp.Vector3(gap, w, t), center=mp.Vector3(0, l/2-w/2, 0), material=mp.Medium(epsilon=1)), # Gap
]

freqs = np.linspace(0.3, 0.8, 50) # Frequency range

sim = mm.MetamaterialSimulation(period=a, geometry=geometry, freq_range=freqs, resolution=resolution)

sim.run()

[S11, S21] = sim.get_s_parameters()

params = retrieval.eff_parameters(freqs, t, S11, S21)
epsilon = params['eps'] # Retrieved permittivity
mu = params['mu']       # Retrieved permeability

sim.reset_meep()