import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import sys, os

from meep.materials import Ag

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from meep_metamaterials import metamaterials as mm


l = .320
w = .090
a = .450
gap = .070
t = .020

geometry = [
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, -t/2)+mp.Vector3(a/2,a/2), material=Ag),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2)+mp.Vector3(a/2,a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(l/2-w/2, 0, -t/2)+mp.Vector3(a/2,a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(-l/2+w/2, 0, -t/2)+mp.Vector3(a/2,a/2), material=Ag),
    mp.Block(size=mp.Vector3(gap, w, t), center=mp.Vector3(0, l/2-w/2, -t/2)+mp.Vector3(a/2,a/2), material=mp.Medium(epsilon=1)),

    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, -t/2)+mp.Vector3(-a/2,a/2), material=Ag),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2)+mp.Vector3(-a/2,a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(l/2-w/2, 0, -t/2)+mp.Vector3(-a/2,a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(-l/2+w/2, 0, -t/2)+mp.Vector3(-a/2,a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, gap, t), center=mp.Vector3(-l/2+w/2, 0, -t/2)+mp.Vector3(-a/2,a/2), material=mp.Medium(epsilon=1)),

    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, -t/2)+mp.Vector3(a/2,-a/2), material=Ag),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2)+mp.Vector3(a/2,-a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(l/2-w/2, 0, -t/2)+mp.Vector3(a/2,-a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(-l/2+w/2, 0, -t/2)+mp.Vector3(a/2,-a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, gap, t), center=mp.Vector3(l/2-w/2, 0, -t/2)+mp.Vector3(a/2,-a/2), material=mp.Medium(epsilon=1)),

    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, -t/2)+mp.Vector3(-a/2,-a/2), material=Ag),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2)+mp.Vector3(-a/2,-a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(l/2-w/2, 0, -t/2)+mp.Vector3(-a/2,-a/2), material=Ag),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(-l/2+w/2, 0, -t/2)+mp.Vector3(-a/2,-a/2), material=Ag),
    mp.Block(size=mp.Vector3(gap, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2)+mp.Vector3(-a/2,-a/2), material=mp.Medium(epsilon=1)),
]

freqs = np.linspace(0.3, 0.8, 50)
layers = 1

sim = mm.MetamaterialSimulation(period=2*a, geometry=geometry, freq_range=freqs, resolution=100, layers=layers)

sim.plot2D(plane='xy', save=True, filename='test.png')

# exit()

sim.run()

[S11, S21] = sim.get_s_parameters()

# Save S11 and S21 to file
np.savetxt('examples/misc/NL/s11_NL.txt', S11)
np.savetxt('examples/misc/NL/s21_NL.txt', S21)

sim.sim.reset_meep()