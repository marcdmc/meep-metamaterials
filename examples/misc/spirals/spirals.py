import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from meep_metamaterials import metamaterials as mm
from meep_metamaterials import geometries as geom
from meep.materials import Ag

sxy = 0.45
R = 0.14
r = 0.04
P = 0.1 # Frequency of turns
h = 0.32 # Height of spring

spring = geom.spring(R, r, P, h, Ag, axis='x')

geometry = [mp.Block(center=mp.Vector3(), size=mp.Vector3(sxy, sxy, sxy), material=spring)]

sim = mm.MetamaterialSimulation(period=sxy,
                                geometry=geometry,
                                freq_range=np.linspace(0.2, 1, 50),
                                resolution=100,
                                pol=mp.Hx)

sim.run()

[S11, S21] = sim.get_s_parameters()

# Save S11 and S21 to file
np.savetxt('examples/misc/spirals/s11_test3.txt', S11)
np.savetxt('examples/misc/spirals/s21_test3.txt', S21)
