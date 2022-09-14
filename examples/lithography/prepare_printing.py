# prepare_printing.py

"""
Script to prepare various lithography printings in one sample.

Objectives:
- Square grating with resonances at 1.5 and 3 µm for silver.
- Square grating with resonances at 1.5 and 3 µm for a dielectric (around n=3).
- Square gratings with varying silver density.
- Square gratings with varying power and speed.
"""

import meep as mp
import numpy as np
import pandas as pd
import sys
sys.path.append('../metamaterials')
from metamaterials import lithography as lt

# Rows and columns to print
nrows = 50
ncols = 50


## Silver resonances
print('Silver resonances')
from meep.materials import Ag
# 3 µm
a = 2.29e-3
d = 0.45*a
t = 0.02e-3
pos = [-1.5, -2.4]
dx = 1e-5 # Step size
geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, 0), material=Ag)]
M = lt.draw_metamaterial(geometry, a, nrows, ncols, pos=pos, dx=dx) # 10x10 grid

# 1.5 µm
a = 0.85e-3
d = 0.45*a
t = 0.02e-3
pos = [0, -2.4]
dx = 1e-5 # Step size
geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, 0), material=Ag)]
M = pd.concat([M, lt.draw_metamaterial(geometry, a, nrows, ncols, pos=pos, dx=dx)])


## Dielectric resonances
print('Dielectric resonances')
# 3 µm
a = 6.9e-3
d = 0.8*a
t = 0.02e-3
pos = [1.5, -2.4]
dx = 1e-5 # Step size
geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, 0), material=mp.Medium(index=3.5))]
M = pd.concat([M, lt.draw_metamaterial(geometry, a, nrows, ncols, pos=pos, dx=dx)])

# 1.5 µm
a = 3.5e-3
d = 0.8*a
t = 0.02e-3
pos = [3, -2.4]
dx = 1e-5 # Step size
geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, 0), material=mp.Medium(index=3.5))]
M = pd.concat([M, lt.draw_metamaterial(geometry, a, nrows, ncols, pos=pos, dx=dx)])


## Silver density
print('Silver density')
# Here we try five different densities of silver for the 1.5 µm resonance
a = 0.85e-3
d = 0.45*a
t = 0.02e-3
positions = [[-1.5, -1.2], [0, -1.2], [1.5, -1.2], [3, -1.2], [-1.5, 0]]
for dx in np.linspace(1e-5, 1e-4, 5):
    pos = positions.pop(0)
    geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, 0), material=Ag)]
    M = pd.concat([M, lt.draw_metamaterial(geometry, a, nrows, ncols, pos=pos, dx=dx)])


## Power and speed
print('Power and speed')
# Here we try five different powers and speeds for the 1.5 µm resonance
a = 0.85e-3
d = 0.45*a
t = 0.02e-3
dx = 1e-5 # Step size
i = 0
j = 0
for power in np.linspace(0.1, 1, 4):
    for speed in [1.2, 0.8]:
        pos = [1.5*i - 1.5, 1.2*j + 1.2]
        geometry = [mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, 0), material=Ag)]
        M = pd.concat([M, lt.draw_metamaterial(geometry, a, nrows, ncols, pos=pos, dx=dx, pwr=power, speed=speed)])
        j += 1
    i += 1


## Save matrix
print('Saving matrix')
lt.save_to_matlab(M, 'printing.mat')