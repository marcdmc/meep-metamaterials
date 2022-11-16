import meep as mp
from meep import mpb
import numpy as np

geometry_lattice = mp.Lattice(size=mp.Vector3(5, 5))

geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))]
geometry = mp.geometric_objects_lattice_duplicates(geometry_lattice, geometry)
geometry.append(mp.Cylinder(0.2, material=mp.air))

resolution = 16
k_points = [
    mp.Vector3(),
    mp.Vector3(0.5, 0.5),
]
k_points = mp.interpolate(4, k_points)
num_bands = 50

ms = mpb.ModeSolver(num_bands=num_bands,
                    k_points=k_points,
                    geometry=geometry,
                    geometry_lattice=geometry_lattice,
                    resolution=resolution)

ms.run_tm()
tm_bands = ms.all_freqs

ms.run_te()
te_bands = ms.all_freqs

import matplotlib.pyplot as plt

for i in range(num_bands):
    te_band = [b[i] for b in te_bands]
    tm_band = [b[i] for b in tm_bands]
    plt.plot(te_band, 'r')
    plt.plot(tm_band, 'b')

plt.show()
plt.savefig('tutorial_bands.png')