import meep as mp
from meep.materials import Au
import meep_retrieval as mr
import numpy as np
import matplotlib.pyplot as plt
import pickle

resolution = 60  # pixels/um

pol = mp.Ex # Incident polarization

lambda_min = 1
lambda_max = 4

# Dimensions in um
l = .320
w = .090
a = .450 # Lattice constant
gap = .070
t = .020 # Thickness in um

sim = mr.Simulation(resolution=resolution, cell=a, fmin=1/lambda_max, fmax=1/lambda_min, nfreqs=100, pol=pol, directory='results/basic')

c_ = 299792458
wp = 2.175e15/(c_*1e6) # Plasma frequency
gamma = 6.5e12/(c_*1e6) # Collision frequency
susc = mp.DrudeSusceptibility(frequency=wp, gamma=gamma*1.65, sigma=1)
Gold = mp.Medium(epsilon=1, E_susceptibilities=[susc])

metal = Gold
geometry = [
    #mp.Block(size=mp.Vector3(mp.inf, mp.inf, depth/2), center=mp.Vector3(0,0,depth/4), material=substrate),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, -t/2), material=metal),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2), material=metal),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(l/2-w/2, 0, -t/2), material=metal),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(-l/2+w/2, 0, -t/2), material=metal),
    mp.Block(size=mp.Vector3(gap, w, t), center=mp.Vector3(0, l/2-w/2, -t/2), material=mp.Medium(epsilon=1)), # Gap
]


#sim.view_structure()

# If simulation does not exist run it and pickle it, else load it.
if not sim.sim:
    sim.set_geometry(geometry, metal)
    sim.run()
    pickle.dump(sim, 'sim.pkl')
else:
    sim = pickle.load(open('sim.pkl', 'rb'))


[S11,S21] = sim.get_s_params()

freqs = np.linspace(1/lambda_max, 1/lambda_min, 100)
[eps, mu, n, z] = mr.retrieval(freqs, S11, S21, t)

plt.figure()
plt.plot(freqs, np.real(eps))
plt.savefig('results/basic/plot.png')