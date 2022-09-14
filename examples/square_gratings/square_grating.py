import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep.materials import Ag

n = 5          # index of square grating
a = 1.2            # period 
d = 0.5*a            # side of square
t = 0.2 # thickness of grating
dpml = 1 # PML thickness
pad = 2 # padding between grating and PML

res = 50 # pixels/μm
pol = mp.Ex

fcen = 1
df = 0.8
nfreqs = 50

dims = 3 # dimensions of cell

# block = mp.Block(mp.Vector3(d,t), center=mp.Vector3(), material=mp.Medium(index=n))
block = mp.Block(size=mp.Vector3(d,t), center=mp.Vector3(), material=Ag)

src = mp.Source(mp.GaussianSource(fcen, fwidth=df), component=pol, center=mp.Vector3(y=-t/2-pad+0.1), size=mp.Vector3(a))
cell = mp.Vector3(a, 2*dpml+2*pad+t)
pml_layers = [mp.PML(dpml, direction=mp.Y)]

sim = mp.Simulation(cell_size=cell,
                    default_material=mp.Medium(index=1),
                    geometry=[block],
                    sources=[src],
                    resolution=50,
                    boundary_layers=[mp.PML(dpml, direction=mp.Y)],
                    k_point=mp.Vector3())

# Add monitors
refl = sim.add_mode_monitor(fcen, df, nfreqs, mp.FluxRegion(center=mp.Vector3(y=-t/2-pad+0.2), size=mp.Vector3(a)))
tran = sim.add_mode_monitor(fcen, df, nfreqs, mp.FluxRegion(center=mp.Vector3(y=t/2+pad-0.1), size=mp.Vector3(a)))

# Run the simulation
sim.run(until_after_sources=mp.stop_when_fields_decayed(100, pol, mp.Vector3(y=pad), 1e-6))

# Get eigenmode data
p1_coeff = sim.get_eigenmode_coefficients(refl, [1]).alpha[0]
p2_coeff = sim.get_eigenmode_coefficients(tran, [1]).alpha[0]

c1 = np.array([coef[1] for coef in p1_coeff])
c2 = np.array([coef[0] for coef in p2_coeff])
c3 = np.array([coef[0] for coef in p1_coeff])

freqs = np.linspace(fcen-df, fcen+df, nfreqs)
wl = 1/freqs
R = np.abs(c1)**2 / np.abs(c3)**2
T = np.abs(c2)**2 / np.abs(c3)**2

# res_freq = 0.5*(wl[np.argmax(T)]+wl[np.argmin(T)])
res_freq = wl[np.argmin(T)]

plt.plot(wl, R, label='R')
plt.plot(wl, T, label='T')
plt.title('Ag grating period %.1f µm resonance at %.2f nm' % (a, res_freq))
plt.xlabel('Wavelength (µm)')
plt.legend()
title = 'square_grating_%.2fum' % res_freq
plt.savefig('results/'+title+'.png')
