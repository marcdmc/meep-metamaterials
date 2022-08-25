import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import time

tic = time.time()

resolution = 60  # pixels/um

pol = mp.Ex # Incident polarization

# Dimensions in um
l = .320
w = .090
a = .450 # Lattice constant
gap = .070

t = .020 # Thickness in um

# Dimensions of the cell
sep = a - l # Sepparation between resonators
width = l + sep
height = l + sep

st = 10*a # Substrate thickness
depth = 2*st

cell = mp.Vector3(a, a, depth) # 3D cell

# PML layers
dpml = 2 # Width of the PML
pml = mp.PML(dpml, direction=mp.Z) # PML in the non-periodic direction

# Wavelengths in um
wmin = 1 # minimum wavelength
wmax = 5 # maximum wavelength
wcen = (wmax-wmin)/2 + wmin # central wavelength

fmin = 1/wmax # minimum frequency
fmax = 1/wmin # maximum frequency
fcen = (fmax-fmin)/2 + fmin # central frequency
nfreqs = 100 # number of frequencies calculated

sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=fmax-fmin),
              component=pol,
              center=mp.Vector3(0,0,-depth/2+dpml),
              size=mp.Vector3(a, a, 0)),
]

refl_fr = mp.ModeRegion(center=mp.Vector3(0,0,-depth/2+dpml+0.2), size=mp.Vector3(width, height, 0))
tran_fr = mp.ModeRegion(center=mp.Vector3(0,0,depth/2-dpml-0.1), size=mp.Vector3(width, height, 0))

c_ = 299792458
wp = 2.175e15/(c_*1e6) # Plasma frequency
gamma = 6.5e12/(c_*1e6) # Collision frequency
susc = mp.DrudeSusceptibility(frequency=wp, gamma=gamma*1.65, sigma=1)

Gold = mp.Medium(epsilon=1, E_susceptibilities=[susc])

from meep.materials import Au, SiO2

n_SiO2 = 1.4059

# substrate = SiO2
metal = Gold
substrate = mp.Medium(index=n_SiO2)

# Define the geometry
geometry = [
    mp.Block(size=mp.Vector3(mp.inf, mp.inf, depth/2), center=mp.Vector3(0,0,depth/4), material=substrate),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, -t/2), material=metal),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2), material=metal),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(l/2-w/2, 0, -t/2), material=metal),
    mp.Block(size=mp.Vector3(w, l, t), center=mp.Vector3(-l/2+w/2, 0, -t/2), material=metal),
    mp.Block(size=mp.Vector3(gap, w, t), center=mp.Vector3(0, l/2-w/2, -t/2), material=mp.Medium(epsilon=1)), # Gap
]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=[pml],
    geometry=geometry,
    k_point=mp.Vector3(0,0,0), # Periodicity
    sources=sources,
    resolution=resolution)

refl = sim.add_mode_monitor(fcen, fmax-fmin, nfreqs, refl_fr)
tran = sim.add_mode_monitor(fcen, fmax-fmin, nfreqs, tran_fr)
freqs = mp.get_flux_freqs(refl)

# Run the simulation
pt = mp.Vector3(0,0,depth/2-dpml-0.1)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50,pol,pt,0.001))

refl_flux = mp.get_fluxes(refl)
tran_flux = mp.get_fluxes(tran)

S11 = sim.get_eigenmode_coefficients(refl, [1]).alpha[0]
S21 = sim.get_eigenmode_coefficients(tran, [1]).alpha[0]

a = np.array([coef[1] for coef in S11])
b = np.array([coef[0] for coef in S21])
c = np.array([coef[0] for coef in S11])

toc = time.time()

freqs = np.array(freqs)
wl = 1/freqs
plt.plot(wl, np.abs(a/c)**2, label='$|S_{11}|^2$')
plt.plot(wl, np.abs(b/c)**2, label='$|S_{21}|^2$')
plt.legend()

# Save plot to image file
plt.savefig('srr.png')

time_elapsed = toc - tic

# Wait for all processes to finish
mp.all_wait()

# Append to file
with open('times.txt', 'a') as f:
    f.write(str(time_elapsed) + '\n')