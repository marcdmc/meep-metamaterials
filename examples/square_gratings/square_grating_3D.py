import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep.materials import Ag

# 3D square grating simulation
a = 1.2
d = a/2
t = 0.2
dpml = 2.0
pad = 2

resolution = 80
pol = mp.Ex

# N = 5
# n = mp.divide_parallel_processes(N)
# index = np.linspace(2.5, 5, N)

block = mp.Block(size=mp.Vector3(d,d,t), center=mp.Vector3(), material=Ag)
# block = mp.Block(size=mp.Vector3(d,d,t), center=mp.Vector3(), material=mp.Medium(index=index[n]))

fcen = 1
df = 0.8
nfreq = 120

src = mp.Source(mp.GaussianSource(fcen, fwidth=df), component=pol, center=mp.Vector3(z=-t/2-pad+0.1), size=mp.Vector3(a,a))

cell = mp.Vector3(a,a,t+2*pad+2*dpml)

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=[mp.PML(dpml, direction=mp.Z)],
                    sources=[src],
                    geometry=[block],
                    resolution=resolution,
                    k_point=mp.Vector3(),
                    default_material=mp.Medium(epsilon=1))

refl = sim.add_mode_monitor(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=-t/2-pad+0.2), size=mp.Vector3(a,a)))
tran = sim.add_mode_monitor(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=t/2+pad-0.2), size=mp.Vector3(a,a)))

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(z=t/2+pad-0.1), 1e-3))

p1_coeff = sim.get_eigenmode_coefficients(refl, [1]).alpha[0]
p2_coeff = sim.get_eigenmode_coefficients(tran, [1]).alpha[0]

c1 = np.array([coef[1] for coef in p1_coeff])
c2 = np.array([coef[0] for coef in p2_coeff])
c3 = np.array([coef[0] for coef in p1_coeff])

freqs = np.linspace(fcen-df, fcen+df, nfreq)
wl = 1/freqs
R = np.abs(c1)**2 / np.abs(c3)**2
T = np.abs(c2)**2 / np.abs(c3)**2

res_freq = 0.5*(wl[np.argmax(T)]+wl[np.argmin(T)])

# mp.all_wait()

# Rs = mp.merge_subgroup_data(R)
# Ts = mp.merge_subgroup_data(T)
# res_freqs = mp.merge_subgroup_data(res_freq)

# for i in range(N):
#     r = [a[i] for a in Rs]
#     t = [a[i] for a in Rs]
#     f = [a[i] for a in res_freqs]
#     plt.figure()
#     plt.plot(wl, r, label='R')
#     plt.plot(wl, t, label='T')
#     plt.title('3D grating period %.1f µm, index %.1f and resonance at %.2f nm' % (a, n, f))
#     plt.xlabel('Wavelength (µm)')
#     plt.legend()
#     title = 'square_grating_%.2fum_n=%.1f' % (f, n)
#     plt.savefig('results/'+title+'.png')

plt.plot(wl, R, label='R')
plt.plot(wl, T, label='T')
plt.title('3D Ag grating period %.1f µm' % a)
plt.xlabel('Wavelength (µm)')
plt.legend()
# title = 'square_grating_%.2fum_n=%.1f' % (res_freq, n)
title = 'square_grating_%.2fum' % (res_freq)
plt.savefig('results/3D_'+title+'.png')