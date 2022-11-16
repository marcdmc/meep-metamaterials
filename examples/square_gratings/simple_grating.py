import meep as mp
import meep_metamaterials.metamaterials as mm

lambda_min = 1
lambda_max = 5
nfreqs = 100
pol = mp.Ex
d = .45
a = 2*d
t = 0.02

sim = mm.Simulation(resolution=60, cell=a, fmin=1/lambda_max, fmax=1/lambda_min, nfreqs=nfreqs, pol=pol)

metal = mp.Medium(index=3)

geometry = [
    mp.Block(size=mp.Vector3(a, a, t), center=mp.Vector3(0, 0, -t/2), material=metal),
]

sim.set_geometry(geometry, metal)
sim.run()
[S11, S12] = sim.get_s_params()