from tkinter import N
import meep as mp
import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from IPython.display import Video
from scipy.signal import argrelextrema

resolution = 50        # pixels/μm

nperiods = 5

fsrc = 1.25            # frequency of the source

dpml = 2.0             # PML thickness
dsub = 3.0             # substrate thickness
dpad = 3.0             # padding between grating and PML
gp = 2.4              # grating period
gh = 0.05               # grating height
gdc = 0.5              # grating duty cycle

ff_distance = 1e3      # far-field distance from near-field monitor
ff_angle = 50          # far-field cone angle
ff_npts = 500          # number of far-field points

ff_length = ff_distance*math.tan(math.radians(ff_angle))
ff_res = ff_npts/ff_length

sx = dpml+dsub+gh+dpad+dpml
sy = gp
cell_size = mp.Vector3(sx)

# Materials
from meep.materials import Ag

n_air = 1
n_mat = 1.5

def predict_n(angle, l=1):
    theta_i = angle * math.pi/180

    cell_size = mp.Vector3(sx,sy)

    pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

    k_point = mp.Vector3(fsrc * n_air).rotate(mp.Vector3(z=1), theta_i)

    src_pt = mp.Vector3(-0.5*sx+dpml+0.5*dsub)
    sources = [
        mp.EigenModeSource(
            src=mp.ContinuousSource(fsrc),
            center=src_pt,
            size=mp.Vector3(y=sy),
            direction=mp.AUTOMATIC if theta_i == 0 else mp.NO_DIRECTION,
            eig_kpoint=k_point,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if theta_i == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    n2f_pt = mp.Vector3(0.5*sx-dpml-0.5*dpad)

    geometry = [
        mp.Block(material=mp.Medium(index=n_mat), center=mp.Vector3(sx/4), size=mp.Vector3(sx/2, mp.inf)),
        mp.Block(material=Ag, center=mp.Vector3(), size=mp.Vector3(gh, gdc*gp)),
    ]

    sim = mp.Simulation(resolution=resolution,
                        split_chunks_evenly=True,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        k_point=k_point,
                        sources=sources)

    n2f_obj = sim.add_near2far(fsrc, 0, 1, mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy)), nperiods=nperiods)

    sim.run(until=50)

    ff_unitcell = sim.get_farfields(n2f_obj, ff_res, center=mp.Vector3(ff_distance,0.5*ff_length), size=mp.Vector3(y=ff_length))

    sim.reset_meep()

    freqs = mp.get_near2far_freqs(n2f_obj)
    wvl = np.divide(1,freqs)
    ff_lengths = np.linspace(0,ff_length,ff_npts)
    angles = [math.degrees(math.atan(f)) for f in ff_lengths/ff_distance]

    I = np.abs(ff_unitcell['Ez'])**2 # Intensity
    max_ind = np.array(argrelextrema(I, np.greater))
    max_ind = max_ind[I[max_ind] > 0.1*np.max(I)]
    idx = np.argmax(I)
    max_ind = max_ind[idx < max_ind]

    # Predict n from peak at order l
    theta_out = np.radians(np.array(angles)[max_ind][l-1])
    n_pred = (np.sin(theta_i) + l/(fsrc*gp)) / np.sin(theta_out)
    
    sim.reset_meep()

    return n_pred

n_preds = np.array([])

angle_min = 0
angle_max = 45

for angle in range(angle_min, angle_max):
    try:
        print("Predicting n for angle: ", angle)
        n_pred = predict_n(angle)
        print("Predicted n: {}".format(n_pred))
        n_preds = np.append(n_preds, n_pred)
    except:
        print("Failed to predict n for angle: ", angle)
        n_preds = np.append(n_preds, 0)

plt.figure()
plt.plot(range(angle_min, angle_max), n_preds)
plt.hlines(n_mat, angle_min, angle_max, linestyles='dashed')
plt.xlabel("Angle (degrees)")
plt.ylabel("Predicted n")
plt.savefig("n_pred2.png")