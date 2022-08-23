import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import meep as mp
from meep.materials import Ag

shrink_factor = 10 # Shrinkage factor of the gel
# Gel dimensions in mm
gel_x = 10
gel_y = 10

# Units in µm
a = 1.2 # Lattice constant
d = .55*a # Square dimensions
t = 0.020 # Thickness of metal
metal = Ag

d = 0.01
# geometry = [
#     mp.Block(size=mp.Vector3(d, d, t), center=mp.Vector3(0, 0, -t/2), material=metal),
# ]

l = .320
w = .090
a = .450 # Lattice constant
gap = .070
geometry = [
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, l/2-w/2, -t/2), material=metal),
    mp.Block(size=mp.Vector3(l, w, t), center=mp.Vector3(0, -l/2+w/2, -t/2), material=metal),
    mp.Block(size=mp.Vector3(w, l-2*w, t), center=mp.Vector3(l/2-w/2, 0, -t/2), material=metal),
    mp.Block(size=mp.Vector3(w, l-2*w, t), center=mp.Vector3(-l/2+w/2, 0, -t/2), material=metal),
    # mp.Block(size=mp.Vector3(gap, w, t), center=mp.Vector3(0, l/2-w/2, -t/2), material=mp.Medium(epsilon=1)), # Gap
]

def draw_block(block, x_offset, y_offset):
    """Draws a MEEP block geometry object in lithography system units."""
    M = pd.DataFrame(columns=['xi', 'yi', 'xf', 'yf'])
    # dx = 0.0005
    # dy = 0.0005
    dx = 0.01
    dy = 0.01
    pwr = 0.1
    width = block.size.x
    height = block.size.y
    for x in np.arange(-width/2, width/2+dx, dx):
        for y in np.arange(-height/2, height/2, dy):
            row = pd.DataFrame(data={'xi': [x], 'yi': [y], 'xf': [x], 'yf': [y+dy]})
            M = pd.concat([M, row])
        row = pd.DataFrame({'xi': [x], 'yi': [height/2], 'xf': [x+dx], 'yf': [-height/2]})
        M = pd.concat([M, row])
    M = M[:-1]
    M['xi'] = M['xi'] + x_offset
    M['yi'] = M['yi'] + y_offset
    M['xf'] = M['xf'] + x_offset
    M['yf'] = M['yf'] + y_offset
    return M

def draw_geometry(geometry, X_offset, Y_offset):
    """Draws a unit cell in lithography system units given a MEEP geometry."""
    # TODO: Sort blocks by location to avoid crossing lines
    M = pd.DataFrame(columns=['xi', 'yi', 'xf', 'yf'])
    prev = False
    for block in geometry:
        x_offset = block.center.x + X_offset
        y_offset = block.center.y + Y_offset
        width = block.size.x
        height = block.size.y

        # Connect last block to this one
        if prev:
            last = M.iloc[-1]
            row = pd.DataFrame(data={'xi': [last['xf']], 'yi': [last['yf']], 'xf': [-width/2+x_offset], 'yf': [-height/2+y_offset]})
            M = pd.concat([M, row])
        prev = True
        
        # Draw block
        M = pd.concat([M, draw_block(block, x_offset, y_offset)])

        # Insert remaining columns
        # TODO: Change laser power, XYZ coordinates
        M.insert(2, 'pi', [0.1]*len(M))
        M.insert(5, 'pf', [0.1]*len(M))
        M.insert(6, 't', [1]*len(M))
        M.insert(7, 'X', [0]*len(M))
        M.insert(8, 'Y', [0]*len(M))
        M.insert(9, 'Z', [0]*len(M))

    return M

def draw_metamaterial(geometry, nrows, ncols):
    """Draws a metamaterial given a MEEP geometry."""
    # TODO: Center cells instead of beginning at (0, 0)
    M = pd.DataFrame(columns=['xi', 'yi', 'pi', 'xf', 'yf', 'pf', 't', 'X', 'Y', 'Z'])
    for i in np.arange(0, nrows):
        for j in np.arange(0, ncols):
            x_offset = i*a
            y_offset = j*a
            M = pd.concat([M, draw_geometry(geometry, x_offset, y_offset)])

            # Join with last unit cell
            # if i > 0 and j > 0:
            #     last = M.iloc[-1]
            #     next = geometry[0].center
            #     row = pd.DataFrame(data={'xi': [last['xf']], 'yi': [last['yf']], 'xf': [-width/2+x_offset], 'yf': [-height/2+y_offset]})
            #     M = pd.concat([M, row])
    
    return M


def save_to_matlab(M, filename, name='data'):
    """Saves the lithography printing instructions to a matlab file."""
    data = []
    for index, row in M.iterrows():
        data.append(np.array(row, dtype=float))
    io.savemat(filename, {name: data})