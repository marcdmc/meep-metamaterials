import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import meep as mp
from meep.materials import Ag


def draw_block(block, x_offset, y_offset, speed=1, power=1, dx=0.0005):
    """Draw 3d block in lithography system units.
    
    Parameters:
    - `block`: MEEP block with units in mm
    - `x_offset`
    - `y_offset`
    - `speed`: Speed of laser in mm/time step
    - `power`: Power of laser
    """
    M = pd.DataFrame(columns=['xi', 'yi', 'xf', 'yf', 'pi', 'pf', 't', 'X', 'Y', 'Z'])
    width = block.size.x
    height = block.size.y
    depth = block.size.z
    for z in np.arange(-depth/2, depth/2, dx):
        for x in np.arange(-width/2, width/2, dx):
            row1 = pd.DataFrame(data={'xi': [x], 'yi': [-height/2], 'xf': [x], 'yf': [height/2], 'pi': [power], 'pf': [power], 't': [1], 'X': [0], 'Y': [0], 'Z': [z]})
            row2 = pd.DataFrame(data={'xi': [x], 'yi': [height/2], 'xf': [x+dx], 'yf': [-height/2], 'pi': [power], 'pf': [power], 't': [1], 'X': [0], 'Y': [0], 'Z': [z]})
            M = pd.concat([M, row1, row2])
        M = M[:-1]

    # If there is no depth print in 2D
    if depth == 0:
        for x in np.arange(-width/2, width/2, dx):
            row1 = pd.DataFrame(data={'xi': [x], 'yi': [-height/2], 'xf': [x], 'yf': [height/2], 'pi': [power], 'pf': [power], 't': [1], 'X': [0], 'Y': [0], 'Z': [0]})
            row2 = pd.DataFrame(data={'xi': [x], 'yi': [height/2], 'xf': [x+dx], 'yf': [-height/2], 'pi': [power], 'pf': [power], 't': [1], 'X': [0], 'Y': [0], 'Z': [0]})
            M = pd.concat([M, row1, row2])
        M = M[:-1]

    M['xi'] = M['xi'] + x_offset
    M['yi'] = M['yi'] + y_offset
    M['xf'] = M['xf'] + x_offset
    M['yf'] = M['yf'] + y_offset

    return M

def draw_geometry(geometry, X_offset, Y_offset, dx=0.0005, pwr=1, speed=1):
    """Draws a single unit cell in lithography system units given a MEEP geometry."""
    # TODO: Sort blocks by location to avoid crossing lines, change laser power
    M = pd.DataFrame(columns=['xi', 'yi', 'xf', 'yf'])
    prev = False
    for block in geometry:
        x_offset = block.center.x + X_offset
        y_offset = block.center.y + Y_offset
        width = block.size.x
        height = block.size.y

        # Draw block
        M = pd.concat([M, draw_block(block, x_offset, y_offset, dx=dx, power=pwr, speed=speed)])

    return M

def draw_metamaterial(geometry, a, nrows, ncols, nlayers=1, pos=[0,0], dx=0.0005, pwr=1, speed=1):
    """
    Draws a metamaterial given a MEEP geometry.
    
    Parameters
    - `geometry`: MEEP geometry in mm
    - `a`: Unit cell size in mm
    - `nrows`: Number of rows of unit cells to draw
    - `ncols`: Number of columns of unit cells to draw
    - `nlayers`: Number of layers of unit cells to draw
    - `pos`: Offset from the origin in mm
    - `dx`: Step size of drawing in mm
    """
    M = pd.DataFrame(columns=['xi', 'yi', 'pi', 'xf', 'yf', 'pf', 't', 'X', 'Y', 'Z'])
    centroid = [(ncols-1)*a/2, (nrows-1)*a/2, (nlayers-1)*a/2]

    M_sing = draw_geometry(geometry, 0, 0, dx, pwr, speed) # Single cell

    for i in np.arange(0, ncols):
        for j in np.arange(0, nrows):
            for k in np.arange(0, nlayers):
                x_offset = i*a - centroid[0]
                y_offset = j*a - centroid[1]
                z_offset = k*a - centroid[2]
                m = M_sing.copy()
                m['xi'] = m['xi'] + x_offset
                m['yi'] = m['yi'] + y_offset
                m['xf'] = m['xf'] + x_offset
                m['yf'] = m['yf'] + y_offset
                m['Z'] = m['Z'] + z_offset

                M = pd.concat([M, m])

    # Add offset
    M['X'] = M['X'] + pos[0]
    M['Y'] = M['Y'] + pos[1]
    
    return M


def plot_lithography(M):
    """Plots the lithography printing instructions."""
    xi = np.array(M['xi'])
    yi = np.array(M['yi'])
    xf = np.array(M['xf'])
    yf = np.array(M['yf'])
    pi = np.array(M['pi']) # Power
    Z = np.array(M['Z']) # Z coordinate
    nrows = len(M)

    plt.figure()
    plt.axis('equal')

    last = 0
    for i in np.arange(0, nrows):
        plt.plot([xi[i], xf[i]], [yi[i], yf[i]], '-', color=str(1-pi[i]))

    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.show()

    return


def plot3d_lithography(M):
    """Plots the lithograpy printing instructions in 3D."""
    xi = np.array(M['xi'])
    yi = np.array(M['yi'])
    xf = np.array(M['xf'])
    yf = np.array(M['yf'])
    pi = np.array(M['pi']) # Power
    X = np.array(M['X'])
    Y = np.array(M['Y'])
    Z = np.array(M['Z'])
    nrows = len(M)

    # Make 3d plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    for i in np.arange(0, nrows):
        ax.plot([xi[i], xf[i]], [yi[i], yf[i]], [Z[i], Z[i]], '-', color=str(1-pi[i]))
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.show()


def save_to_matlab(M, filename, name='data'):
    """Saves the lithography printing instructions to a matlab file."""
    data = []
    for index, row in M.iterrows():
        data.append(np.array(row, dtype=float))
    io.savemat(filename, {name: data})


def read_matlab(filename, name='data'):
    """Reads the lithography printing instructions from a mat file."""
    data = io.loadmat(filename)
    return pd.DataFrame(data[name], columns=['xi', 'yi', 'pi', 'xf', 'yf', 'pf', 't', 'X', 'Y', 'Z'])


def img_to_geometry(img, scale=1):
    """Converts an image to a MEEP geometry.
    
    Parameters
    - `img`: Image to convert
    - `scale`: Scale factor to convert image to mm
    """
    img = np.array(img).astype(float)
    img = img / np.max(img)
    img = 1 - img

    geometry = []
    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if img[i,j] > 0.5:
                geometry.append(mp.Block(center=mp.Vector3(j*scale, i*scale),
                                         size=mp.Vector3(scale, scale),
                                         material=mp.Medium(epsilon=1)))

    return geometry
