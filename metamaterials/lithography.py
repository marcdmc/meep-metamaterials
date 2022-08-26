from turtle import left
import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import meep as mp
from meep.materials import Ag

# def draw_block(block, x_offset, y_offset):
#     """Draws a MEEP block geometry object in lithography system units."""
#     M = pd.DataFrame(columns=['xi', 'yi', 'xf', 'yf'])
#     dx = 0.0001
#     dy = 0.0001
#     # dx = 0.01
#     # dy = 0.01
#     pwr = 0.1
#     width = block.size.x
#     height = block.size.y
#     for x in np.arange(-width/2, width/2+dx, dx):
#         for y in np.arange(-height/2, height/2, dy):
#             row = pd.DataFrame(data={'xi': [x], 'yi': [y], 'xf': [x], 'yf': [y+dy]})
#             M = pd.concat([M, row])
#         row = pd.DataFrame({'xi': [x], 'yi': [y+dy], 'xf': [x+dx], 'yf': [-height/2]})
#         M = pd.concat([M, row])
#     M = M[:-1]
#     M['xi'] = M['xi'] + x_offset
#     M['yi'] = M['yi'] + y_offset
#     M['xf'] = M['xf'] + x_offset
#     M['yf'] = M['yf'] + y_offset
#     return M

def draw_block(block, x_offset, y_offset, speed=1, power=1):
    M = pd.DataFrame(columns=['xi', 'yi', 'xf', 'yf'])
    width = block.size.x
    height = block.size.y
    dx = 0.0001
    for x in np.arange(-width/2, width/2, dx):
        row1 = pd.DataFrame(data={'xi': [x], 'yi': [-height/2], 'xf': [x], 'yf': [height/2]})
        row2 = pd.DataFrame(data={'xi': [x], 'yi': [height/2], 'xf': [x+dx], 'yf': [-height/2]})
        M = pd.concat([M, row1, row2])
    M = M[:-1]

    M['xi'] = M['xi'] + x_offset
    M['yi'] = M['yi'] + y_offset
    M['xf'] = M['xf'] + x_offset
    M['yf'] = M['yf'] + y_offset

    M.insert(2, 'pi', [power]*len(M)) # Define power
    M.insert(5, 'pf', [power]*len(M)) # Define power
    M.insert(6, 't', [height/speed]*len(M)) # Add time according to speed
    M.insert(7, 'X', [0]*len(M))
    M.insert(8, 'Y', [0]*len(M))
    M.insert(9, 'Z', [0]*len(M))

    return M

def draw_geometry(geometry, X_offset, Y_offset):
    """Draws a single unit cell in lithography system units given a MEEP geometry."""
    # TODO: Sort blocks by location to avoid crossing lines
    M = pd.DataFrame(columns=['xi', 'yi', 'xf', 'yf'])
    prev = False
    for block in geometry:
        x_offset = block.center.x + X_offset
        y_offset = block.center.y + Y_offset
        width = block.size.x
        height = block.size.y

        # # Connect last block to this one
        # if prev:
        #     last = M.iloc[-1]
        #     row = pd.DataFrame(data={'xi': [last['xf']], 'yi': [last['yf']], 'xf': [-width/2+x_offset], 'yf': [-height/2+y_offset]})
        #     M = pd.concat([M, row])
        # prev = True
        
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

def draw_metamaterial(geometry, a, nrows, ncols):
    """
    Draws a metamaterial given a MEEP geometry.
    
    Parameters
    - `geometry`: MEEP geometry in mm
    - `a`: Unit cell size in mm
    - `nrows`: Number of rows of unit cells to draw
    - `ncols`: Number of columns of unit cells to draw
    """
    M = pd.DataFrame(columns=['xi', 'yi', 'pi', 'xf', 'yf', 'pf', 't', 'X', 'Y', 'Z'])
    centroid = [(nrows-1)*a/2, (ncols-1)*a/2]
    for i in np.arange(0, nrows):
        for j in np.arange(0, ncols):
            x_offset = i*a - centroid[0]
            y_offset = j*a - centroid[1]

            # # Join with last unit cell
            # if i > 0 or j > 0:
            #     last = M.iloc[-1]
            #     # Sort to find the leftmost block and join with it
            #     leftmost_corners = [block.center.x-block.size.x/2 for block in geometry]
            #     lm = geometry[leftmost_corners.index(max(leftmost_corners))]
            #     row = pd.DataFrame(data={'xi': [last['xf']], 'yi': [last['yf']], 'xf': [lm.center.x-lm.size.x/2+x_offset], 'yf': [lm.center.y-lm.size.y/2+y_offset]})
            #     M = pd.concat([M, row])

            # Draw unit cell
            M = pd.concat([M, draw_geometry(geometry, x_offset, y_offset)])

            
    
    return M


def plot_lithography(M):
    """Plots the lithography printing instructions."""
    xi = np.array(M['xi'])
    yi = np.array(M['yi'])
    xf = np.array(M['xf'])
    yf = np.array(M['yf'])
    pi = np.array(M['pi']) # Power
    nrows = len(M)

    plt.figure()
    plt.axis('equal')

    last = 0
    for i in np.arange(0, nrows):
        plt.plot([xi[i], xf[i]], [yi[i], yf[i]], 'k-', color=str(1-pi[i]))
    #     if xf[i] != xi[last]:   
    #         plt.plot([xi[last], xi[i]], [yi[last], yi[i]], 'k-')
    #         plt.plot([xi[i], xf[i]], [yi[i], yf[i]], 'k-')
    #         last = i+1
    # plt.plot([xi[last], xf[i]], [yi[last], yf[i]], 'k-')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.show()

    return


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