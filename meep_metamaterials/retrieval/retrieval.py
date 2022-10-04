from doctest import SkipDocTestCase
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from meep_metamaterials import constants as cnt
from meep_metamaterials import metamaterials as mm


def eff_parameters(freqs: np.ndarray,
                   d: float,
                   S11: np.ndarray,
                   S21: np.ndarray,
                   branch: int = 0,
                   plot_branches: bool = False,
                   continuity: bool = False):
    """
    Retrieve the effective parameters from the S-parameters.

    Parameters
    `freqs` : `np.ndarray`
        Frequencies at which the S-parameters are evaluated.
    `t` : `float`
        Thickness of the metamaterial.
    `S11` : `np.ndarray`
        S11 parameters.
    `S21` : `np.ndarray`
        S21 parameters.
    `branch` : `int`, default `0`
    `plot_branches` : `bool`, default `False`
    `continuity` : `bool`, default `False`

    Returns
    `eps` : `float`
        Effective permittivity.
    `mu` : `float`
        Effective permeability.
    `n` : `float`
        Effective index of refraction.
    `z` : `float`
        Effective impedance.
    """
    wl = 1/freqs
    k = 2*np.pi/wl
    z = np.sqrt(((1+S11)**2-S21**2)/((1-S11)**2-S21**2))
    einkd = S21/(1-S11*(z-1)/(z+1))

    min_branch = -10
    max_branch = 10
    
    n = [] # For each branch contains the effective index of refraction
    # Calculate branches from -10 to 10
    branches = np.arange(min_branch, max_branch)
    for br in branches:
        n.append(1/(k*d) * ((np.imag(np.log(einkd))+2*np.pi*br) - 1j*np.real(np.log(einkd))))

    final_n = np.zeros(len(freqs), dtype=np.complex128)
    n_branch = n[branch-min_branch] # The value that we will return

    dn = [0]
    # If we want to ensure continuity, whenever there is a change in n we must change branch
    if continuity:
        # Calculate derivatives of n
        dn = [[] for i in range(len(branches))]
        for br in branches:
            n_br = np.real(n[br-min_branch])
            for i in range(len(n_br)):
                if i > 0 and i < len(n_br)-1:
                    dn[br-min_branch].append((n_br[i+1] - n_br[i-1]) / (freqs[i+1] - freqs[i-1]))


        # final_n = np.ones(len(freqs))
        
        current_branch = branch
        final_n[0] = n[current_branch-min_branch][0]
        i = 1
        while i < len(freqs)-1:
            final_n[i] = n[current_branch-min_branch][i]
            # If there is a discontinuity, change branch
            if dn[branch-min_branch][i-1] > 50:
                current_branch -= 1
                final_n[i+1] = n[current_branch-min_branch][i+1]
                i += 1
                
            elif dn[branch-min_branch][i-1] < -50:
                current_branch += 1
                final_n[i+1] = n[current_branch-min_branch][i+1]
                i += 1
            i += 1

        final_n[-1] = n[current_branch-min_branch][-1]
        n_branch = final_n


    eps = n_branch/z
    mu = n_branch*z
        

    if plot_branches:
        plot_complex_branches(freqs, n, z, einkd, k, d, branch, n_branch)

    return {'eps': eps, 'mu': mu, 'n': n_branch, 'z': z}

def plot_complex_branches(freqs, n, z, einkd, k, d, selected_branch, n_branch):
    plt.figure()
    mid_values = []
    for branch in np.arange(-10, 10):
        # w = 2 if branch == selected_branch else 1 # Plot selected branch thicker
        w = 1
        
        n_br = n[branch+10]
        plt.plot(freqs, np.real(n_br), 'r', linewidth=w, label='|Re($n$)$cdot$Im($z$)|>Im($n$)$cdot$Re($z$)')
        
        # Plot blue line when conditions are met
        correct = np.abs(np.real(n_br)*np.imag(z)) <= np.imag(n_br)*np.real(z) # Condition
        line_x = []
        line_y = []
        for i in range(len(freqs)):
            if correct[i]:
                line_x.append(freqs[i])
                line_y.append(np.real(n_br[i]))
            else:
                plt.plot(line_x, line_y, 'b', linewidth=w, label='|Re($n$)$cdot$Im($z$)|$le$Im($n$)$cdot$Re($z$)')
                line_x = []
                line_y = []

        mid_values.append(np.real(n_br[int(len(n_br)/2)]))

    plt.plot(freqs, np.real(n_branch), 'r', linewidth=3, label='Selected branch')
    # Plot in thick blue the selected branch
    correct = np.abs(np.real(n_branch)*np.imag(z)) <= np.imag(n_branch)*np.real(z) # Condition
    line_x = []
    line_y = []
    for i in range(len(freqs)):
        if correct[i]:
            line_x.append(freqs[i])
            line_y.append(np.real(n_branch[i]))
        else:
            plt.plot(line_x, line_y, 'b', linewidth=3, label='|Re($n$)$cdot$Im($z$)|$le$Im($n$)$cdot$Re($z$)')
            line_x = []
            line_y = []

    plt.xlabel('Frequency')
    plt.ylabel('Re($n$)')
    # plt.legend()
    plt.ylim([np.min(mid_values), np.max(mid_values)])
    plt.show()