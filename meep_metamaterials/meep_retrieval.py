"""Library to simulate metamaterials effective response with meep."""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from . import s_parameters, constants as cnst, aux

# Constants
c = 299792458 # Speed of light in vacuum (m/s)

class Simulation:
    """
    Simulation class for meep.
    """

    # Constants
    c = 299792458 # Speed of light in vacuum (m/s)
    n_SiO2 = 1.4059 # Index of refraction of SiO2

    def __init__(self, resolution, cell, fmin, fmax, nfreqs, pol, directory=''):
        """
        Initialize simulation.

        Attributes:
        - `cell` (`float`): Cell size.
        - `resolution` (int): Resolution.
        - `fmin` (`float`): Minimum frequency.
        - `fmax` (`float`): Maximum frequency.
        - `nfreqs` (`int`): Number of frequencies.
        - `pol` (`str`): Polarization.
        """
        self.cell = cell # Cell parameter(s)
        self.resolution = resolution

        self.fmin = fmin
        self.fmax = fmax
        self.nfreqs = nfreqs
        self.freqs = np.linspace(fmin, fmax, nfreqs)

        self.pol = pol # Incident polarization

        self.dpml = 1/(2*fmin) # We create self absorbing layers that are half of the maximum wavelength.
        self.depth = 2/fmin # The sepparation from the source to the metamaterial is also half wavelength.

        self.dir = directory # Directory to save results
        if directory != '':
            self.dir = directory + '/'
        


    def set_geometry(self, geometry, metal, substrate=mp.Medium(epsilon=n_SiO2), all=False, nlayers=1):
        """Set geometry."""
        self.metal = metal # Metal material (eg. Au, Ag)
        self.substrate = substrate # Substrate material (by default SiO2)

        if nlayers > 1:
            self.geometry = self.add_layers(nlayers-1, geometry)

        self.geometry = geometry # Geometry (only the metal part has to be specified)
        
        # Add substrate ('all' will mean that the whole space is filled with the substrate)
        self.geometry.insert(0, mp.Block(size=mp.Vector3(mp.inf, mp.inf, self.depth/2+self.depth/2*all), center=mp.Vector3(0,0,self.depth/4-self.depth/4*all), material=self.substrate))

        fcen = 0.5*(self.fmin+self.fmax)
        sources = [
        mp.Source(mp.GaussianSource(fcen, fwidth=self.fmax-self.fmin),
                component=self.pol,
                center=mp.Vector3(0,0,-self.depth/2+self.dpml),
                size=mp.Vector3(self.cell, self.cell, 0)),
        ]

        cell = mp.Vector3(self.cell, self.cell, self.depth)

        sim = mp.Simulation(cell_size=cell,
                            resolution=self.resolution,
                            sources=sources,
                            boundary_layers=[mp.PML(self.dpml, direction=mp.Z)],
                            geometry=self.geometry,
                            k_point=mp.Vector3(0,0,0))

        self.sim = sim


    def run(self):
        """Run simulation."""

        if not hasattr(self, 'geometry') or not hasattr(self, 'sim'):
            raise Exception('Geometry not set. Use Simulation.set_geometry()')

        # Add field monitors
        refl_fr = mp.ModeRegion(center=mp.Vector3(0,0,-self.depth/2+self.dpml+0.2), size=mp.Vector3(self.cell, self.cell, 0))
        tran_fr = mp.ModeRegion(center=mp.Vector3(0,0,self.depth/2-self.dpml-0.1), size=mp.Vector3(self.cell, self.cell, 0))

        fcen = 0.5*(self.fmin+self.fmax)
        self.refl = self.sim.add_mode_monitor(fcen, self.fmax-self.fmin, self.nfreqs, refl_fr)
        self.tran = self.sim.add_mode_monitor(fcen, self.fmax-self.fmin, self.nfreqs, tran_fr)
        # self.freqs = mp.get_flux_freqs(self.refl)

        pt = mp.Vector3(0,0,self.depth/2-self.dpml-0.1)
        self.sim.run(until_after_sources=mp.stop_when_fields_decayed(50,self.pol,pt,0.001)) # Run for 50 steps after field has decayed to 0.001


    def view_structure(self, z=-.01):
        """Plots the metal structure, by default at 10nm from the substrate."""
        if not hasattr(self, 'geometry'):
            raise Exception('Geometry not set. Use Simulation.set_geometry()')
        
        print(self.sim.geometry[0].center)
        print(z)
        # self.sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(self.cell, self.cell, 0), center=mp.Vector3(0,0,z)))
        self.sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(self.cell, 0, self.depth), center=mp.Vector3(0,0,0)))


    def get_s_params(self, do_plot=False, do_save=False, d=0.02, plot='freqs', plot_title=None, filename=None):
        """Get s parameters."""

        self.d = d

        coefs1 = self.sim.get_eigenmode_coefficients(self.refl, [1]).alpha[0]
        coefs2 = self.sim.get_eigenmode_coefficients(self.tran, [1]).alpha[0]
        p1 = np.array([coef[1] for coef in coefs1]) # Reflected field
        p2 = np.array([coef[0] for coef in coefs2]) # Transmitted field
        p3 = np.array([coef[0] for coef in coefs1]) # Incident field

        S11 = p1/p3
        S21 = p2/p3

        n_SiO2 = 1.4059

        k = 2*np.pi*np.array(self.freqs)
        d_ref = self.depth/2-self.dpml-0.2-d
        d_tran = self.depth/2-self.dpml-0.1
        self.S11 = S11 * np.exp(-1j*k*(2*d_ref+0.2))
        self.S21 = S21 * np.exp(-1j*k*(d_ref+0.2)-1j*k*n_SiO2*d_tran) # TODO: Generalize for any substrate

        if do_save:
            self.save_s_params()
        if do_plot:
            self.plot_s_params(plot, plot_title, filename)

        return [self.S11, self.S21]


    def save_s_params(self):
        """Save results to excel file."""
        f = np.array(self.freqs)

        if hasattr(self, 'S11') and hasattr(self, 'S21'):
            df = pd.DataFrame(data={'f': f, 'S11': self.S11, 'S21': self.S21})

            # Name of the file is set according to polarization and current time
            timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
            if self.pol == 0:
                df.to_excel(self.dir + 's_params_TE' + timestr +'.xlsx', index=False)
            elif self.pol == 1:
                df.to_excel(self.dir + 's_params_TM' + timestr + '.xlsx', index=False)
        else:
            raise Exception('S parameters not set. Use Simulation.get_s_params()')

    def plot_s_params(self, plot, plot_title, filename=None):
        """Plot and save plot of s parameters."""
        if not hasattr(self, 'S11') or not hasattr(self, 'S21'):
            raise Exception('S parameters not set. Use Simulation.get_s_params()')

        f = np.array(self.freqs)
        S11 = self.S11
        S21 = self.S21

        if plot == 'freqs':
            print('Plotting S parameters...')
            plt.figure()
            plt.plot(f*c/1e6, np.abs(S11)**2, 'b', label='$|S_{11}|^2$')
            plt.plot(f*c/1e6, np.abs(S21)**2, 'r', label='$|S_{12}|^2$')
            plt.xlabel('f (THz)')
            plt.ylabel('Transmission and reflection')
            if plot_title is not None:
                plt.title(plot_title)
            plt.legend()
        elif plot == 'wl':
            plt.figure()
            plt.plot(1/f, np.abs(S11)**2, 'b', label='$|S_{11}|^2$')
            plt.plot(1/f, np.abs(S21)**2, 'r', label='$|S_{12}|^2$')
            plt.xlabel('Wavelength (µm)')
            plt.ylabel('Transmission and reflection')
            # plt.title('S parameters for a=' + str(self.cell) + ' µm and d=' + str(self.d) + ' µm')
            if plot_title is not None:
                plt.title(plot_title)
            plt.legend()

        timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
        if self.pol == 0:
            filename = self.dir + 's_params_TE' + timestr +'.png' if filename is None else filename
            plt.savefig(self.dir + 's_params_TE' + timestr +'.png')
        elif self.pol == 1:
            filename = self.dir + 's_params_TE' + timestr +'.png' if filename is None else filename
            plt.savefig(self.dir + 's_params_TM' + timestr + '.png')
        print('Saving S parameters plot...')

    
    def add_layers(self, n, geometry):
        """Add n layers of metamaterial"""
        # Check if layers can be added
        if n*self.cell > self.depth/2:
            raise Exception('Not enough space to add layers')

        # for i in range(n):
        #     geometry.add(mp.Block(size=mp.Vector3(a, a, a), center=mp.Vector3(0,0,i*a)))
        # return geometry

    
    def get_freqs(self):
        return self.freqs



def retrieval(freqs, S11, S21, d, branch=0, prepad=0, postpad=0, n_in=1, n_out=1, plot_branches=False, ensure_continuity=False):
    """
    Get the bulk permittivity and permeability of any material given its S parameters.
    Based on the method by Chen et. al. (2004).
    ---
    Parameters:
    `freqs`: Frequencies at which S parameters are measured.
    `S11`: Reflection coefficient.
    `S21`: Transmission coefficient.
    `d`: Thickness of the material.
    `branch`: Cosine branch to use. Default is `0`. (See paper by Chen et. al.)
    ---
    Optional parameters:
    `prepad`: Space between material and where the S11 parameter is measured. Default is `0`.
    `postpad`: Space between material and where the S21 parameter is measured. Default is `0`.
    `n_in`: Refractive index of the medium where S11 is measured. Default is `1` (air).
    `n_out`: Refractive index of the medium where S21 is measured. Default is `1` (air).
    `plot_branches`: Plot the cosine branches. Default is `False`.
    `ensure_continuity`: Ensure continuity of the cosine branches. Default is `False`.
    """
    freqs, S11, S21 = np.array(freqs), np.array(S11), np.array(S21)
    wl = 1 / freqs

    k_in = 2*np.pi*freqs * n_in
    k_out = 2*np.pi*freqs * n_out

    # Compensate phase delay between material and monitors/source.
    S11 *= np.exp(-1j*k_in*prepad)
    S21 *= np.exp(-1j*k_out*postpad)

    k = 2*np.pi/wl
    z = np.sqrt(((1+S11)**2-S21**2)/((1-S11)**2-S21**2))
    einkd = S21/(1-S11*(z-1)/(z+1))
    n = 1/(k*d) * ((np.imag(np.log(einkd))+2*np.pi*branch) - 1j*np.real(np.log(einkd)))

    eps = n/z
    mu = n*z

    if plot_branches:
        _plot_complex_branches(freqs, einkd, k, d, branch)

    return {'eps': eps, 'mu': mu, 'n': n, 'z': z}

def _plot_complex_branches(freqs, einkd, k, d, selected_branch):
    plt.figure()
    for branch in np.arange(-10, 10):
        w = 2 if branch == selected_branch else 1
        n = 1/(k*d) * ((np.imag(np.log(einkd))+2*np.pi*branch) - 1j*np.real(np.log(einkd)))
        plt.plot(freqs, np.real(n), 'r', linewidth=w)
    plt.ylim([-40, 40])
    plt.show()


def load_from_excel(file):
    """
    Load S parameters from excel file.
    The columns must be 'f', 'R' and 'T' for the frequencies, S11 and S21 respectively.

    Parameters:
    `file`(`str`): Path to the excel file.
    """
    df = pd.read_excel(file)
    f = np.array(df['f'].values)
    R = np.array([complex(r) for r in df['R'].values])
    T = np.array([complex(t) for t in df['T'].values])

    return {'f': f, 'R': R, 'T': T}


def DrudeMetal(wp, wc):
    """Define metal using Drude model.
    
    Attributes:
        `wp` (`float`): Plasma frequency.
        `wc` (`float`): Collision frequency.
    """
    wp /= (c*1e6) # Convert to meep units
    wc /= (c*1e6)
    susc = mp.DrudeSusceptibility(frequency=wp, gamma=wc, sigma=1)
    return mp.Medium(epsilon=1, E_susceptibilities=[susc])



def to_matlab(geometry, file):
    """Translate geometry to matlab file for lithography printing."""
    pass



##### MATERIALS ##### 
# TODO: Define with classes or whatever
def Gold():
    wp = 2.175e15/(c*1e6) # Plasma frequency
    gamma = 6.5e12/(c*1e6) # Collision frequency
    susc = mp.DrudeSusceptibility(frequency=wp, gamma=gamma*1.65, sigma=1)

    return mp.Medium(epsilon=1, E_susceptibilities=[susc])




# Methods for meep simulations
def get_s_params(sim, refl, tran):
    if not type(sim) == mp.simulation.Simulation:
        raise Exception('Simulation object not provided')

    p1_coeff = sim.get_eigenmode_coefficients(refl, [1]).alpha[0]
    p2_coeff = sim.get_eigenmode_coefficients(tran, [1]).alpha[0]

    c1 = np.array([coef[1] for coef in p1_coeff])
    c2 = np.array([coef[0] for coef in p2_coeff])
    c3 = np.array([coef[0] for coef in p1_coeff])