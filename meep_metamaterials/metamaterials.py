import meep as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from typing import Callable, List, Tuple, Union, Optional
from meep.geom import Vector3, init_do_averaging, GeometricObject, Medium

from . import constants as cnt

class MetamaterialSimulation():
    def __init__(self,
                 period: int,
                 geometry: Optional[List[GeometricObject]],
                 source = None,
                 freq: float = None,
                 freq_range: np.ndarray = None,
                 wavelength: float = None,
                 wavelength_range: np.ndarray = None,
                 grating_n_range: np.ndarray = None,
                 substrate: mp.Medium = mp.Medium(index=1),
                 substrate2: mp.Medium = mp.Medium(index=1),
                 substrate_n_range: np.ndarray = None,
                 substrate2_n_range: np.ndarray = None,
                 resolution: int = 50,
                 period_x = None, period_y = None, period_z = None,
                 sx: float = None, sy: float = None, sz: float = None,
                 dimensions: int = 3,
                 pol: int = mp.Ex,
                ):

        self.period = period
        self.geometry = geometry
        self.source = source

        self.substrate = substrate
        self.substrate2 = substrate2
        self.substrate_n_range = substrate_n_range
        self.substrate2_n_range = substrate2_n_range
        self.period_x, self.period_y, self.period_z = period_x, period_y, period_z
        self.dimensions = dimensions
        self.pol = pol
        self.resolution = resolution

        self.pixel_size = 1/self.resolution

        self.mm_thickness = get_mm_thickness(geometry) # Thickness of metamaterial

        # Sources
        if source:
            self.source = source # for the moment, only one source
            if isinstance(self.source.src, mp.ContinuousSource):
                self.source_type = 'continuous'
                self.fcen = self.source.src.frequency
                self.wvl = 1/self.fcen
            elif isinstance(self.source.src, mp.GaussianSource):
                self.source_type = 'gaussian'
                self.fcen = self.source.src.frequency
                self.width = self.source.src.width
                self.wvl = 1/self.fcen

                self.fmin = self.fcen - self.width/2
                self.fmax = self.fcen + self.width/2
        else:
            if freq is not None:
                self.fcen = freq
                self.source_type = 'continuous'
            elif freq_range is not None:
                self.freqs = freq_range
                self.fmin = self.freqs[0]
                self.fmax = self.freqs[-1]
                self.fcen = 0.5*(self.fmin+self.fmax)
                self.wvl = 1/self.fcen
                self.source_type = 'gaussian'
                self.width = self.fmax-self.fmin
                self.nfreqs = len(self.freqs)
            elif wavelength is not None:
                self.wvl = wavelength
                self.fcen = 1/self.wvl
                self.source_type = 'continuous'
            elif wavelength_range is not None:
                self.freqs = 1/wavelength_range
                self.fmin = 1/wavelength_range[-1]
                self.fmax = 1/wavelength_range[0]
                self.fcen = 0.5*(self.fmin+self.fmax)
                self.wvl = 1/self.fcen
                self.source_type = 'gaussian'
                self.width = self.fmax-self.fmin
                self.nfreqs = len(self.freqs)

        # Depth of the simulation
        self.depth = 4/self.fmin if self.source_type == 'gaussian' else 4*self.wvl

        self.sx = self.period
        self.sy = self.depth if dimensions == 2 else self.period
        self.sz = 0 if dimensions == 2 else self.depth

        # Absorbing boundary conditions
        self.dpml = 2/self.fmin if self.source_type == 'gaussian' else 2/self.fcen
        if dimensions == 2:
            self.sy += 2*self.dpml
            self.depth += 2*self.dpml
        else:
            self.sz += 2*self.dpml
            self.depth += 2*self.dpml

        if source is None:
            source_plane = mp.Vector3(self.sx, self.sy)
            source_center = mp.Vector3(z=-self.depth/2+self.dpml+0.1*self.depth/4)
            self.source = mp.EigenModeSource(mp.ContinuousSource(self.fcen),
                                    component=pol,
                                    center=source_center,
                                    size=source_plane) if self.source_type == 'continuous' else \
                          mp.EigenModeSource(mp.GaussianSource(self.fcen, fwidth=self.width),
                                    component=pol,
                                    center=source_center,
                                    size=source_plane)

        # Check if geometry is in two or three dimensions
        has_depth = False
        for block in geometry:
            if block.size.z != 0:
                has_depth = True
                break
        self.dimensions = 2 if not has_depth else 3

        if dimensions == 1:
            pass 
        elif dimensions == 2:
            # first check that z dimension is zero
            if self.sz != 0:
                raise ValueError("sz must be zero for 2D simulations")

            self.cell = mp.Vector3(self.sx, self.sy)
            # Take only x and z components of the geometry
            geometry_2d = []
            for block in geometry:
                if block.size.z != 0:
                    geometry_2d.append(mp.Block(mp.Vector3(x=block.size.x, y=block.size.z),
                                                center=block.center,
                                                material=block.material))
                else:
                    geometry_2d.append(block)

            # Transform into 2D source
            source_2d = mp.EigenModeSource(self.source.src,
                                  component=self.source.component,
                                  center=mp.Vector3(y=self.source.center.z),
                                  size=mp.Vector3(self.sx))
        elif dimensions == 3:
            self.cell = mp.Vector3(self.sx, self.sy, self.sz)
        else:
            raise ValueError('Too many dimensions????')
        
        print('Cell size: ', self.cell.x, self.cell.y, self.cell.z)
        print('Geometry size: ', self.geometry[0].size.x, self.geometry[0].size.y, self.geometry[0].size.z)

        self.sim = mp.Simulation(
            cell_size=self.cell,
            sources=[self.source] if dimensions == 3 else [source_2d],
            geometry=self.geometry if dimensions == 3 else geometry_2d,
            resolution=resolution,
            boundary_layers=[mp.PML(self.dpml, direction=mp.Y)] if dimensions == 2 else [mp.PML(self.dpml, direction=mp.Z)],
            k_point=mp.Vector3(),
            default_material=substrate,
            dimensions=self.dimensions,
        )

        # Add field monitors
        refl_center = -self.sz/2+self.dpml+0.2*self.depth/4
        tran_center = self.sz/2-self.dpml-0.1*self.depth/4
        if self.dimensions == 2:
            refl_fr = mp.ModeRegion(center=mp.Vector3(y=-self.depth/2+self.dpml+0.2*self.depth), size=mp.Vector3(self.sx))
            tran_fr = mp.ModeRegion(center=mp.Vector3(y=+self.depth/2-self.dpml-0.1*self.depth), size=mp.Vector3(self.sx))
        elif self.dimensions == 3:
            refl_fr = mp.ModeRegion(center=mp.Vector3(z=-self.depth/2+self.dpml+10*self.pixel_size), size=mp.Vector3(self.sx, self.sy))
            tran_fr = mp.ModeRegion(center=mp.Vector3(z=+self.depth/2-self.dpml-5*self.pixel_size), size=mp.Vector3(self.sx, self.sy))


        fcen = 0.5*(self.fmin+self.fmax)
        self.refl = self.sim.add_mode_monitor(fcen, self.fmax-self.fmin, self.nfreqs, refl_fr)
        self.tran = self.sim.add_mode_monitor(fcen, self.fmax-self.fmin, self.nfreqs, tran_fr)

        self.has_run = False


    def run(self,
            timesteps: int = None,
            threshold: float = 1e-3):
        """
        Runs the simulation.
        """
        if timesteps is not None:
            self.sim.run(until=timesteps)
        else:
            pt = mp.Vector3(z=self.sz/2-self.dpml-self.pixel_size) if self.dimensions == 3 else mp.Vector3(y=self.sz/2-self.dpml-self.pixel_size)
            self.sim.run(until_after_sources=mp.stop_when_fields_decayed(50, self.pol, pt, threshold))

        self.has_run = True



    def plot2D(self, field=None, plane='xy'):
        """Plot simulation results in 2D."""

        if self.dimensions == 2:
            self.sim.plot2D()
            return

        if plane == 'xy':
            output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(self.sx, self.sy, 0))
        elif plane == 'xz':
            output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(self.sx, 0, self.sz))
        elif plane == 'yz':
            output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, self.sy, self.sz))
        else:
            raise ValueError('Invalid plane.')

        if field is None:
            self.sim.plot2D(output_plane=output_plane)
        else:
            if self.has_run:
                self.sim.plot2D(fields=field, output_plane=output_plane)
            else:
                raise ValueError('Simulation has not been run yet.')


    def get_s_params(self,
                     plot: str = None,
                     save: bool = False,
                     plot_title: str = None,
                     filename: str = None):
        """Get s parameters from simulation."""

        coefs1 = self.sim.get_eigenmode_coefficients(self.refl, [1]).alpha[0]
        coefs2 = self.sim.get_eigenmode_coefficients(self.tran, [1]).alpha[0]

        p1 = np.array([coef[1] for coef in coefs1]) # Reflected field
        p2 = np.array([coef[0] for coef in coefs2]) # Transmitted field
        p3 = np.array([coef[0] for coef in coefs1]) # Incident field

        S11 = p1/p3
        S21 = p2/p3

        d = self.mm_thickness
        n_subs = self.substrate.epsilon_diag[0]

        k = 2*np.pi*np.array(self.freqs)
        d_ref = self.sz/2-self.dpml-0.2-d
        d_tran = self.sz/2-self.dpml-0.1
        self.S11 = S11 * np.exp(-1j*k*(2*d_ref+0.2))
        self.S21 = S21 * np.exp(-1j*k*(d_ref+0.2)-1j*k*n_subs*d_tran)

        if save:
            self.save_s_params()
        if plot:
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
            plt.plot(f*cnt.c/1e6, np.abs(S11)**2, 'b', label='$|S_{11}|^2$')
            plt.plot(f*cnt.c/1e6, np.abs(S21)**2, 'r', label='$|S_{12}|^2$')
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



def get_mm_thickness(geometry: List[GeometricObject]):
    h_max = 0
    h_min = 0
    for block in geometry:
        if block.center.z + block.size.z/2 > h_max:
            h_max = block.center.z + block.size.z/2
        if block.center.z - block.size.z/2 < h_min:
            h_min = block.center.z - block.size.z/2

    return h_max - h_min