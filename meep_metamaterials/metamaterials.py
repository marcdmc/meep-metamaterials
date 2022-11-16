import meep as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, datetime
import pickle
from meep import mpb
from IPython.display import Video
from typing import Callable, List, Tuple, Union, Optional
from meep.geom import Vector3, init_do_averaging, GeometricObject, Medium

from . import constants as cnt

class MetamaterialSimulation():
    def __init__(self,
                 period: float,
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
                 layers: int = 1,
                ):

        self.period = period
        self.geometry = geometry
        self.source = source

        self.substrate = substrate
        self.substrate2 = substrate2
        self.substrate_n_range = substrate_n_range
        self.substrate2_n_range = substrate2_n_range

        self.period_x = period_x if period_x is not None else period
        self.period_y = period_y if period_y is not None else period
        self.period_z = period_z if period_z is not None else period

        self.dimensions = dimensions
        self.pol = pol
        self.resolution = resolution

        self.pixel_size = 1/self.resolution

        self.mm_thickness = get_mm_thickness(geometry, self.dimensions) # Thickness of metamaterial
        # Add some depth if no geometry is detected (ex: for material function)
        self.mm_thickness = self.period_z if self.mm_thickness == 0 else self.mm_thickness

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

        # If layers are specified, add them to the geometry
        if layers > 1:
            self.geometry = self.add_layers(self.geometry, layers-1, separation=self.period_z)
            # Update whole metamaterial thickness
            self.mm_thickness = layers*self.period_z + self.mm_thickness

        # Depth of the simulation
        self.depth = 2/self.fmin+self.mm_thickness if self.source_type == 'gaussian' else 2*self.wvl+self.mm_thickness

        # Simulation domain size
        self.sx = self.period
        self.sy = self.depth if dimensions == 2 else self.period
        self.sz = 0 if dimensions == 2 else self.depth

        # Absorbing boundary conditions
        self.dpml = 1/self.fmin if self.source_type == 'gaussian' else 1/self.fcen
        if dimensions == 2:
            self.sy += 2*self.dpml
            self.depth += 2*self.dpml
        else:
            self.sz += 2*self.dpml
            self.depth += 2*self.dpml

        # Automatically add source
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

        # Transform according to dimensions
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
            self.source = source_2d
        elif dimensions == 3:
            self.cell = mp.Vector3(self.sx, self.sy, self.sz)
        else:
            raise ValueError('Too many dimensions????')
        

        # Add base substrate if substrate2 is different from substrate
        # Now the metamaterial will be on top of the substrate2
        # TODO: options for the metamaterial to be on top or fully embedded
        if self.substrate2 != self.substrate:
            if self.dimensions == 2:
                sb = mp.Block(size=mp.Vector3(mp.inf, self.depth/2),
                              center=mp.Vector3(y=-self.depth/4))
                self.geometry.insert(0, sb)
            elif self.dimensions == 3:
                sb = mp.Block(size=mp.Vector3(mp.inf, mp.inf, self.depth/2),
                              center=mp.Vector3(z=-self.depth/4))
                self.geometry.insert(0, sb)


        # Create simulation
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
        refl_center = -self.depth/2+self.dpml+0.2*self.depth/4
        tran_center = self.depth/2-self.dpml-0.1*self.depth/4
        if self.dimensions == 2:
            refl_fr = mp.ModeRegion(center=mp.Vector3(y=refl_center), size=mp.Vector3(self.sx))
            tran_fr = mp.ModeRegion(center=mp.Vector3(y=tran_center), size=mp.Vector3(self.sx))
        elif self.dimensions == 3:
            refl_fr = mp.ModeRegion(center=mp.Vector3(z=refl_center), size=mp.Vector3(self.sx, self.sy))
            tran_fr = mp.ModeRegion(center=mp.Vector3(z=tran_center), size=mp.Vector3(self.sx, self.sy))

        self.refl_fr = refl_fr
        self.tran_fr = tran_fr

        fcen = 0.5*(self.fmin+self.fmax)
        self.refl = self.sim.add_mode_monitor(fcen, self.fmax-self.fmin, self.nfreqs, refl_fr)
        self.tran = self.sim.add_mode_monitor(fcen, self.fmax-self.fmin, self.nfreqs, tran_fr)

        self.has_run = False


    def run(self,
            timesteps: int = None,
            threshold: float = 1e-3,
            animate: bool = False):
        """
        Runs the simulation.
        """
        if not animate:
            if timesteps is not None:
                self.sim.run(until=timesteps)
            else:
                pt = mp.Vector3(z=self.sz/2-self.dpml-self.pixel_size) if self.dimensions == 3 else mp.Vector3(y=self.sz/2-self.dpml-self.pixel_size)
                self.sim.run(until_after_sources=mp.stop_when_fields_decayed(50, self.pol, pt, threshold))
        elif animate:
            if timesteps is not None:
                self.sim.run(mp.at_every(0.4/10,animate), until=timesteps)
            else:
                animate = mp.Animate2D(self.sim, fields=self.pol, normalize=True) # Init animation
                pt = mp.Vector3(z=self.sz/2-self.dpml-self.pixel_size) if self.dimensions == 3 else mp.Vector3(y=self.sz/2-self.dpml-self.pixel_size)
                self.sim.run(mp.at_every(0.4/10, animate),
                             until_after_sources=mp.stop_when_fields_decayed(50, self.pol, pt, threshold))
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"animation-{timestamp}.mp4"
            fps = 10
            animate.to_mp4(fps,filename)
            Video(filename)

        self.has_run = True



    def plot2D(self, field=None, plane='xy', save: bool = False, filename: str = ''):
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

        if save:
            filename = filename if filename != '' else 'plot2D.png'
            plt.savefig(filename)

    def get_s_params(self,
                     plot: str = None,
                     save: bool = False,
                     plot_title: str = None,
                     filename: str = None):
        """Get s parameters from simulation."""

        # Get reflection and transmission data
        coefs1 = self.sim.get_eigenmode_coefficients(self.refl, [1]).alpha[0]
        coefs2 = self.sim.get_eigenmode_coefficients(self.tran, [1]).alpha[0]

        p1 = np.array([coef[1] for coef in coefs1]) # Reflected field
        p2 = np.array([coef[0] for coef in coefs2]) # Transmitted field
        p3 = np.array([coef[0] for coef in coefs1]) # Incident field

        self.raw_coefs = [p1, p2, p3]

        # S parameters
        S11 = p1/p3
        S21 = p2/p3

        self.raw_coefs = [p1, p2, p3, S11, S21] # Save for debugging

        k = 2*np.pi*np.array(self.freqs)
        d = self.mm_thickness
        n_subs = self.substrate.epsilon_diag[0]
        n_subs2 = self.substrate2.epsilon_diag[1]
        # Distances from source and monitors to material slab
        d_from_source = np.abs(self.source.center.z)-d/2 if self.dimensions == 3 else np.abs(self.source.center.y)-d/2
        d_refl = np.abs(self.refl_fr.center.z)-d/2 if self.dimensions == 3 else np.abs(self.refl_fr.center.y)-d/2
        d_tran = np.abs(self.tran_fr.center.z)-d/2 if self.dimensions == 3 else np.abs(self.tran_fr.center.y)-d/2

        self.d_from_source = d_from_source
        self.d_refl = d_refl
        self.d_tran = d_tran

        # Compensate phase shift due to distance from the source and monitors
        self.S11 = S11 * np.exp(-1j*k*2*d_refl*n_subs)
        self.S21 = S21 * np.exp(-1j*k*d_refl*n_subs-1j*k*d_tran*n_subs2)

        if save:
            self.save_s_params()
        if plot:
            self.plot_s_params(plot, plot_title, filename)

        return [self.S11, self.S21]

    # Alias of the previous function
    # TODO: Can this alias thing be done in a more pro or cool way?
    def get_s_parameters(self, plot: str = None,
                         save: bool = False,
                         plot_title: str = None,
                         filename: str = None):
        return self.get_s_params(plot, save, plot_title, filename)

    
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

    
    def reset_meep(self):
        """Reset simulation."""
        self.sim.reset_meep()


    def calculate_bands(self, num_bands, k_points, interpolate=4, use_meep=False, save=False, plot=False, plot_title=None, filename=None):
        # TODO: Allow non rectangular unit cells (and by default calculate the important k points)
        # TODO: Allow for 3D simulations
        if self.dimensions == 3:
            raise Exception('3D bands not implemented yet')

        if use_meep:
            return self._calculate_bands_meep(num_bands, k_points, interpolate)

        geometry_lattice = mp.Lattice(size=mp.Vector3(self.period_x, self.period_y))

        # Interpolate between k points
        k_points = mp.interpolate(8, k_points)

        # Initialize mode solver
        self.ModeSolver = mpb.ModeSolver(
            geometry=self.geometry,
            geometry_lattice=geometry_lattice,
            k_points=k_points,
            resolution=self.resolution,
            num_bands=num_bands,
            default_material=self.substrate,
        )

        # Calculate TE bands
        self.ModeSolver.run_te()
        te_bands = self.ModeSolver.all_freqs
        # Calculate TM bands
        self.ModeSolver.run_tm()
        tm_bands = self.ModeSolver.all_freqs

        if plot:
            self.plot_bands([te_bands, tm_bands], plot_title, filename)
        if save:
            # Save with pickle
            with open(self.dir + filename + '.pickle', 'wb') as f:
                pickle.dump([te_bands, tm_bands], f)

        return [te_bands, tm_bands, num_bands]

    def _calculate_bands_meep(self, num_bands, k_points, interpolate=4):
        """Calculate bands using meep (point by point)."""

        # TODO: change this to a user defined frequency or reasonable default
        fcen = 1
        df = 2

        # Create momentary simulation
        source = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df), component=mp.Ez, center=mp.Vector3())]
        sim = mp.Simulation(
            cell_size=mp.Vector3(self.period_x, self.period_y),
            geometry=self.geometry,
            resolution=self.resolution,
            default_material=self.substrate,
            k_point=mp.Vector3(),
        )

        k_interp = interpolate
        k_points = mp.interpolate(k_interp, k_points)
        freqs = sim.run_k_points(300, k_points)

        # TODO: Make this optional
        # Save
        with open('freqs_te.pickle', 'wb') as f:
            pickle.dump(freqs, f)

        # Plot
        ks = np.linspace(-1, 1, k_interp+2)
        for i in range(k_interp+2):
            for w in freqs[i]:
                plt.plot(ks[i], w, '.k')
        plt.xlim([-1, 1])
        plt.ylim([1.3, 1.7])
        plt.savefig('Marc/LandauLevel/meep/freqs_te.png')

        return [freqs, num_bands]

    def plot_bands(self, bands, k_points, plot_title=None, filename=None):
        te_bands = bands[0]
        tm_bands = bands[1]
        num_bands = bands[2]

        # Get three periods of the geometry
        md = mpb.MPBData(rectify=True, periods=3, resolution=32)
        eps = self.ModeSolver.get_epsilon()
        converted_eps = md.convert(eps)

        fig, ax1 = plt.subplots()

        left, bottom, width, height = [0.7, 0.65, 0.2, 0.2]
        ax2 = fig.add_axes([left, bottom, width, height])

        for i in range(num_bands):
            te_band = [b[i] for b in te_bands]
            tm_band = [b[i] for b in tm_bands]
            ax1.plot(te_band, 'r')

        ax2.imshow(converted_eps.T, interpolation='spline36', cmap='binary')
        plt.xticks([])
        plt.yticks([])
        ax1.grid()

        plt.show()
        plt.savefig(filename+'.png')

        return


def get_mm_thickness(geometry: List[GeometricObject], dimensions: int = 3) -> float:
    """Given a geometry returns the thickness of the metamaterial layer."""
    h_max = 0
    h_min = 0
    if dimensions == 3:
        for block in geometry:
            if block.center.z + block.size.z/2 > h_max:
                h_max = block.center.z + block.size.z/2
            if block.center.z - block.size.z/2 < h_min:
                h_min = block.center.z - block.size.z/2
    elif dimensions == 2:
        for block in geometry:
            if block.center.y + block.size.y/2 > h_max:
                h_max = block.center.y + block.size.y/2
            if block.center.y - block.size.y/2 < h_min:
                h_min = block.center.y - block.size.y/2

    return h_max - h_min

def mm_hmax(geometry: List[GeometricObject], dimensions: int = 3) -> float:
    """Given a geometry returns the maximum height of the metamaterial layer in the propagation axis."""
    h_max = 0
    if dimensions == 3:
        for block in geometry:
            if block.center.z + block.size.z/2 > h_max:
                h_max = block.center.z + block.size.z/2
    elif dimensions == 2:
        for block in geometry:
            if block.center.y + block.size.y/2 > h_max:
                h_max = block.center.y + block.size.y/2

    return h_max

def mm_hmin(geometry: List[GeometricObject], dimensions: int = 3) -> float:
    """Given a geometry returns the minimum height of the metamaterial layer in the propagation axis."""
    h_min = 0
    if dimensions == 3:
        for block in geometry:
            if block.center.z - block.size.z/2 < h_min:
                h_min = block.center.z - block.size.z/2
    elif dimensions == 2:
        for block in geometry:
            if block.center.y - block.size.y/2 < h_min:
                h_min = block.center.y - block.size.y/2

    return h_min


def add_layers(geometry: List[GeometricObject], nlayers: int, separation: float):
    """Add nlayers of the given geometry separated by a distance."""
    assert nlayers > 0
    h = get_mm_thickness(geometry)
    result = []
    for i in range(nlayers+1):
        for block in geometry:
            shift = mp.Vector3(0, 0, separation*(i-nlayers/2))
            result.append(mp.Block(size=block.size, center=block.center+shift, material=block.material))

    return result