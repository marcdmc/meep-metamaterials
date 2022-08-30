import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import time

from . import constants as cnst
from . import aux

class S_parameters:
    def __init__(self, freqs, S11, S21):
        """
        Initialize S parameters.

        Attributes:
        - `freqs` (`float`): Frequency in Hz.
        - `S11` (`float`): S11.
        - `S21` (`float`): S21.
        """
        self.freqs = np.array(freqs)*cnst.C/1e6 # Convert to dimensionless units
        self.S11 = np.array(S11, dtype=complex)
        self.S21 = np.array(S21, dtype=complex)

        self.wl = 1/self.freqs # Wavelength in µm
        self.R = np.abs(self.S11)**2 # Reflection
        self.T = np.abs(self.S21)**2 # Transmission

    def get_freqs(self):
        return self.freqs

    def plot(self, plot, plot_title):
        if not hasattr(self, 'S11') or not hasattr(self, 'S21'):
            raise Exception('S parameters not set')

        f = np.array(self.freqs)
        R = self.R
        T = self.T

        if plot == 'freqs':
            print('Plotting S parameters...')
            plt.figure()
            plt.plot(f, R, 'b', label='$|S_{11}|^2$')
            plt.plot(f, T, 'r', label='$|S_{12}|^2$')
            plt.xlabel('f (THz)')
            plt.ylabel('Transmission and reflection')
            if plot_title is not None:
                plt.title(plot_title)
            plt.legend()
        elif plot == 'wl':
            plt.figure()
            plt.plot(1/f, R, 'b', label='$|S_{11}|^2$')
            plt.plot(1/f, T, 'r', label='$|S_{12}|^2$')
            plt.xlabel('Wavelength (µm)')
            plt.ylabel('Transmission and reflection')
            # plt.title('S parameters for a=' + str(self.cell) + ' µm and d=' + str(self.d) + ' µm')
            if plot_title is not None:
                plt.title(plot_title)
            plt.legend()

        timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
        if self.pol == 0:
            plt.savefig(self.dir + 's_params_TE' + timestr +'.png')
        elif self.pol == 1:
            plt.savefig(self.dir + 's_params_TM' + timestr +'.png')
        print("S parameters saved to " + self.dir + 's_params_' + timestr +'.png')

    
    def get_resonance(self):
        if not hasattr(self, 'S11') or not hasattr(self, 'S21'):
            raise Exception('S parameters not set')

        S11 = self.S11
        S21 = self.S21

        # Find reflection peak
        peak_idx = np.argmax(np.abs(S11)**2)
        f_res = self.freqs[peak_idx] # Resonant frequency
        wl_res = self.wl[peak_idx] # Resonant wavelength

        # Approximate "strength" of resonance peak
        peak = np.max(np.abs(S11)**2)

        # Find the approximate width of the resonance (fwhm)
        half_idx = aux.find_nearest(self.R, peak/2)
        fwhm = 2*np.abs(self.wl[peak_idx] - self.wl[half_idx])


        resonance = Resonance(f_res, peak, fwhm)

        return resonance

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]



class Resonance:
    def __init__(self, frequency, peak, fwhm):
        """
        Initialize resonance.

        Attributes:
        - `frequency` (`float`): Resonant frequency in Hz.
        - `peak` (`float`): Resonant peak.
        - `fwhm` (`float`): Resonant FWHM.
        """
        self.frequency = frequency
        self.peak = peak
        self.fwhm = fwhm
        
