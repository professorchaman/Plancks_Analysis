import numpy as np
class PlancksLaw:
    def __init__(self):
        self.h = 6.62607015 * 10**-34 # Planck's constant
        self.c = 299792458 # Speed of light
        self.k = 1.380649 * 10**-23 # Boltzmann constant

    def calculate_spectral_irradiance(self, wavelength_nm, scale_factor, temperature):
        # wavelength_m = wavelength_nm * 1e-9 # Convert wavelength from nm to m 
        spectral_irradiance = scale_factor * (119.029*1e27) / (wavelength_nm**5) * (1 / (np.exp((14.3838*1e6)/(wavelength_nm*temperature)) - 1))
    
        return spectral_irradiance
    
    
        