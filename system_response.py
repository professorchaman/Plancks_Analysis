import os
import numpy as np
from oceanopticsdatareader import OceanOpticsDataReader
from plancks_law import PlancksLaw
import matplotlib.pyplot as plt

class SystemResponse:
    def __init__(self, flamp_file):
        self.flamp_file = flamp_file
        self.temperature = 2413.41 # Temperature of the lamp in Kelvin

    def calculate_system_response(self):
        # Read lamp data
        lamp_data_reader = OceanOpticsDataReader()
        lamp_wavelength, lamp_intensity = lamp_data_reader.get_ocean_optics_data(self.flamp_file[0])

        # Compute Planck's function at given temperature
        plancks_law = PlancksLaw()
        plancks_function = plancks_law.calculate_spectral_irradiance(lamp_wavelength, 1, self.temperature)

        # Calculate system response
        system_response = lamp_intensity / plancks_function

        return system_response

    def correct_system_response(self, data_file, save_correction=True):
        
        # Split file name into head and tail
        head, tail = os.path.split(data_file)
        
        # Calculate system response
        system_response = self.calculate_system_response()
        
        # Read data file
        data_reader = OceanOpticsDataReader()
        data_wavelength, data_intensity = data_reader.get_ocean_optics_data(data_file)
        data_intensity_corrected = data_intensity / system_response
        
        # Save corrected data to a txt file
        corrected_data = np.column_stack((data_wavelength, data_intensity_corrected))
        if save_correction == True:
            np.savetxt(os.path.join(head, tail[:-4] + '_corrected.txt'), corrected_data, delimiter=',')
            
        return data_wavelength, data_intensity, data_intensity_corrected, system_response
