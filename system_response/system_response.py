import os
import numpy as np
from ocean_optics_reader.oceanopticsdatareader import OceanOpticsDataReader
from plancks_law.plancks_law import PlancksLaw
import matplotlib.pyplot as plt

class SystemResponse:
    def __init__(self, **flamp_file):
        
        if flamp_file:
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
    
    def calculate_system_response_only_data(self, lamp_wavelength, lamp_intensity):
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
    
    def correct_system_response_only_data(self, raw_data, lamp_data, save_correction=True):
        
        data_wavelength, data_intensity = raw_data
        lamp_wavelength, lamp_intensity = lamp_data
        
        warning_message = ''
        
        # lamp_intensity = np.nan_to_num(lamp_intensity, nan=0, posinf=0, neginf=0)
        system_response = self.calculate_system_response_only_data(lamp_wavelength, lamp_intensity)
        
        # Check for zero values in system_response
        zero_mask = (system_response == 0)
        system_response[zero_mask] = np.finfo(float).eps  # Replace zeros with a small non-zero value
        
        # Check for NaNs and infinities in system_response
        invalid_mask = np.isnan(system_response) | np.isinf(system_response)
        system_response[invalid_mask] = 1  # Replace NaNs and infinities with a default value
        
        # Handle division by zero errors
        data_intensity_corrected = np.where(zero_mask | invalid_mask, 0, data_intensity / system_response)
        
        # Save corrected data to a txt file
        corrected_data = np.column_stack((data_wavelength, data_intensity_corrected))
        if save_correction == True:
            np.savetxt('corrected_data.txt', corrected_data, delimiter=',')
        
        # Add a warning message to the return of function if the system response encountered any zero or NaN values or handled division by zero errors
        if np.any(zero_mask) or np.any(invalid_mask):
            warning_message = 'Warning: System response encountered zero or NaN values or handled division by zero errors.\n Errors have been handeled. Press OK to continue.'
        
        return data_wavelength, data_intensity, data_intensity_corrected, system_response, warning_message
        