import importlib

# List of required libraries
libraries = [
    'ipywidgets',
    'IPython',
    'scipy',
    'numpy',
    'matplotlib',
    'tqdm',
    'pandas',
    'pybaselines',
    'spe2py'
]

# Check if each library is installed, and install if necessary
for library in libraries:
    try:
        importlib.import_module(library)
        print(f"{library} is already installed")
    except ImportError:
        print(f"{library} is not installed. Installing...")
        os.system("pip install {library}")
        print(f"{library} installed successfully")

# Import the required packages
from helpers import *
from ipywidgets import *
from IPython.display import display, Javascript
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.signal import medfilt
from scipy import signal
import peakutils
from peakutils.plot import plot as pplot
from matplotlib.animation import FuncAnimation
from scipy import stats
from numpy.fft import fft, fftfreq, ifft
import glob
from tqdm import tqdm
import pandas as pd
import Plancks_Analysis.ocean_optics_reader.oceanopticsdatareader as oodr
import Plancks_Analysis.data_processing.dataprocessing as dp
from system_response import SystemResponse
import sys
import tkinter as tk
from tkinter import filedialog
from Plancks_Analysis.plancks_law.plancks_law import PlancksLaw

## Scientific Constants

h = 6.626e-34 #Planck's constant in Js
heV = 4.136e-15 #Planck's constant in eVs
c = 2.9979e8 #Speed of light in m/s
kb = 1.381e-23 #Boltzmann constant in J/K
kbeV = 8.617e-5 #Boltzmann constant in eV/K
scale = 2e-9
#pixelArea = (20e-6)*(20e-6)*100; # LN Spectrometer Area of 100, 20um x 20um pixels per wavelength
pixelArea = (14e-6)*(200e-6); # Ocean Optics UV Vis 14um x 200um per wavelength
# acquisitionTime = 0.5; # Acquisition time is seconds

do_system_correction = sys.argv[1]

# Create a Tkinter root window (it won't be displayed)
root = tk.Tk()
root.withdraw()

# Function to open a file dialog and return the selected file path
def select_file(type):
    file_path = filedialog.askopenfilename(title=f"Select the {type} file")
    return file_path

print("Select data files:")
files = select_file(type='data')
display(files)

flamp = None
if do_system_correction == 'y':
    print("\nSelect tungsten halogen (Blue Ocean optics HL2000) lamp spectrum for intensity correction \n(optional; if none is chosen, default intensity correction factor will be used): ")
    flamp = select_file(type='lamp')
    display(flamp)
    print(f"Selected halogen lamp spectrum file: {flamp}")
    print("________________________________________________________________________________________________________")
else:
    pass

## Variables for Planck's fitting

acquisitionTime = 0.0007 # 500ms acquistion time

do_system_correction = 'y'
do_intensity_correction = 'n'
do_baseline_subtraction = 'n'
do_median_filtering = 'n'
do_data_cleaning = 'n'
do_planck_fit = 'y' # Do Planck's fit
p_order = 5 # Polynomial order for baseline subtraction
k_size = 3 # Kernel size for Median Filtering
erp = -11

wvl_start = 370 # Wavelength range for correction in nm
wvl_end = 920 # Wavelength range for correction in nm

fit_start = 400 # Wavelength range for fitting in nm
fit_end = 580 # Wavelength range for fitting in nm

T0 = 1000 # Initial temperature guess in degrees K
scale_factor = 1e-6 # Scale factor for fitting

# Create instances of the classes
data_reader = oodr.OceanOpticsDataReader()
data_processor = dp.DataProcessing()

df_1 = pd.DataFrame(columns=['file_name','Scaling Factor' ,'Temperature(K)'])
counter = 0

files = files
flamp = flamp
# files = glob.glob(r"H:\My Data\LH_data\05102023\G4_CA-Exp_16-LH_R3-loc_1_Subt16__28**__39494.txt")
# flamp = glob.glob(r"H:\My Data\LH_data\Intensity Correction\Intensity_correction-300ms_Subt12__20__021_cleaned.txt")

for file in tqdm([files]):
    # Create a figure and axes with a 2x2 grid layout
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout(pad=3.0)
    
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('Integration Time (sec):'):
                integration_time = line.split(':')[1].strip()
                acquisitionTime = float(integration_time)
                # print(f'Integration time: {acquisitionTime} s')

    head, tail = os.path.split(file)
    
    counter = counter + 1

    x_data, y_data = data_reader.get_ocean_optics_data(file)
    intens_data = np.zeros_like(y_data)
    system_response = np.zeros_like(y_data)
    
    if do_system_correction == 'y':
        sr = SystemResponse([flamp])
        x_data, y_data, intens_data, system_response = sr.correct_system_response(file, save_correction=False)
        start = np.argmin(abs(x_data[1:] - wvl_start))
        end = np.argmin(abs(x_data[1:] - wvl_end))
        # print(start, end)
        x_data = x_data[start:end]
        y_data = y_data[start:end]
        intens_data = intens_data[start:end]
        system_response = system_response[start:end]
        
    if do_intensity_correction == 'y':
        x_data, y_data, intens_data = data_processor.i_corr(file, flamp)
    else:
        intens_data = intens_data

    norm_intens_data = intens_data

    if do_median_filtering == 'y':
        filt_data = data_processor.filter_median(norm_intens_data, k_size)
    else:
        filt_data = norm_intens_data

    if do_baseline_subtraction == 'y':
        base = data_processor.subtract_bsl(filt_data, p_order)
    else:
        base = 0

    bsl_subt_data = filt_data - base

    if do_data_cleaning == 'y':
        cleaned_data = data_processor.data_cleaning(bsl_subt_data, erp)
    else:
        cleaned_data = bsl_subt_data

    # Plancks fitting process
    
    if do_planck_fit == 'y':

        pixelResolution = (x_data[-1] - x_data[-2])  # delta wavelength nm
        # print('Pixel Resolution = ' + str(pixelResolution))

        y_spec_irr = np.zeros(shape=len(x_data))
        photonFlux = np.zeros(shape=len(cleaned_data))
        for j in range(len(x_data)):
            photonFlux[j] = cleaned_data[j] / (pixelArea * acquisitionTime)
            y_spec_irr[j] = photonFlux[j] * (h * c / (x_data[j] * 10 ** -9)) / (pixelResolution)  # spectral irradiance in W/m^2-nm

        # y_spec_irr_norm = y_spec_irr
        y_spec_irr_norm = cleaned_data

        # print(x)
        start = np.argmin(abs(x_data[1:] - fit_start))
        end = np.argmin(abs(x_data[1:] - fit_end))
        # print(start, end)
        wa = x_data[start:end]
        cleaned_data_crop = y_spec_irr_norm[start:end]

        # Create an instance of the PlancksLaw class
        plancks_law = PlancksLaw()
        
        # Initial temperature guess in degrees K using Wien's displacement law
        T0 = (29*1e5)/(wa[np.argmax(cleaned_data_crop)]+100)
        # T0 = 2413.31 # Initial temperature guess in degrees K
        
        scale_factor = max(cleaned_data_crop) / max(plancks_law.calculate_spectral_irradiance(wa, 1, T0)) # Scale factor guess for fitting
        # scale_factor = 1e3 # Scale factor for fitting
        
        try: # Try to fit the data to the blackbody radiation function
            fit_params, _ = curve_fit(plancks_law.calculate_spectral_irradiance, wa, cleaned_data_crop, p0=[scale_factor, T0])
            # Extract the fitted parameters
            fitted_scale_factor, fitted_temperature = fit_params

            # get the best fitting parameter values and their 1 sigma errors
            # (assuming the parameters aren't strongly correlated).

            ybest = plancks_law.calculate_spectral_irradiance(wa, fitted_scale_factor ,fitted_temperature)
            yblack = plancks_law.calculate_spectral_irradiance(wa, fitted_scale_factor ,T0)
            
            # plot the solution
            ax[1, 1].set_title(f"Planck's Fitting")
            ax[1, 1].scatter(wa, cleaned_data_crop/max(cleaned_data_crop), label='Intensity corrected data', s=10, c='k', marker='x')
            ax[1, 1].plot(wa,yblack/max(yblack),label=f'Input Temperature {T0:.2f} K')
            ax[1, 1].plot(wa, ybest/max(ybest), ls= '--',label=f'Best fitting model {fitted_temperature:.2f} K')
            ax[1, 1].set_xlabel("Wavelength (nm)")
            ax[1, 1].set_ylabel("Intensity (a.u.)")
            ax[1, 1].legend()
            
            df_1.loc[counter] = [tail, fitted_scale_factor, fitted_temperature]
            
        except RuntimeError: # If the fit fails, print an error message
            print(f"Error - curve_fit failed for {tail}")
            ax[1, 1].annotate("Planck's fit failed", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
            continue
        
    else:
        # Annotate in the center of the plot: "No Planck's fit performed"
        ax[1, 1].annotate("No Planck's fit performed", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
        # ax[1, 1].axis('off')
    
    ax[0, 0].set_title(f"Raw Data")
    ax[0, 0].plot(x_data, y_data, label='Raw data')
    ax[0, 0].set_xlabel("Wavelength (nm)")
    ax[0, 0].set_ylabel("Intensity (a.u.)")
    ax[0, 0].legend()
    
    if do_system_correction == 'y':
        ax[0, 1].set_title(f"System Response")
        ax[0, 1].plot(x_data, system_response, label='System Response')
        ax[0, 1].set_xlabel("Wavelength (nm)")
        ax[0, 1].set_ylabel("System Response (a.u.)")
        ax[0, 1].legend()
    
        ax[1, 0].set_title(f"Intensity Corrected Data")
        ax[1, 0].plot(x_data, cleaned_data, label='Intensity corrected data')
        ax[1, 0].set_xlabel("Wavelength (nm)")
        ax[1, 0].set_ylabel("Intensity (a.u.)")
        ax[1, 0].legend()
        
    else: # Annotate plot with "No intensity correction performed"
        ax[1, 0].annotate("No intensity correction performed", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
        ax[0, 1].annotate("No system response calculated", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
print(df_1)
# df_1.to_csv(r'temp_val_2.csv',index=False,header=True)