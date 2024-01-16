## Introduction

This Python script, `plancks_analysis.py`, is designed for the analysis of spectral data using Planck's radiation law. It provides functionality for system correction and Planck's fit to extract temperature and scaling factor information from the given data. This README file will guide you through the usage of this script.

## Dependencies

Before using the script, ensure that you have the necessary Python libraries installed. The script relies on the following libraries:

* `ipywidgets`
* `IPython`
* `scipy`
* `numpy`
* `matplotlib`
* `tqdm`
* `pandas`
* `pybaselines`
* `spe2py`
* `peakutils`

You can install these libraries using the script's automatic installation feature. If any of these libraries are missing, the script will attempt to install them automatically during runtime.

## Usage

To use this script, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory containing the `plancks_analysis.py` file.
3. Run the script with the following command:

   <pre><div class="bg-black rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 dark:bg-token-surface-primary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-sm"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 4C10.8954 4 10 4.89543 10 6H14C14 4.89543 13.1046 4 12 4ZM8.53513 4C9.22675 2.8044 10.5194 2 12 2C13.4806 2 14.7733 2.8044 15.4649 4H17C18.6569 4 20 5.34315 20 7V19C20 20.6569 18.6569 22 17 22H7C5.34315 22 4 20.6569 4 19V7C4 5.34315 5.34315 4 7 4H8.53513ZM8 6H7C6.44772 6 6 6.44772 6 7V19C6 19.5523 6.44772 20 7 20H17C17.5523 20 18 19.5523 18 19V7C18 6.44772 17.5523 6 17 6H16C16 7.10457 15.1046 8 14 8H10C8.89543 8 8 7.10457 8 6Z" fill="currentColor"></path></svg>Copy code</button></span></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-bash">python plancks_analysis.py --do_system_correction <y/n> --do_planck_fit <y/n> [optional arguments]
   </code></div></div></pre>

   Replace `<y/n>` with 'y' if you want to perform the respective operation (system correction or Planck's fit) or 'n' if you don't.

   Optional arguments:

   * `--wvl_start`: Specify the start wavelength for system correction in nanometers (default: 370 nm).
   * `--wvl_end`: Specify the end wavelength for system correction in nanometers (default: 920 nm).
   * `--fit_start`: Specify the start wavelength for Planck's fit in nanometers (default: 400 nm).
   * `--fit_end`: Specify the end wavelength for Planck's fit in nanometers (default: 580 nm).
   * `--initial_temperature`: Specify the initial temperature guess in Kelvin (default: 1000 K).
   * `--scale_factor`: Specify the initial scale factor guess (default: 1e-6).
4. The script will perform the selected operations based on your input and display plots and information.

## File Selection

* The script will prompt you to select the data file containing the spectral data you want to analyze.
* If system correction is enabled (`--do_system_correction y`), you will also need to select the tungsten halogen lamp spectrum file for system correction.

## System Correction

* If system correction is enabled, the script will perform system correction on the selected data using the lamp spectrum.
* It will apply corrections based on the specified wavelength range (`--wvl_start` and `--wvl_end`).
* You can view the raw data, system response, and intensity-corrected data in plots.

## Planck's Fit

* If Planck's fit is enabled, the script will perform a fitting operation to extract temperature and scaling factor information.
* It will use the specified wavelength range (`--fit_start` and `--fit_end`) for the fitting process.
* The script will display a plot showing the intensity-corrected data, the best-fitting model, and the input temperature.
* Extracted temperature and scaling factor values will be logged and displayed.

## Data Output

* The script logs the extracted temperature and scaling factor values for each selected data file.
* The results are stored in a Pandas DataFrame and printed at the end of the script's execution.
* You can also uncomment and use the provided code to save the results to a CSV file.

## Notes

* The script provides flexibility in terms of data selection, correction, and fitting parameters.
* It handles missing library installations automatically by attempting to install them using `pip`.
* Ensure that the input data files are in the appropriate format and contain the required information for analysis.

Feel free to customize the script further to suit your specific data analysis needs or integrate it into your data processing workflow.
