# Import the required packages
from helpers import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, medfilt
import pandas as pd
import ocean_optics_reader.oceanopticsdatareader as oodr
import data_processing.dataprocessing as dp
from system_response.system_response import SystemResponse
import sys
import tkinter as tk
from plancks_law.plancks_law import PlancksLaw
import os
from system_response.system_response import SystemResponse
from tkinter import ttk, filedialog, Tk, Frame, VERTICAL, Canvas
from plot_config import plot_config

class PlancksAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Planck's Analysis GUI")

        self.oodr = oodr.OceanOpticsDataReader()

        # Create GUI widgets
        self.plot_frames = []
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.grid(row=0, column=1, padx=5, pady=5)
        
        self.data_plot_frame = self.create_plot_frame(self.left_frame, 0, 0, "Raw Data")
        self.system_response_frame = self.create_plot_frame(self.left_frame, 0, 1, "System Response")
        self.intensity_corrected_frame = self.create_plot_frame(self.left_frame, 1, 0, "Intensity Corrected Data")
        self.plancks_fit_frame = self.create_plot_frame(self.left_frame, 1, 1, "Planck's Fitting")

        # Widgets for Planck's fit
        self.fit_start_label = ttk.Label(self.right_frame, text="Fit Start (nm):")
        self.fit_start_label.grid(row=0, column=0)
        self.fit_start_value = tk.StringVar()
        self.fit_start_slider = ttk.Scale(self.right_frame, from_=300, to=900, length=200, orient=tk.HORIZONTAL, variable=self.fit_start_value, command=self.update_fit_start_value, style='TScale')
        self.fit_start_slider.set(400)
        self.fit_start_slider.grid(row=0, column=1)
        self.fit_start_entry = ttk.Entry(self.right_frame, textvariable=self.fit_start_value)
        self.fit_start_entry.grid(row=0, column=2)
        self.fit_start_display = ttk.Label(self.right_frame, textvariable=self.fit_start_value)
        self.fit_start_display.grid(row=0, column=3)

        self.fit_end_label = ttk.Label(self.right_frame, text="Fit End (nm):")
        self.fit_end_label.grid(row=1, column=0)
        self.fit_end_value = tk.StringVar()
        self.fit_end_slider = ttk.Scale(self.right_frame, from_=400, to=1000, length=200, orient=tk.HORIZONTAL, variable=self.fit_end_value, command=self.update_fit_end_value, style='TScale')
        self.fit_end_slider.set(430)
        self.fit_end_slider.grid(row=1, column=1)
        self.fit_end_entry = ttk.Entry(self.right_frame, textvariable=self.fit_end_value)
        self.fit_end_entry.grid(row=1, column=2)
        self.fit_end_display = ttk.Label(self.right_frame, textvariable=self.fit_end_value)
        self.fit_end_display.grid(row=1, column=3)

        # For fit_start_value
        self.fit_start_value.trace_add("write", lambda name, index, mode: self.update_fit_start_value())

        # For fit_end_value
        self.fit_end_value.trace_add("write", lambda name, index, mode: self.update_fit_end_value())
            
        self.temperature_label = ttk.Label(self.right_frame, text="Initial Temperature (K):")
        self.temperature_label.grid(row=2, column=0)
        self.temperature_entry = ttk.Entry(self.right_frame)
        self.temperature_entry.insert(tk.END, "1000")
        self.temperature_entry.grid(row=2, column=1)

        self.scale_factor_label = ttk.Label(self.right_frame, text="Initial Scale Factor:")
        self.scale_factor_label.grid(row=3, column=0)
        self.scale_factor_entry = ttk.Entry(self.right_frame)
        self.scale_factor_entry.insert(tk.END, "1e-6")
        self.scale_factor_entry.grid(row=3, column=1)

        self.plancks_fit_button = ttk.Button(self.right_frame, text="Perform Planck Fit", command=self.perform_plancks_fit, style='Large.TButton')
        self.plancks_fit_button.grid(row=13, column=0, columnspan=2)

        # Define the style for the button
        style = ttk.Style()
        style.configure('Large.TButton', font=('Helvetica', 20))  # Change the font size to 20
        # Define the style for the buttons
        style.configure('TButton', font=('Helvetica', 12), background='light gray')

        # Buttons for importing data files
        self.data_button = self.create_import_button(self.right_frame, "Import Data", self.import_data, _row=4, _column=0, _columnspan=2, _pady=5)
        self.system_response_button = self.create_import_button(self.right_frame, "Import System Response", self.import_system_correction, _row=5, _column=0, _columnspan=2, _pady=5)
        
        # Button for performing system intensity correction
        self.intensity_correction_button = self.create_import_button(self.right_frame, "Perform Intensity Correction", self.perform_intensity_correction, _row=6, _column=0, _columnspan=2, _pady=5)
        
        # Add a display widget to show the fit temperature with error and the scale factor with error
        self.fit_temperature_label = ttk.Label(self.right_frame, text="Fit Temperature (K):")
        self.fit_temperature_label.grid(row=10, column=0)
        self.fit_temperature_value = ttk.Label(self.right_frame, text="")
        self.fit_temperature_value.grid(row=10, column=1)
        
        self.fit_scale_factor_label = ttk.Label(self.right_frame, text="Fit Scale Factor:")
        self.fit_scale_factor_label.grid(row=11, column=0)
        self.fit_scale_factor_value = ttk.Label(self.right_frame, text="")
        self.fit_scale_factor_value.grid(row=11, column=1)
        
        # Add widget to save all the plots
        self.save_plots_button = self.create_import_button(self.right_frame, "Save Plots", self.save_plots, _row=15, _column=0, _columnspan=2, _pady=5)
        
        # Create variables to store raw data, system response, and intensity corrected data
        self.raw_data = None
        self.system_response = None
        self.intensity_corrected_data = None
        
    # Then define the update methods
    def update_fit_start_value(self, *args):
        self.fit_start_value.set("{:.2f}".format(float(self.fit_start_value.get())))

    def update_fit_end_value(self, *args):
        self.fit_end_value.set("{:.2f}".format(float(self.fit_end_value.get())))
        
    def save_plots(self):
        file_path = filedialog.asksaveasfilename(title="Save the plot", filetypes=[("PNG files", "*.png")])
        if file_path:
            # Create a new figure and add subplots for each frame
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            fig.tight_layout(pad=3.0)
            for i, (frame, fig_old, ax_old) in enumerate(self.plot_frames):
                ax = axs[i//2, i%2]
                ax.set_title(ax_old.get_title())
                ax.set_xlabel(ax_old.get_xlabel())
                ax.set_ylabel(ax_old.get_ylabel())
                ax.set_xlim(ax_old.get_xlim())
                ax.set_ylim(ax_old.get_ylim())
                # Copy the lines from the old axes to the new one
                for line in ax_old.get_lines():
                    ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), ls=line.get_linestyle(), lw=line.get_linewidth())
                try:
                    for scatter in ax_old.collections:
                        ax.scatter(scatter.get_offsets()[:, 0], scatter.get_offsets()[:, 1], label=scatter.get_label(), s=scatter.get_sizes()[0], marker=scatter.get_paths()[0], color=scatter.get_facecolors()[0])
                except AttributeError:
                    pass
                ax.legend()
            # Save the new figure with all subplots
            fig.savefig(file_path + ".png")
            plt.close(fig)  # Close the figure to free up memory

    def create_plot_frame(self, parent_frame, row, column, title):
        frame = ttk.Frame(parent_frame, borderwidth=1, relief="ridge")
        frame.grid(row=row, column=column, padx=3, pady=3)
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust the figure size as needed
        ax.set_title(title)
        fig.tight_layout(pad=1.0)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.plot_frames.append((frame, fig, ax))  # Store frame, fig, and ax as a tuple
        return frame
    
    def create_import_button(self, parent_frame, text, command, _row=5, _column=0, _columnspan=2, _pady=5):
        button = ttk.Button(parent_frame, text=text, command=command)
        button.grid(row=_row, column=_column, columnspan=_columnspan, pady=_pady)  # Adjust the row and column as needed
        return button

    def create_button(self, frame, text, command):
        button = ttk.Button(frame, text=text, command=command)
        button.pack(side=tk.TOP, pady=5)
        return button

    def import_data(self):
        file_path = filedialog.askopenfilename(title="Select the data file")
        if file_path:
            self.raw_data = (self.oodr.get_ocean_optics_data(file_path))
            self.plot_data(file_path, self.data_plot_frame)

    def import_system_correction(self):
        file_path = filedialog.askopenfilename(title="Select the system response file")
        if file_path:
            self.system_response = (self.oodr.get_ocean_optics_data(file_path))
            self.plot_system_correction(file_path)
            
    def perform_intensity_correction(self):
        if self.raw_data is not None and self.system_response is not None:
            wvl_start = 380 # Default wavelength range for system correction
            wvl_end = 940 # Default wavelength range for system correction
            
            sr = SystemResponse()
            x_data, y_data, intens_data, system_response, warning_message = sr.correct_system_response_only_data(self.raw_data,self.system_response, save_correction=False)
            
            if warning_message:
                tk.messagebox.showwarning("Warning", warning_message)
                
            start = np.argmin(abs(x_data[1:] - wvl_start))
            end = np.argmin(abs(x_data[1:] - wvl_end))
            # logger.info(f"Default wavelength range for system correction: {wvl_start} - {wvl_end} nm")
            # print(start, end)

            x_data = x_data[start:end]
            y_data = y_data[start:end]
            intens_data = intens_data[start:end]
            system_response = system_response[start:end]
            
            self.plot_intensity_corrected_data(x_data, intens_data)

    def plot_system_correction(self, system_response_file):
        fig, ax = self.get_fig_ax(self.system_response_frame)
        fig.tight_layout(pad=3.0)
        ax.clear()
        x_data, y_data = self.oodr.get_ocean_optics_data(system_response_file)
        ax.plot(x_data, y_data)
        ax.set_title("System Response")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("System Response (a.u.)")
        fig.canvas.draw()
        
    def plot_intensity_corrected_data(self, x_data, y_data):
        fig, ax = self.get_fig_ax(self.intensity_corrected_frame)
        fig.tight_layout(pad=3.0)
        ax.clear()
        ax.plot(x_data, y_data)
        ax.set_title("Intensity Corrected Data")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        fig.canvas.draw()

    def plot_data(self, file_path, plot_frame):
        fig, ax = self.get_fig_ax(plot_frame)
        fig.tight_layout(pad=3.0)
        ax.clear()
        # Plot the data
        x_data, y_data = self.oodr.get_ocean_optics_data(file_path)
        ax.plot(x_data, y_data)
        ax.set_title(os.path.basename(file_path))
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        fig.canvas.draw()

    def get_fig_ax(self, plot_frame):
        for frame, fig, ax in self.plot_frames:  # Unpack frame, fig, and ax from the tuple
            if frame == plot_frame:
                return fig, ax

    def perform_plancks_fit(self):
        
        if self.intensity_corrected_data is not None:
            (x_data, y_data) = self.intensity_corrected_data
            self.plancks_fit(x_data, y_data)
        elif self.intensity_corrected_data is None and self.raw_data is not None:
            (x_data, y_data) = self.raw_data
            self.plancks_fit(x_data, y_data)
        else:
            # Display an error message in a dialog box
            tk.messagebox.showerror("Error", "No data available for Planck's fit")
                
    def plancks_fit(self,x_data, y_data):
        fit_start = int(self.fit_start_slider.get())
        fit_end = int(self.fit_end_slider.get())
        initial_temperature = float(self.temperature_entry.get())
        scale_factor = float(self.scale_factor_entry.get())
        fig, ax = self.get_fig_ax(self.plancks_fit_frame)
        fig.tight_layout(pad=3.0)
        ax.clear()
        # x_data, y_data = self.oodr.get_ocean_optics_data(file_path)
        wa = x_data[(x_data >= fit_start) & (x_data <= fit_end)]
        cleaned_data_crop = y_data[(x_data >= fit_start) & (x_data <= fit_end)]

        plancks_law = PlancksLaw()
        try:
            fit_params, fit_errors = curve_fit(plancks_law.calculate_spectral_irradiance, wa, cleaned_data_crop, p0=[scale_factor, initial_temperature])
            fitted_scale_factor, fitted_temperature = fit_params
            scale_factor_error, temperature_error = np.sqrt(np.diag(fit_errors))
            
            # Update the fit temperature and scale factor display widgets along with their errors
            self.fit_temperature_value.config(text=f"{fitted_temperature:.2f} ± {temperature_error:.2f} K")
            self.fit_scale_factor_value.config(text=f"{fitted_scale_factor:.2e} ± {scale_factor_error:.2e}")
            
            ybest = plancks_law.calculate_spectral_irradiance(wa, fitted_scale_factor, fitted_temperature)
            yblack = plancks_law.calculate_spectral_irradiance(wa, fitted_scale_factor, initial_temperature)
            ax.scatter(wa, cleaned_data_crop / max(cleaned_data_crop), label='Intensity corrected data', s=10, c='k', marker='x')
            ax.plot(wa, yblack / max(yblack), label=f'Input Temperature {initial_temperature:.2f} K')
            ax.plot(wa, ybest / max(ybest), ls='--', label=f'Best fitting model {fitted_temperature:.2f} K')
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title("Planck's Fitting")
            ax.legend()
            fig.canvas.draw()
        except RuntimeError:
            ax.annotate("Planck's fit failed", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
            fig.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlancksAnalysisGUI(root)
    root.mainloop()
