# Import all the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import peakutils
from scipy import signal
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
import Plancks_Analysis.ocean_optics_reader.oceanopticsdatareader as oodr


class DataProcessing:
    
    def __init__(self):
        """
        Initialize the DataProcessing class.

        Parameters:
        - h: Planck's constant (default: 6.626e-34 Js)
        - c: Speed of light (default: 2.9979e8 m/s)
        - k: Boltzmann constant (default: 1.381e-23 J/K)
        - pixelArea: Pixel area (default: 14e-6 * 200e-6 m^2)
        """
        ## Scientific Constants

        self.h = 6.626e-34 #Planck's constant in Js
        self.heV = 4.136e-15 #Planck's constant in eVs
        self.c = 2.9979e8 #Speed of light in m/s
        self.k = 1.381e-23 #Boltzmann constant in J/K
        self.kbeV = 8.617e-5 #Boltzmann constant in eV/K
        self.scale = 2e-9
        #self.pixelArea = (20e-6)*(20e-6)*100; # LN Spectrometer Area of 100, 20um x 20um pixels per wavelength
        self.pixelArea = (14e-6)*(200e-6); # Ocean Optics UV Vis 14um x 200um per wavelength
        # self.acquisitionTime = 0.5; # Acquisition time is seconds
        
    def data_cleaning(self, data, erp):
        # Code to remove cosmic rays
        err = erp / 100

        Ndatapts = len(data)
        ydata = data

        for i in range(0, Ndatapts):  # This loop removes hot regions one pixel wide
            if i > 0 and i < Ndatapts - 1:
                if ydata[i] > (1 + err) * ydata[i + 1] and ydata[i] > (1 + err) * ydata[i - 1]:
                    ydata[i] = (ydata[i - 1] + ydata[i + 1]) / 2

        for i in range(0, Ndatapts):  # This loop removes hot regions up to 3 pixels wide
            if i > 1 and i < Ndatapts - 2 and ydata[i] > (1 + err) * ydata[i + 2] and ydata[i] > (1 + err) * ydata[
                i - 2]:
                ydata[i], ydata[i - 1], ydata[i + 1] = (ydata[i - 2] + ydata[i + 2]) / 2, (ydata[i] + ydata[i - 2]) / 2, (
                            ydata[i + 2] + ydata[i]) / 2

        for i in range(0, Ndatapts):  # This loop removes hot regions up to 5 pixels wide
            if i > 4 and i < Ndatapts - 5 and ydata[i] > (1 + err) * ydata[i + 5] and ydata[i] > (1 + err) * ydata[
                i - 5]:
                ydata[i], ydata[i - 1], ydata[i - 2], ydata[i - 3], ydata[i - 4], ydata[i + 1], ydata[i + 2], ydata[
                    i + 3], ydata[i + 4] = (ydata[i - 5] + ydata[i + 5]) / 2, (ydata[i] + ydata[i - 2]) / 2, (
                                                          ydata[i - 1] + ydata[i - 3]) / 2, (
                                                          ydata[i - 2] + ydata[i - 4]) / 2, (
                                                          ydata[i - 3] + ydata[i - 5]) / 2, (
                                                          ydata[i + 2] + ydata[i]) / 2, (
                                                          ydata[i + 3] + ydata[i + 1]) / 2, (
                                                          ydata[i + 4] + ydata[i + 2]) / 2, (
                                                          ydata[i + 5] + ydata[i + 3]) / 2

        for i in range(0, Ndatapts):  # This loop removes dead regions one pixel wide
            if i > 0 and i < Ndatapts - 1 and ydata[i] < (1 - err) * ydata[i + 1] and ydata[i] < (1 - err) * ydata[i - 1]:
                ydata[i] = (ydata[i - 1] + ydata[i + 1]) / 2

        for i in range(0, Ndatapts):  # This loop removes dead regions up to 3 pixels wide
            if i > 1 and i < Ndatapts - 2 and ydata[i] < (1 - err) * ydata[i + 2] and ydata[i] < (1 - err) * ydata[
                i - 2]:
                ydata[i], ydata[i - 1], ydata[i + 1] = (ydata[i - 2] + ydata[i + 2]) / 2, (ydata[i] + ydata[i - 2]) / 2, (
                            ydata[i + 2] + ydata[i]) / 2

        return ydata

    def subtract_bsl(self, data, p_order):
        base = peakutils.baseline(data, p_order)
        return base

    def filter_savgol(self, data, k_size, p_order):
        data[np.isnan(data)] = 0
        sav_filt_data = signal.savgol_filter(data, k_size, p_order)
        return sav_filt_data

    def filter_median(self, data, k_size):
        filt_data = medfilt(data, kernel_size=k_size)
        return filt_data

    def data_averaging(self, selectFiles, average, batching, batch_size,
                       final_file_name='-averaged_data.csv'):
        fdata = selectFiles.files

        if len(fdata) < 2:
            print("Please select at least 2 files to average")

        head_i, tail_i = os.path.split(fdata[0])
        y_all = []
        for idx, file in enumerate(fdata):
            x_data, y_data, metadata = DataReader(file_name=file).read_file()
            y_all.append(y_data)

        column_vectors = [np.expand_dims(arr, axis=1) for arr in y_all]

        # Concatenate the column vectors along the axis=1
        concatenated_array = np.concatenate(column_vectors, axis=1)
        average_array = concatenated_array

        average_batches = []

        if batching == True:
            # Get the total number of elements and batches
            total_elements = concatenated_array.shape[1]
            total_batches = total_elements // batch_size

            # Calculate the median within each batch
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch_concat = np.concatenate(column_vectors[:][start_idx:end_idx], axis=1)
                batch_average = np.average(batch_concat, axis=1)
                average_batches.append(batch_average)

                # Save each batch_median to a CSV file
                batch_filename = f'-batch_average_{i + 1}.csv'
                total_median_data = np.column_stack((x_data, batch_average))
                np.savetxt(os.path.join(head_i, tail_i[:-13] + batch_filename), total_median_data, delimiter=',')

        averaged_data = np.average(y_all, axis=1)
        total_average_data = np.column_stack((x_data, averaged_data))
        np.savetxt(os.path.join(head_i, tail_i[:-13] + final_file_name), total_average_data, delimiter=',')

        return x_data, averaged_data
    
    def i_corr(self, f, flamp):
        start = 0
        end = -1

        head_i, tail_i = os.path.split(f)
        
        data_reader = oodr.OceanOpticsDataReader()

        ## Important note to change the location of this .txt file below to where it is in your computer.
        calibstd = np.loadtxt(r"G:\Shared drives\Pauzauskie Team Drive\Users\CG\Scripts\030410638_HL-2000-CAL_2014-01-15-14-09_VISEXT1EXT2_FIB.txt")
        xcalib = calibstd[:,0]
        ycalib = calibstd[:,1]

        x, y = data_reader.get_ocean_optics_data(f)

        HglampFunc = CubicSpline(xcalib, ycalib)
        hglampI = HglampFunc(x)*1e6 # Create interpolation of true lamp spectrum

        hglampdata_x, hglampdata_y = data_reader.get_multiple_ocean_optics_data(flamp, 0) # Split true lamp spectra into x and y
        # print(hglampdata_y)

        ICF = (hglampI) / (hglampdata_y) # Creates ratio of true lamp spectra to real lamp data, ICF = Intensity Correction Factor
        ynew = (y) * ICF # multiplies real data by intensity correction factor

        # ynew = np.nan_to_num(ynew, nan=0, posinf=0, neginf=0)
        # # datamatrix = np.column_stack((x, ynew)) # Compiles corrected data into a new matrix
        # start = 400
        # end = 800
        # x = x[start:end]
        # y_new = ynew[np.where((x >= start) & (x <= end))]
        
        fig1, ax1 = plt.subplots()
        ax1.plot(x, ynew, label='Intensity Corrected Data')
        ax1.set_xlim(400,800)
        ax1.set_ylim(0,0.5e6)
        

        # fig2, ax2 = plt.subplots()
        # # ax2.plot(x, ynew / np.max(ynew), label='Intensity Corrected Data')
        # # plt.plot(x, y / np.max(y), label='Actual Data')
        # ax2.plot(x, ICF, label='ICF')
        # ax2.set_xlim(400,800)
        # ax2.set_ylim(0,0.0002)
        # ax2.legend()
        
        # fig3, ax3 = plt.subplots()
        # ax3.plot(x, ICF, label='Intensity Corrected Data')
        # ax3.set_ylabel('ICF * 1e4')
        # plt.show()

        return x, y, ynew
