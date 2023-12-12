import pandas as pd

class OceanOpticsDataReader:
    def __init__(self):
        pass
    
    def get_ocean_optics_data(self, f):
        with open(f, 'r') as file:
            first_line = file.readline()
            if first_line.startswith("Data from"):
                # Skip lines until the ">>>>>Begin Spectral Data<<<<<" marker is found
                for line in file:
                    if line.strip() == ">>>>>Begin Spectral Data<<<<<":
                        break
            else:
                file.seek(0)  # Return to the beginning of the file
            
            # Read the data into a DataFrame
            data = pd.read_csv(file, sep=None, engine='python', names=['col_a_data', 'col_b_data'])
        
        x = data['col_a_data'].to_numpy()
        y = data['col_b_data'].to_numpy()
    
        return x, y
    
    def get_multiple_ocean_optics_data(self, f, i):
        with open(f[i], 'r') as file:
            first_line = file.readline()
            if first_line.startswith("Data from"):
                # Skip lines until the ">>>>>Begin Spectral Data<<<<<" marker is found
                for line in file:
                    if line.strip() == ">>>>>Begin Spectral Data<<<<<":
                        break
            else:
                file.seek(0)  # Return to the beginning of the file
            
            # Read the data into a DataFrame
            data = pd.read_csv(file, sep=None, engine='python', names=['col_a_data', 'col_b_data'])
        
        x = data['col_a_data'].to_numpy()
        y = data['col_b_data'].to_numpy()
    
        return x, y
