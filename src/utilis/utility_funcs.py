import datetime
from netCDF4 import Dataset
from pathlib import Path
import pyproj 
import os

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def num_unique_pairs(arr):
    """
    Count the number of unique pairs in a list.

    Args:
        arr (list): List of elements.

    Returns:
        int: Number of unique pairs.
    """
    return len(arr) * (len(arr) - 1) // 2



def from_timestamp_to_date(timestamp):
    """
    Convert a timestamp to a date string.

    Args:
        timestamp (int): Timestamp in seconds.

    Returns:
        str: Date string in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')





def get_midnor_projection():
    """
    Function to set up the MIDNOR projection using pyproj.
    """
    # Load the MIDNOR dataset
    # Assuming 'physStates.nc' is in the current working directory
    # Adjust the path as necessary
    work_dir = get_project_root()
    print(f"Current working directory: {work_dir}")
    file_path = "data/sinmod/physStates.nc"
    file_pathh = os.path.join(work_dir, file_path)
    print(f"Loading file from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_pathh} does not exist.")
    MIDNOR_df = Dataset(file_pathh, 'r')

    MIDNOR_GM = MIDNOR_df.variables['grid_mapping']
    MIDNOR_xc = MIDNOR_df.variables['xc']
    MIDNOR_yc = MIDNOR_df.variables['yc']
    
    x0_MIDNOR = MIDNOR_GM.false_easting
    y0_MIDNOR = MIDNOR_GM.false_northing
    dx_MIDNOR = dy_MIDNOR = MIDNOR_GM.horizontal_resolution
    lat_0_MIDNOR = MIDNOR_GM.latitude_of_projection_origin
    lon_0_MIDNOR = MIDNOR_GM.longitude_of_projection_origin
    lat_ts_MIDNOR = MIDNOR_GM.standard_parallel
    # Create Meshgrid
    x_MIDNOR, y_MIDNOR = np.meshgrid(MIDNOR_xc[:], MIDNOR_yc[:])
    # Define Projection Object for MIDS
    radius = 6370000.0
    proj4string_MIDNOR = f'stere +lat_0={lat_0_MIDNOR} +lon_0={lon_0_MIDNOR} +lat_ts={lat_ts_MIDNOR} +a={radius} +b={radius} +x_0={x0_MIDNOR} +y_0={y0_MIDNOR}'
    proj_MIDNOR = pyproj.Proj(proj=proj4string_MIDNOR)
    lon_MIDNOR, lat_MIDNOR = proj_MIDNOR(x_MIDNOR, y_MIDNOR, inverse=True)

    return proj_MIDNOR




if __name__ == "__main__":
    

    import numpy as np
    import os 
     # Example usage
    arr = [1, 2, 3, 4]
    print(num_unique_pairs(arr))  # Output: 6

    timestamp = 1633072800
    print(from_timestamp_to_date(timestamp))  # Output: '2021-10-01 09:20:00'


    # Example usage of get_midnor_projection
    midnor_proj = get_midnor_projection()

    # Test
    import matplotlib.pyplot as plt
    import numpy as np

    xc = np.linspace(0,100000,50)
    yc = np.linspace(0,100000,50)
    x, y = np.meshgrid(xc, yc)
    lon, lat = midnor_proj(x, y, inverse=True)
    xn, yn = midnor_proj(latitude=lat, longitude=lon)

    print("Error x", np.sum(np.abs(xn - x)))   
    print("Error y", np.sum(np.abs(yn - y)))


    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].scatter(x, y, c='blue', s=1)
    ax[0].set_title('MIDNOR Projection')
    ax[0].set_xlabel('X Coordinate')
    ax[0].set_ylabel('Y Coordinate')
    ax[1].scatter(lon, lat, c='red', s=1)
    ax[1].set_title('Geographic Coordinates')
    ax[1].set_xlabel('Longitude')
    ax[1].set_ylabel('Latitude')
    plt.tight_layout()
    plt.show()

