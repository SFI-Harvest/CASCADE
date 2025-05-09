import numpy as np
import random
from scipy.sparse import lil_matrix
from scipy.ndimage import gaussian_filter
import sys
import os



sys.path.insert(0, '/Users/ajolaise/Library/CloudStorage/OneDrive-NTNU/PhD/code/CASCADE/src')
# Load utilitys
from utilis.Logger import Logger
from utilis.Timing import Timing
from utilis.utility_funcs import *
from utilis.Grid import Grid
from utilis.WGS import WGS
from Covariance import Covariance



# Load other
from Sinmod import Sinmod
from Boundary import Boundary
from plotting.CascadePlotting import CascadePlotting


# Example usage
sinmod_path = "/Users/ajolaise/Library/CloudStorage/OneDrive-NTNU/PhD/code/CASCADE/data/sinmod/transfer_382450_files_da8d2b9a"
files = ["BioStates_froshelf.nc", "BioStates_midnor.nc", "mixOut.nc"]
plot_test_path = "/Users/ajolaise/Library/CloudStorage/OneDrive-NTNU/PhD/code/CASCADE/figures/tests/Sinmod/"


def filter_function(calanus_finmarchicus):
    """
    Filter function to filter the data.
    shape: (time, yc, xc) <----> (time, yc, xc)  Keeping the same shape
    """
    filtered_data = np.zeros(calanus_finmarchicus.shape)    
    for i in range(calanus_finmarchicus.shape[0]):
        filtered_data[i, :, :] = gaussian_filter(calanus_finmarchicus[i, :, :], sigma=1)
    return filtered_data



file_ind = 0
logging_kwargs = { 
    "log_file": os.path.join(plot_test_path, "Sinmod.log"),
    "print_to_console": True,
    "overwrite_file": True
}
boundary_kwargs = {
    "border_file": "/border_files/cascade_test_xy.csv",
    "file_type": "xy"}

sinmod_c = Sinmod(sinmod_path + "/" + files[file_ind ], 
                    plot_path=plot_test_path,
                    log_kwargs=logging_kwargs,
                    boundary_kwargs=boundary_kwargs,
                    print_while_running=True,
                    filter_function=filter_function)



grid = sinmod_c.sinmod_data["grid"]


def upper_bound(x, y, t):
    """
    Upper bound function for the sinmod data.
    """
    return 0.5 * (x + y) + 0.5 * (x - y) * np.sin(2 * np.pi * x / 10)


def lower_bound(x, y, t):
    """
    Lower bound function for the sinmod data.
    """
    return 0.5 * (x + y) - 0.5 * (x - y) * np.sin(2 * np.pi * x / 10)



def biomass_function(x, y, t):
    """
    Biomass function for the sinmod data.
    """
    return 0.5 * (x + y) + 0.5 * (x - y) * np.sin(2 * np.pi * x / 10)


tau = {
    "AUV_Thor": {
        "bounds" : 2,
        "biomass" : 4
    },
    "ASV_greta": {
        "bounds" : 2,
        "biomass" : 4
    },
    "Autonaut": {
        "bounds" : 1,
        "biomass" : 4
    },
}



