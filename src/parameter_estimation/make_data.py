import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
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
sinmod_path = "/Users/ajolaise/Library/CloudStorage/OneDrive-NTNU/PhD/code/CASCADE/data/sinmod/"
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

boundary = Boundary(
    border_file=boundary_kwargs["border_file"],
    file_type=boundary_kwargs["file_type"]
)



sinmod_c = Sinmod(sinmod_path + "/" + files[file_ind ], 
                    plot_path=plot_test_path,
                    log_kwargs=logging_kwargs,
                    boundary_kwargs=boundary_kwargs,
                    print_while_running=True,
                    filter_function=filter_function)

grid = Grid()
grid.make_regular_grid_inside_boundary(dxdy=160, boundary=boundary, origin=sinmod_c.get_origin())




def upper_bound(x, y, t):
    """
    Upper bound function for the sinmod data.
    """

    time_mean = (np.sin(2 * np.pi * t / 3600) + 1) / 2 * 5
    xy_mean = (np.sin(2 * np.pi * x / 10000) + 1) * (np.sin(2 * np.pi * y / 10000) + 1) * 3
    return time_mean + xy_mean + 10




def lower_bound(x, y, t):
    """
    Lower bound function for the sinmod data.
    """
    time_mean = (np.sin(2 * np.pi * t / 3600) + 1) / 2 * 5
    xy_mean = (np.sin(2 * np.pi * x / 10000) + 1) * (np.sin(2 * np.pi * y / 10000) + 1) * 3
    return time_mean + xy_mean + 30



n_masses = 10
x_min, x_max = np.min(grid.get_x_ax()) , np.max(grid.get_x_ax())
y_min, y_max = np.min(grid.get_y_ax()) , np.max(grid.get_y_ax())
x_locs_major = np.random.uniform(x_min, x_max, n_masses)
y_locs_major = np.random.uniform(y_min, y_max, n_masses)
n_masses_minor = 30
x_locs_minor = np.random.uniform(x_min, x_max, n_masses_minor)
y_locs_minor = np.random.uniform(y_min, y_max, n_masses_minor)



max_v = 100



def biomass_function(x, y, t):
    """
    Biomass function for the sinmod data.
    """

    value = 0

    xy_d = 2000

    for i, (x_loc, y_loc) in enumerate(zip(x_locs_major, y_locs_major)):
        value += np.exp(-((x - x_loc) ** 2 + (y - y_loc) ** 2) / (2 * xy_d** 2)) * 80
    
    for i, (x_loc, y_loc) in enumerate(zip(x_locs_minor, y_locs_minor)):
        value += np.exp(-((x - x_loc) ** 2 + (y - y_loc) ** 2) / (2 * xy_d ** 2)) * 15

    return value




    


tau = {
    "AUV_Thor": {
        "bounds" : 3,
        "biomass" : 5
    },
    "ASV_greta": {
        "bounds" : 2,
        "biomass" : 4
    },
    "Autonaut": {
        "bounds" : 3,
        "biomass" : 8
    },
}


xg = grid.grid[:, 0]
yg = grid.grid[:, 1]

lower_bounds = lower_bound(xg, yg, 0)
upper_bounds = upper_bound(xg, yg, 0)
biomass = biomass_function(xg, yg, 0)




plt.scatter(xg, yg, c=lower_bounds, s=1)
# Equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.title("Lower bounds")
plt.show()


plt.scatter(xg, yg, c=upper_bounds, s=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.title("Upper bounds")
plt.show()


plt.scatter(xg, yg, c=biomass, s=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.title("Biomass")
plt.show()



n_transects = 10
sample_spacing = 50
speed = 1
transect_length = 10000
vehicles = ["AUV_Thor", "ASV_greta", "Autonaut"]

for i in range(n_transects):
    vehicle = random.choice(vehicles)
    print("Vehicle: ", vehicle)

    x0, y0 = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
    x1, y1 = x0 + transect_length * np.cos(np.random.uniform(0, 2 * np.pi)), y0 + transect_length * np.sin(np.random.uniform(0, 2 * np.pi))
    
    loc_leagal = boundary.is_loc_legal(np.array([x1, y1]))
    while loc_leagal == False:
        x0, y0 = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
        x1, y1 = x0 + transect_length * np.cos(np.random.uniform(0, 2 * np.pi)), y0 + transect_length * np.sin(np.random.uniform(0, 2 * np.pi))
        loc_leagal = boundary.is_loc_legal(np.array([x1, y1]))
    transect_x = np.linspace(x0, x1, int(transect_length / sample_spacing)) 
    transect_y = np.linspace(y0, y1, int(transect_length / sample_spacing))
    t = np.linspace(0, transect_length / speed, len(transect_x))
    upper_bounds_transect = upper_bound(transect_x, transect_y, t)
    upper_bounds_transect_y = upper_bounds_transect + np.random.normal(0, tau[vehicle]["bounds"], len(transect_x))
    biomass_transect = biomass_function(transect_x, transect_y, t)
    biomass_transect_y = biomass_transect + np.random.normal(0, tau[vehicle]["biomass"], len(transect_x))
    lower_bounds_transect = lower_bound(transect_x, transect_y, t)
    lower_bounds_transect_y = lower_bounds_transect + np.random.normal(0, tau[vehicle]["bounds"], len(transect_x))

    plt.plot(t, upper_bounds_transect, label="Upper bounds")
    plt.plot(t, lower_bounds_transect, label="Lower bounds")
    plt.plot(t, upper_bounds_transect_y, label="Upper bounds with noise")
    plt.plot(t, lower_bounds_transect_y, label="Lower bounds with noise")
    plt.title("Transect bounds")
    plt.xlabel("Time")
    plt.ylabel("Bounds")
    plt.legend()
    plt.show()

    plt.plot(t, biomass_transect, label="Biomass")
    plt.plot(t, biomass_transect_y, label="Biomass with noise")
    plt.title("Transect biomass")
    plt.xlabel("Time")
    plt.ylabel("Biomass")
    plt.legend()
    plt.show()


    # Make csv 
    transect_df = pd.DataFrame({
        "x": transect_x,
        "y": transect_y,
        "t": t,
        "upper_bounds": upper_bounds_transect_y,
        "lower_bounds": lower_bounds_transect_y,
        "biomass": biomass_transect_y,
        "vehicle": [vehicle] * len(transect_x),
    })

    

    print(transect_df.head())


    wdir = get_project_root()
    transect_df.to_csv(os.path.join(wdir, "data/simulated_data" , f"transect_{i}.csv"), index=True)

    





