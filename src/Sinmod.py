import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interp
from scipy import interpolate
from shapely.geometry import Polygon, Point
import datetime
import time


from Boundary import Boundary

# Utility functions
from utilis.Logger import Logger
from utilis.Timing import Timing
from utilis.Grid import Grid
from utilis.utility_funcs import *


class Sinmod:
    """
    Sinmod class that reads and handles a sinmod simulation file.
    """

    def __init__(self, filename, 
                 plot_path=None,
                boundary_kwargs={},
                log_kwargs={},
                print_while_running=False,
                filter_function=None):
        """
        Initialize the Sinmod class with a filename.
        """
        self.logger = Logger("Sinmod", **log_kwargs) # This is the logger
        self.timing = Timing("Sinmod")  # This is the timing object
        self.logger.log_info("Setting up Sinmod class")

        # Set up the boundary
        self.logger.log_info("Setting up boundary")
        self.logger.log_info(f"Boundary kwargs: {boundary_kwargs}")
        self.boundary = Boundary(**boundary_kwargs)


        self.plot_path = plot_path                  # Path to save plots

        self.filter_function = filter_function # Function to filter the data

        # Read the file
        self.file_name = filename
       
        self.sinmod_data = {}
        self.sinmod_file = self.read_file()


    def __inds_inside_boundary(self):
        """
        Get the indices inside the boundary.
        """
        f_name = "__inds_inside_boundary"
        self.logger.log_info(f"[{f_name}] Getting indices inside boundary")
        to, time_stamp = self.timing.start_time(f_name, return_time_stamp=True)
        x = self.sinmod_file.variables["xc"][:]
        y = self.sinmod_file.variables["yc"][:]
        bounding_box = self.boundary.get_bounding_box()
        x_min, x_max = bounding_box[0], bounding_box[2]
        y_min, y_max = bounding_box[1], bounding_box[3]

        xxc, yyc = np.meshgrid(x, y)
        xxc = xxc.flatten()
        yyc = yyc.flatten()
        points = np.array([xxc, yyc]).T
        is_inside = np.repeat(False, len(points))
        for i in range(len(points)):
            if points[i][0] < x_min or points[i][0] > x_max:
                continue
            if points[i][1] < y_min or points[i][1] > y_max:
                continue
            if self.boundary.is_loc_legal(points[i]):
                is_inside[i] = True
        inside_inds = np.where(is_inside)[0]
        time_stamp_end = self.timing.end_time_id(to, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Indices inside boundary: {inside_inds.shape}")
        self.logger.log_info(f"[{f_name}] Indices inside boundary time: {time_stamp_end - time_stamp:.2f} seconds")
        return inside_inds
    

    def __make_calanus_interpolation_function(self):
        """
        Make the calanus interpolation function.
        """
        f_name = "__make_calanus_interpolation_function"
        self.logger.log_info("Making calanus interpolation function")
        t_id, t_start = self.timing.start_time("Make calanus interpolation function", return_time_stamp=True)

        xxc, yyc = self.sinmod_data["flatten"]["xc"], self.sinmod_data["flatten"]["yc"]
        valid_inds = self.sinmod_data["flatten"]["valid_points"]
        calanus_finmarchicus = self.sinmod_data["flatten"]["calanus_finmarchicus"]
        points = np.array([xxc[valid_inds], yyc[valid_inds]]).T
        interpolate_functions = []
        for i in range(calanus_finmarchicus.shape[0]):
            calanus_finmarchicus_i = calanus_finmarchicus[i, valid_inds].T
            interpolate_function = interpolate.CloughTocher2DInterpolator(points, calanus_finmarchicus_i, tol = 0.1, fill_value=0)
            interpolate_functions.append(interpolate_function)
        t_end = self.timing.end_time_id(t_id, return_time_stamp=True)
        self.logger.log_info(f"Calanus interpolation function time: {t_end - t_start:.2f} seconds")
        return interpolate_functions

    def get_calanus_interpolate_S_T(self, S, T):
        """
        Get the calanus interpolation function for S and T.
        """
        f_name = "get_calanus_interpolate_S_T"
        self.logger.log_info(f"[{f_name}] Getting calanus interpolation function for S and T")
        t_id, t_start = self.timing.start_time(f_name, return_time_stamp=True)
        interpolate_functions = self.interpolate_functions
        #self.logger.log_debug(f"[{f_name}] Interpolate functions: {interpolate_functions}")
        timestamps = self.sinmod_data["timestamp_seconds"]
        ind_below, ind_above = self.__get_t_inds_above_below(T)
        timestamps_below = np.array([timestamps[ib] for ib in ind_below])
        timestamps_above = np.array([timestamps[ia] for ia in ind_above])
        interpolate_values_below = np.zeros(len(T))
        interpolate_values_above = np.zeros(len(T))
        time_weights = np.zeros(len(T))

        for i in np.unique(ind_below):
            inds = np.where(ind_below == i)[0]
            interpolate_values_below[inds] = interpolate_functions[i](S[inds])
        for i in np.unique(ind_above):
            inds = np.where(ind_above == i)[0]
            interpolate_values_above[inds] = interpolate_functions[i](S[inds])
        divide_values = timestamps_above - timestamps_below
        time_weights = np.ones(len(T))
        time_weights[divide_values != 0] = (T[divide_values != 0] - timestamps_below[divide_values != 0]) / divide_values[divide_values != 0]
        calanus_finmarchicus = (1 - time_weights) * interpolate_values_below + time_weights * interpolate_values_above
        t_end = self.timing.end_time_id(t_id, return_time_stamp=True)
        self.logger.log_info(f"Calanus interpolation function for S and T time: {t_end - t_start:.2f} seconds")
        return calanus_finmarchicus

    def __get_t_inds_above_below(self, T):
        """
        Get the indices above and below the timestamp T
        """
        f_name = "__get_t_inds_above_below"
        self.logger.log_info(f"[{f_name}] Getting indices above and below timestamp")
        t_id, t_start = self.timing.start_time(f_name, return_time_stamp=True)
        simulation_time = self.sinmod_data["timestamp_seconds"]

        

        t_start_alt = time.time()
        # Make this a bit more efficient
        ind_above = np.zeros(len(T), dtype=int)
        ind_below = np.zeros(len(T), dtype=int)
        # Check if T is out of bounds
        inds_outside_below = np.where(T < np.min(simulation_time))[0]
        inds_outside_above = np.where(T > np.max(simulation_time))[0]
        inds_in_bounds = np.where((T >= np.min(simulation_time)) & (T <= np.max(simulation_time)))[0]
        ind_below[inds_outside_below] = 0
        ind_above[inds_outside_below] = 0
        ind_below[inds_outside_above] = len(simulation_time) - 1
        ind_above[inds_outside_above] = len(simulation_time) - 1
        dist =  np.tile(simulation_time, (len(T[inds_in_bounds]), 1)).T - T[inds_in_bounds]
        self.logger.log_debug(f"[{f_name}] dist.shape: {dist.shape}")
        dist[dist < 0] = np.inf
        colosest_ind = np.argmin(dist, axis=0)
        self.logger.log_debug(f"[{f_name}] colosest_ind.shape: {colosest_ind.shape}")
        ind_above[inds_in_bounds] = colosest_ind 
        ind_below[inds_in_bounds] = colosest_ind -1 

        """
        ind_below_alt, ind_above_alt = ind_below.copy(), ind_above.copy()
        t_end_alt = time.time()


        # For where T is in bounds

        ind_above = []
        ind_below = []

        any_out_of_bounds = False
        t_start_old = time.time()
        for t in T:
            if t < simulation_time[0]:
                ind_above.append(0)
                ind_below.append(0)
                any_out_of_bounds = True
            elif t > simulation_time[-1]:
                ind_above.append(len(simulation_time) - 1)
                ind_below.append(len(simulation_time) - 1)
                any_out_of_bounds = True
            else:
                ind_above.append(np.where(simulation_time >= t)[0][0])
                ind_below.append(ind_above[-1] - 1)
        ind_above = np.array(ind_above, dtype=int)
        ind_below = np.array(ind_below, dtype=int)
        t_end_old = time.time()

        self.logger.log_debug(f"[{f_name}] TIme taken for new method: {t_end_alt - t_start_alt:.4f} seconds")
        self.logger.log_debug(f"[{f_name}] Time taken for old method: {t_end_old - t_start_old:.4f} seconds")

        # DEBUGGING
        # Check if the indices are the same 
        if np.sum(np.abs(ind_above - ind_above_alt)) > 0 or np.sum(np.abs(ind_below - ind_below_alt)) > 0:
            self.logger.log_error(f"[{f_name}] Indices above are not the same")
            self.logger.log_error(f"[{f_name}] ind_above: {ind_above}")
            self.logger.log_error(f"[{f_name}] ind_above_alt: {ind_above_alt}")
            self.logger.log_error(f"[{f_name}] Indices below are not the same")
            self.logger.log_error(f"[{f_name}] ind_below: {ind_below}")
            self.logger.log_error(f"[{f_name}] ind_below_alt: {ind_below_alt}")
        else:
            self.logger.log_debug(f"[{f_name}] Indices above and below are the same")
        """

        if len(inds_outside_below) > 0 or len(inds_outside_above) > 0:
            self.logger.log_warning(f"[{f_name}] Some timestamps are out of bounds.")
            self.logger.log_warning(f"[{f_name}] n Timestamps below: {len(T[inds_outside_below])}")
            self.logger.log_warning(f"[{f_name}] n Timestamps above: {len(T[inds_outside_above])}")

        t_end = self.timing.end_time_id(t_id, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Indices above and below timestamp time: {t_end - t_start:.4f} seconds")
        return ind_below, ind_above


    def read_file(self):
        """
        Read the sinmod file and parse its contents.
        """
        filename = self.file_name
        if not os.path.exists(filename):
            self.logger.log_error(f"File {filename} does not exist.")
            return
        t_id, t_start = self.timing.start_time("read_file", return_time_stamp=True)
        self.logger.log_info(f"Reading file {filename}")
        sinmod_file = netCDF4.Dataset(filename)
        print(sinmod_file)
        self.sinmod_file = sinmod_file
        t_inter = self.timing.intermittent_time_id(t_id, return_time_stamp=True)
        

        inds_inside_boundary = self.__inds_inside_boundary()

        x_max = np.max(sinmod_file.variables["xc"])
        x_min = np.min(sinmod_file.variables["xc"])
        y_max = np.max(sinmod_file.variables["yc"])
        y_min = np.min(sinmod_file.variables["yc"])
        xxc, yyc = np.meshgrid(sinmod_file.variables["xc"][:], sinmod_file.variables["yc"][:])
        xxc = xxc.flatten()
        yyc = yyc.flatten()
        self.sinmod_data["xxc"] = xxc
        self.sinmod_data["yyc"] = yyc
        self.sinmod_data["xc"] = sinmod_file.variables["xc"][:]
        self.sinmod_data["yc"] = sinmod_file.variables["yc"][:]
        self.sinmod_data["zc"] = sinmod_file.variables["zc"][:]
        self.sinmod_data["dx"] = self.sinmod_data["xc"][1] - self.sinmod_data["xc"][0]
        self.sinmod_data["dy"] = self.sinmod_data["yc"][1] - self.sinmod_data["yc"][0]
        self.sinmod_data["dz"] = self.sinmod_data["zc"][1] - self.sinmod_data["zc"][0]
        self.sinmod_data["x_min"] = x_min
        self.sinmod_data["x_max"] = x_max
        self.sinmod_data["y_min"] = y_min
        self.sinmod_data["y_max"] = y_max

        # Getting the timestamps in seconds
        time = sinmod_file.variables["time"][:]
        years, month, days = time[:,0], time[:,1], time[:,2]
        hours, minutes, seconds = time[:,3], time[:,4], time[:,5]
        datetime_list = [datetime.datetime(int(years[i]), int(month[i]), int(days[i]), int(hours[i]), int(minutes[i]), int(seconds[i])) for i in range(len(time))]
        self.sinmod_data["datetime"] = datetime_list
        self.sinmod_data["timestamp_seconds"] = np.array([datetime_list[i].timestamp() for i in range(len(datetime_list))])
        self.sinmod_data["dt"] = self.sinmod_data["timestamp_seconds"][1] - self.sinmod_data["timestamp_seconds"][0]
        # Here we store some flattened data
        self.sinmod_data["flatten"] = {}
        self.sinmod_data["flatten"]["depth"] = sinmod_file.variables["depth"][:].flatten()
        self.sinmod_data["flatten"]["xc"] = xxc
        self.sinmod_data["flatten"]["yc"] = yyc
        self.sinmod_data["flatten"]["inds_inside_boundary"] = inds_inside_boundary
        self.sinmod_data["flatten"]["ocean_inds"] = np.where(self.sinmod_data["flatten"]["depth"] >= 0)[0]
        self.sinmod_data["flatten"]["land_inds"] = np.where(self.sinmod_data["flatten"]["depth"] < 0)[0]
        self.sinmod_data["flatten"]["shore_inds"] = np.where((self.sinmod_data["flatten"]["depth"] >= 0) & (self.sinmod_data["flatten"]["depth"] < 20))[0]
        inds_inside_bound = np.setdiff1d(inds_inside_boundary, self.sinmod_data["flatten"]["land_inds"])
        inds_inside_bound = np.setdiff1d(inds_inside_bound, self.sinmod_data["flatten"]["shore_inds"])
        self.sinmod_data["flatten"]["valid_points"] = inds_inside_bound
        #self.sinmod_data["flatten"]["mask_inds"] = 
        calanus_finmarchicus = sinmod_file.variables["calanus_finmarchicus"]

        self.sinmod_data["flatten"]["calanus_finmarchicus"] = np.array([calanus_finmarchicus[i, :, :].flatten() for i in range(calanus_finmarchicus.shape[0])])


        self.sinmod_data["land_ind"] = np.where(sinmod_file.variables["depth"][:] < 0)
        self.sinmod_data["ocean_ind"] = np.where(sinmod_file.variables["depth"][:] >= 0)
        self.logger.log_info(f"Land_ind.shape: {self.sinmod_data['land_ind'][0].shape}")
        self.logger.log_info(f"Ocean_ind.shape: {self.sinmod_data['ocean_ind'][0].shape}")
        self.sinmod_data["inds_inside_boundary"] = self.__get_inds_inside_boundary()
        self.logger.log_info(f"inds_inside_boundary.shape: {self.sinmod_data['inds_inside_boundary'].shape}")

        self.interpolate_functions = self.__make_calanus_interpolation_function()

        t_end = self.timing.end_time_id(t_id,return_time_stamp=True)
        self.logger.log_info(f"File {filename} read in {t_end - t_start:.2f} seconds")
        return sinmod_file

    def __make_interpolated_grid(self):
        pass

    def __get_inds_inside_boundary(self):
        """
        Get the indices inside the boundary.
        """
        x = self.sinmod_file.variables["xc"][:]
        y = self.sinmod_file.variables["yc"][:]
        xxc, yyc = np.meshgrid(x, y)
        xxc = xxc.flatten()
        yyc = yyc.flatten()
        points = np.array([xxc, yyc]).T
        is_inside = np.repeat(False, len(points))
        for i in range(len(points)):
            if self.boundary.is_loc_legal(points[i]):
                is_inside[i] = True
        inside_inds = np.where(is_inside)[0]
        return inside_inds


    def print_sinmod_data_shape(self):
        """
        Print the shape of the sinmod data.
        """
        self.logger.log_info("Sinmod data shape:")
        self.print_dict_shape(self.sinmod_data)
    

    def print_dict_shape(self, prefix=""):
        """
        Print the shape of the sinmod data.
        """
        for key in self.sinmod_data.keys():
            if isinstance(self.sinmod_data[key], dict):
                print(f"{prefix}{key}:")
                self.print_dict_shape(self.sinmod_data[key], prefix=prefix + "  ")
            else:
                cont = self.sinmod_data[key]
                if isinstance(cont, np.ndarray):
                    print(f"{prefix}{key}: {cont.shape}")
                else:
                    print(f"{prefix}{key}: {type(cont)}")

    

    def get_prior_S_T(self, S, T):
        """
        If the sinmod is a prior this is the function to get the prior S and T.

        For now this is just a wrapper for the get_calanus_interpolate_S_T function.
        But in the future this can be a bit fancier.
        """
        f_name = "get_prior_S_T"
        self.logger.log_debug(f"[{f_name}] {S.shape}")
        # Check if T does not have a length
        # Then it is a single time step, and will caouse some issues later  
        if not hasattr(T, "__len__"):
            # A bit of a hack to make sure that the interpolation works
            # for a single time step, probalby not the best way to do it, but not worth the time to fix it
            S = np.concatenate((S, S), axis=0)
            T = np.array([T, T])
            values = self.get_calanus_interpolate_S_T(S, T)
            return np.array([values[0]])
        interpolated_values = self.get_calanus_interpolate_S_T(S, T)
        interpolated_values = np.clip(interpolated_values, 0, None)
        return interpolated_values
    

    def make_grid_format(self, S, T, y):
        """
        Make the grid format for the data.
        y_new.shape = (n_time_steps, sx, sy)

        """
        self.logger.log_info("Making grid format for the data")
        t_id, t_start = self.timing.start_time("Make grid format", return_time_stamp=True)

        



    def filter_smooth(self, S, T):
        """
        Filter and smooth the data.
        """
        self.logger.log_info("Filtering and smoothing data")
        t_id, t_start = self.timing.start_time("Filter and smooth", return_time_stamp=True)

        # Use techniques from CNNs 


       


    

    ##########
    # Getters and Setters

    def get_origin(self, in_boundary=True):
        if in_boundary:
            xxc = self.sinmod_data["xxc"][self.sinmod_data["flatten"]["inds_inside_boundary"]]
            yyc = self.sinmod_data["yyc"][self.sinmod_data["flatten"]["inds_inside_boundary"]]
            return np.array([np.min(xxc), np.min(yyc)])
        else:
            xc = self.sinmod_data["xc"]
            yc = self.sinmod_data["yc"]
            return np.array([np.min(xc), np.min(yc)])

       



if __name__ == "__main__":

    from plotting.SinmodPlotting import SinmodPlotting
    from scipy.ndimage import gaussian_filter

    # Example usage
    wdir = get_project_root()
    sinmod_path = "data/sinmod/"
    sinmod_path = os.path.join(wdir, sinmod_path)
    files = ["BioStates_froshelf.nc", "BioStates_midnor.nc", "mixOut.nc", "physStates.nc"]

    plot_test_path = os.path.join(wdir, "figures/tests/Sinmod/")


    def filter_function(calanus_finmarchicus):
        """
        Filter function to filter the data.
        shape: (time, yc, xc) <----> (time, yc, xc)  Keeping the same shape
        """
        filtered_data = np.zeros(calanus_finmarchicus.shape)    
        for i in range(calanus_finmarchicus.shape[0]):
            filtered_data[i, :, :] = gaussian_filter(calanus_finmarchicus[i, :, :], sigma=1)
        return filtered_data



    file_ind = 3
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
    
    timing = Timing("Sinmod")

    SinmodPlotting.test_interpolation(sinmod_c)

    SinmodPlotting.plot_different_inds(sinmod_c)

    sinmod_c.timing.plot_timing_data(plot_name="timing.png")
    
    #sinmod_c.plot_depth(plot_name ="depth_b.png")
    #sinmod_c.plot_inside_boundary(plot_name="inside_boundary_b.png")

    # Read the nc file



    sinmod = netCDF4.Dataset(sinmod_path + "/" + files[file_ind ])
    print(sinmod)


    calanus_finmarchicus = sinmod.variables["calanus_finmarchicus"][:]
    print(calanus_finmarchicus.shape)

    plt.imshow(calanus_finmarchicus[1, :, :], cmap="jet")
    plt.colorbar()
    plt.title("Calanus finmarchicus")
    plt.savefig(os.path.join(plot_test_path, "calanus_finmarchicus.png"))
    plt.close()
    

    depth = sinmod.variables["depth"][:]
    max_depth = np.max(depth)
    print(depth.shape)
    plt.imshow(depth, cmap="jet", vmin=0, vmax=max_depth)
    plt.colorbar()
    plt.title("Depth")
    plt.savefig(os.path.join(plot_test_path, "depth.png"))
    plt.close()

    depth = sinmod.variables["depth"][:]
    depth_arr = np.array(depth)
    land_ind = np.where(depth_arr <  0)
    ocean_ind = np.where(depth_arr >= 0)
    plt.imshow(depth_arr, cmap="jet", vmin=0, vmax=max_depth, label="Depth")
    plt.colorbar()
    plt.scatter(land_ind[1], land_ind[0], color="black", s=1)
    # Add the bounding box
    bounds = sinmod_c.boundary.get_bounding_points()
    plt.plot(bounds[:, 0], bounds[:, 1], color="red", linewidth=2)
    plt.title("Depth")
    plt.savefig(os.path.join(plot_test_path, "depth_land_mask_bound.png"))
    plt.close()



    depth = sinmod.variables["depth"][:]
    max_depth = np.max(depth)
    print(depth.shape)
    x_min, x_max = np.min(sinmod.variables["xc"]), np.max(sinmod.variables["xc"])
    y_min, y_max = np.min(sinmod.variables["yc"]), np.max(sinmod.variables["yc"])
    plt.imshow(depth, cmap="jet", vmin=0, vmax=max_depth, extent=(x_min, x_max, y_min, y_max))
    # Change the x and y axis
    plt.colorbar()
    plt.title("Depth")
    plt.savefig(os.path.join(plot_test_path, "depth.png"))
    plt.close()



    # Get dist to land 
    land_ind = np.where(depth_arr <  0)
    ocean_ind = np.where(depth_arr >= 0)
    xc = sinmod.variables["xc"][:]
    yc = sinmod.variables["yc"][:]
    xxc, yyc = np.meshgrid(xc, yc)
    #xxc, yyc = xxc.flatten(), yyc.flatten()
    plt.plot(xxc[land_ind], yyc[land_ind], "o", color="black", markersize=1)
    bounds = sinmod_c.boundary.get_bounding_points()
    plt.plot(bounds[:, 0], bounds[:, 1], color="red", linewidth=2)
    plt.title("Land mask")
    plt.savefig(os.path.join(plot_test_path, "land_mask.png"))
    plt.close()

    shore_line_ind = np.where((0<depth_arr) & (depth_arr< 20))
    plt.plot(xxc[shore_line_ind], yyc[shore_line_ind], "o", color="black", markersize=1)
    plt.title("Shore line")
    plt.savefig(os.path.join(plot_test_path, "shore_line.png"))
    plt.close()


    # Distance to land
    dist_matrix = np.zeros((len(yc), len(xc)))
    yyc, xxc = np.meshgrid(yc, xc)

    for i in range(len(yc)):
        print("  ", i, "of", len(xc), end="\r")
        for j in range(len(xc)):
            if i in land_ind[0] and j in land_ind[1]:   
                dist_matrix[i, j] = 0
            else:
                dist_to_land = np.sqrt((xxc[i,j] - xxc[shore_line_ind])**2 + (yyc[i,j] - yyc[shore_line_ind])**2)
                min_dist = np.min(dist_to_land)
                dist_matrix[i, j] = min_dist    
    plt.imshow(dist_matrix, cmap="jet")
    plt.colorbar()
    plt.title("Distance to land")
    plt.savefig(os.path.join(plot_test_path, "dist_to_land.png"))
    plt.close()
    # Get the land mask


