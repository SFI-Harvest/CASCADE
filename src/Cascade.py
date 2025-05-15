

# -*- coding: utf-8 -*-

import numpy as np
import time
import os
import pandas as pd
import pickle
import netCDF4
from netCDF4 import Dataset
import datetime


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


class Cascade:
    """
    Cascade: Central calanus model

    """

    def __init__(self,
                 prior, 
                 model_parameters={
                     "dxdy": 1000,
                     "dt": 60*60,
                     "tau": 0.1,
                     "prior_shift": 1e-6,
                     "y_shift": 1e-6,
                 },
                 covariance_kwargs={
                     "phi_xy": 1000,
                     "phi_temporal": 60*60,
                     "sigma": 1,
                     "sigma_y": 1,
                     "sigma_bounds": 1,
                     "temporal_covariance_type": "exponential",
                     "xy_covariance_type": "matern_3_2",},  
                 boundary_kwargs={},
                 timing_kwargs={},
                 log_kwargs={}):
        

        """
        Parameters
        ----------
        prior : Sinmod
            Prior model to use for the Cascade model here it is based on the Sinmod model.
        dxdy : int, optional
            Grid spacing in meters. The default is 1000.
        dt : int, optional
            Temporal spacing in seconds. The default is 60*60, which is 1 hour.
        boundary_kwargs : dict, optional
            Boundary kwargs. The default is {}. This means that there is no boundary.
        
        timing_kwargs : dict, optional
            "do_timing": bool, optional
                If True, then the timing will be done. The default is True. if False then it just returns without storing anything.
                the function will still return the timestamp if asked for it.

        log_kwargs : dict, optional
            "log_file": str, optional
                The path to the log file. If none, then the log will not be saved.
            "print_to_console": bool, optional
                If True, then the log will be printed to the console. The default is True. Error messages will always be printed to the console.
            "overwrite_file": bool, optional
                If True, then the old log file will be overwritten. If false the old log file will be kept with a new name by the date
            

        """
        f_name = "__init__"

        

        ### Setting up the timekeeping and logging
        self.logger = Logger("Cascade", **log_kwargs)
        self.logger.log_info(f"[{f_name}] Setting up Cascade model")
        self.timing = Timing("Cascade")

        t1 = self.timing.start_time("__init__")

        # Setting up the boundary
        self.logger.log_info(f"[{f_name}] Setting up boundary")
        self.logger.log_info(f"[{f_name}]  Boundary kwargs: {boundary_kwargs}")
        self.boundary = Boundary(**boundary_kwargs)


        # Setting up the covariance
        self.logger.log_info(f"[{f_name}] Setting up covariance")
        self.logger.log_info(f"[{f_name}] Covariance kwargs: {covariance_kwargs}")
        self.covariance = Covariance(covariance_kwargs, logger_kwargs=log_kwargs)
        self.covariance_y = Covariance(covariance_kwargs, logger_kwargs=log_kwargs)
        self.covariance_y.set_sigma(covariance_kwargs["sigma_y"])
        self.covariance_bounds = Covariance(covariance_kwargs, logger_kwargs=log_kwargs)
        self.covariance_bounds.set_sigma(covariance_kwargs["sigma_bounds"])


        ## Transformation functions
        self.logger.log_info(f"[{f_name}] Setting up transformation functions")
        self.data_transformation_functions = {}
        self.data_transformation_functions["sinmod"] = {
            "transformation": self.log_transformation,
            "inverse_transformation": self.log_transformation_inv
        }
        self.data_transformation_functions["AUV"] = {  
            "transformation": self.log_transformation,
            "inverse_transformation": self.log_transformation_inv
        }
        self.data_transformation_functions["ASV"] = {
            "transformation": self.log_transformation,
            "inverse_transformation": self.log_transformation_inv
        }

      
        self.prior = prior

        self.logger.log_info(f"[{f_name}] Setting up grid")
        self.grid = Grid()
        dxdy = model_parameters["dxdy"]
        dt = model_parameters["dt"]
        self.grid.make_regular_grid_inside_boundary(dxdy=dxdy, boundary=self.boundary, origin=self.prior.get_origin())
        self.logger.log_info(f"[{f_name}] Grid shape: {self.grid.grid.shape}")
        self.grid.make_time_grid(t0=0, dt=dt)   ### TODO: make this more flexible


        # Model
        self.logger.log_info(f"[{f_name}] Setting up model parameters")
        self.logger.log_info(f"[{f_name}] Model parameters: {model_parameters}")
        self.model_parameters = model_parameters


        # Setting up the data storage
        self.data_dict = {}
        self.data_dict["has_data"] = False
        self.data_dict["gridded_data"] = {}
        self.data_dict["all_data"] = {}

        # List of files included 
        self.data_dict["all_data"]["data_source"] = []

        # Storing the parameters
        self.all_parameters = {
            "model_parameters": model_parameters,
            "covariance_kwargs": covariance_kwargs,
            "boundary_kwargs": boundary_kwargs,
            "timing_kwargs": timing_kwargs,
            "log_kwargs": log_kwargs
        }

        ## Setting up the projection 
        ## TODO: make this more flexible
        self.projection = get_midnor_projection()
        

        self.timing.end_time_id(t1)
        self.logger.log_info(f"[{f_name}] Cascade model initialized")


    


    ####################################################
    ######  Data transformation functions  #############
    ####################################################
    def log_transformation(self, y):
        """
        Apply some transformation to the data 
        y* = f(y)
        """
        shift = self.model_parameters.get("y_shift", 1e-6)
        y_shift = y + shift
        log_y = np.log(y_shift)
        return log_y
    
    def log_transformation_inv(self, y_transformed):
        """
        Transform the data back to the original scale
        y = f_inv(y*)
        """
        shift = self.model_parameters.get("y_shift", 1e-6)
        y = np.exp(y_transformed) - shift
        return y

    def data_transformation_y(self, data_dict):
        """
        Apply some transformation to the data 
        y* = f(y)
        """

        # TODO: Implement the transformation function
        # One possible transformation is to take the log of the data
        # check if any data points are negative

        if np.any(data_dict["y"] < 0):
            self.logger.log_error(f"[data_transformation_y] Data points are negative. Cannot take log of negative values.")
            raise ValueError("Data points are negative. Cannot take log of negative values.")

        y_shift = data_dict["y"] + 1e-6
        log_y = np.log(y_shift)
        return log_y

    def data_transformation_y_inv(self, y_transformed):
        """
        Transform the data back to the original scale
        y = f_inv(y*)
        """
        y = np.exp(y_transformed) - 1e-6
        return y


    def data_transformation_sinmod(self, S, T, uncorrected_sinmod):
        """
        Gets the values from the prior model
        """
        f_name = "data_transformation_sinmod"
        # TODO: Implement the correction function
        # This is just a placeholder for now
        #self.logger.log_debug(f"[{f_name}] uncorrected_prior: {uncorrected_prior}")
        #uncorrected_prior
        if np.any(uncorrected_sinmod < 0):
            raise ValueError("Data points are negative. Cannot take log of negative values.")
        shifted_sinmod = uncorrected_sinmod + 1e-6
        log_sinmod = np.log(shifted_sinmod)
        return log_sinmod
    

    def data_transformation_sinmod_inv(self, S, T, corrected_sinmod):
        """
        Transform the data back to the original scale
        y = f_inv(y*)
        """
        f_name = "data_transformation_sinmod_inv"
        min_val = np.exp(-6)
        return np.exp(corrected_sinmod) - min_val


    def check_data(self, data_dict):
        f_name = "check_data"
        """
        Check if the data for the format and other checks
        """

        # Check if the biomass is positive
        if np.any(data_dict["y"] < 0):
            self.logger.log_error(f"[{f_name}] Data points are negative. Cannot take log of negative values.")
            n_negative = np.sum(data_dict["y"] < 0)
            n_total = len(data_dict["y"])
            min_val = np.min(data_dict["y"])
            self.logger.log_error(f"[{f_name}] Number of negative data points: {n_negative} out of {n_total}")
            self.logger.log_error(f"[{f_name}] Minimum value: {min_val}")
            option = input("Do you want to continue? \n if yes then the data will be clipped (y/n): ")

            if option == "y":
                self.logger.log_warning(f"[{f_name}] Data points are negative. Clipping the data to 0.")
                data_dict["y"] = np.clip(data_dict["y"], 0, None)
            else:
                raise ValueError("Data points are negative. Cannot take log of negative values.")
        
        # Check if the upper and lower bounds are positive
        if np.any(data_dict["upper_bound"] < 0) or np.any(data_dict["lower_bound"] < 0):
            self.logger.log_error(f"[{f_name}] Upper and lower bounds are negative. Depths cannot be negative.")
            n_negative = np.sum(data_dict["upper_bound"] < 0) + np.sum(data_dict["lower_bound"] < 0)
            n_total = len(data_dict["upper_bound"]) + len(data_dict["lower_bound"])
            min_val = np.min(np.concatenate((data_dict["upper_bound"], data_dict["lower_bound"])))
            self.logger.log_error(f"[{f_name}] Number of negative data points: {n_negative} out of {n_total}")
            self.logger.log_error(f"[{f_name}] Minimum value: {min_val}")
            option = input("Do you want to continue? \n if yes then the data will be clipped (y/n): ")

            if option == "y":
                self.logger.log_warning(f"[{f_name}] Data points are negative. Clipping the data to 0.")
                data_dict["upper_bound"] = np.clip(data_dict["upper_bound"], 0, None)
                data_dict["lower_bound"] = np.clip(data_dict["lower_bound"], 0, None)
            else:
                raise ValueError("Data points are negative. Cannot take log of negative values.")
            

        # Check if the upper and lower bounds are in the right order
        # The upper bound should have a lower depth than the lower bound, the depth values are positive, so high value is deeper
        if np.any(data_dict["upper_bound"] > data_dict["lower_bound"]):
            self.logger.log_error(f"[{f_name}] Upper bound is deeper than lower bound. The upper bound should be shallower than the lower bound.")
            n_deeper = np.sum(data_dict["upper_bound"] > data_dict["lower_bound"])
            n_total = len(data_dict["upper_bound"])
            max_neg_diff = np.max(data_dict["upper_bound"] - data_dict["lower_bound"])
            self.logger.log_error(f"[{f_name}] Number of upper bounds deeper than lower bounds: {n_deeper} out of {n_total}")
            self.logger.log_error(f"[{f_name}] Maximum diff: {max_neg_diff}")
            option = input("Do you want to continue? \n if yes then the data will be swapped where relevant (y/n): ")

            if option == "y":
                self.logger.log_warning(f"[{f_name}] Data points are negative. Clipping the data to 0.")
                # Swap the values
                data_dict["upper_bound"], data_dict["lower_bound"] = np.minimum(data_dict["upper_bound"], data_dict["lower_bound"]), np.maximum(data_dict["upper_bound"], data_dict["lower_bound"])
            else:
                raise ValueError("Data points are negative. Cannot take log of negative values.")






    def add_data_to_model(self, data_dict):
        """
        Add data to the model
        """
        f_name = "add_data_to_model"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)

        # Check if the data is in the right format
        self.check_data(data_dict)

        n = len(data_dict["Sx"])
        self.logger.log_info(f"[{f_name}] Adding data to the model")
        self.logger.log_info(f"[{f_name}] Number of data points: {n}")

        Sx = data_dict["Sx"]
        Sy = data_dict["Sy"]
        S = np.array([Sx, Sy]).T
        T = data_dict["T"]
        y = data_dict["y"]
        y_transformed = self.data_transformation_y(data_dict)

        data_dict["S"] = S
        data_dict["y_corrected"] = y_transformed


        # Get the prior values
        mu_y_cor, mu_y, mu_ub, mu_lb = self.get_prior(S, T)

        # Store the prior values in the data dictionary
        data_dict["prior_y"] = mu_y
        data_dict["prior_y_corrected"] = mu_y_cor
        data_dict["prior_lower_bound"] = mu_lb
        data_dict["prior_upper_bound"] = mu_ub


        assignments = self.grid.get_assignment(S, T)
        data_dict["S_grid_inds"] = assignments["grid_inds"]
        data_dict["S_grid"] = assignments["grid_points"]
        data_dict["dist_S_to_S_grid"] = assignments["dist_to_closest"]
        data_dict["T_grid_inds"] = assignments["time_inds"]
        data_dict["T_grid"] = assignments["time_points"]
        data_dict["dist_T_to_T_grid"] = assignments["dist_to_closest_time"]
        data_dict["S_T_grid_inds"] = assignments["grid_time_inds"]

        ## Now make the data gridded 
        data_dict_gridded = self.__grid_data(data_dict)

        # Update the data dictionary
        self.__update_data_dict(data_dict, data_dict_gridded)


        # Fit the model with the new data
        self.fit()

        self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Data added to the model")

    def __grid_data(self, data_dict):
        ## Now make the data gridded 
        ## This is also done to reduce the data size
        f_name = "__grid_data"
        n = len(data_dict["Sx"])
        self.logger.log_info(f"[{f_name}] Gridding data with size {n}")
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)


        data_dict_gridded = {}
        unique_S_T_grid_inds = np.unique(data_dict["S_T_grid_inds"])
        n_S_T_grid_inds = len(unique_S_T_grid_inds)

        data_dict_gridded["S_T_grid_inds"] = unique_S_T_grid_inds

        # These are the interesting values
        data_dict_gridded["T_grid"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["S_grid"] = np.zeros((n_S_T_grid_inds, 2))
        data_dict_gridded["S_grid_inds"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["T_grid_inds"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["y"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["y_corrected"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["upper_bound"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["lower_bound"] = np.zeros(n_S_T_grid_inds)

        data_dict_gridded["prior_y"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["prior_y_corrected"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["prior_lower_bound"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["prior_upper_bound"] = np.zeros(n_S_T_grid_inds)
        data_dict_gridded["noise"] = np.zeros(n_S_T_grid_inds)


        for i, unique_S_T_grid_ind in enumerate(unique_S_T_grid_inds):
            inds = np.where(data_dict["S_T_grid_inds"] == unique_S_T_grid_ind)[0]
            ind_0 = inds[0] # For many points we just take the first one, because they are all the same
                            # That should be the case for the gridded data 
            data_dict_gridded["T_grid"][i] = data_dict["T_grid"][ind_0]
            data_dict_gridded["S_grid"][i] = data_dict["S_grid"][ind_0]
            S_grid, T_grid = data_dict["S_grid"][ind_0], data_dict["T_grid"][ind_0]
            data_dict_gridded["S_grid_inds"][i] = data_dict["S_grid_inds"][ind_0]
            data_dict_gridded["T_grid_inds"][i] = data_dict["T_grid_inds"][ind_0]
            T_dists = data_dict["dist_T_to_T_grid"][inds]
            S_dists = data_dict["dist_S_to_S_grid"][inds]
            y_corrected = data_dict["y_corrected"][inds]
            y = data_dict["y"][inds]
            upper_bound = data_dict["upper_bound"][inds]
            lower_bound = data_dict["lower_bound"][inds]
            
            # Averaging over the measurements
            # TODO: Make this a bit better
            # Perhaps consider single value preditions?
            # This should also include some kind of noise es timation
            data_dict_gridded["y_corrected"][i] = np.mean(y_corrected)
            data_dict_gridded["y"][i] = np.mean(y)
            data_dict_gridded["upper_bound"][i] = np.mean(upper_bound)
            data_dict_gridded["lower_bound"][i] = np.mean(lower_bound)
            data_dict_gridded["noise"][i] = self.get_tau()**2


        S_grid, T_grid = data_dict_gridded["S_grid"], data_dict_gridded["T_grid"]
        # Getting the priors in the grid points
        mu_y_cor, mu_y, mu_ub, mu_lb = self.get_prior(S_grid, T_grid)
        data_dict_gridded["prior_y"] = mu_y
        data_dict_gridded["prior_y_corrected"] = mu_y_cor
        data_dict_gridded["prior_lower_bound"] = mu_lb
        data_dict_gridded["prior_upper_bound"] = mu_ub


            

        #data_dict_gridded["covaraince"] = self.covariance.make_covariance_matrix(data_dict_gridded["S_grid"], data_dict_gridded["T_grid"])
        self.logger.log_info(f"[{f_name}] Data size after gridding: {n_S_T_grid_inds}")

        t_end = self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time taken for gridding: {t_end - t1:.2f} seconds")
        return data_dict_gridded

    

    def __update_data_dict(self, data_dict, data_dict_gridded):

        """
        This function is responsible for updating the data dictionaries
        data_dict: dict
        """

        keys = list(data_dict.keys())
        gridded_keys = list(data_dict_gridded.keys())
        
        # These are keys not in list like 
        special_keys = ['data_source']
        if self.data_dict["has_data"] == False:
            for key in keys:
                if key in special_keys:
                    # If the key is in the special keys, then we need to do something special with it
                    if key == "data_source":
                        self.data_dict["all_data"][key] = [data_dict[key] for _ in range(len(data_dict["S"]))]
                else:
                    self.data_dict["all_data"][key] = data_dict[key]
            
            self.data_dict["all_data"]["batch"] = np.repeat(int(0), len(data_dict["S"]))
            for key in gridded_keys:
                if key in special_keys:
                    # If the key is in the special keys, then we need to do something special with it
                    if key == "data_source":
                        self.data_dict["gridded_data"][key] = [data_dict_gridded[key] for _ in range(len(data_dict_gridded["S_grid"]))]
                else:
                    self.data_dict["gridded_data"][key] = data_dict_gridded[key]
            self.data_dict["gridded_data"]["batch"] = np.repeat(int(0), len(data_dict_gridded["S_grid"]))
            self.data_dict["has_data"] = True
        else:
            for key in keys:
                if key in special_keys:
                    # If the key is in the special keys, then we need to do something special with it
                    if key == "data_source":
                        for _ in range(len(data_dict["S"])):
                            self.data_dict["all_data"][key].append(data_dict[key])
                else:
                    try:                 
                        self.data_dict["all_data"][key] = np.concatenate((self.data_dict["all_data"][key], data_dict[key]), axis=0)
                    except:
                        print(f"key: {key}")
                        print(f"data_dict[key].shape: {data_dict[key].shape}")
                        print(f"self.data_dict['all_data'][key].shape: {self.data_dict['all_data'][key].shape}")
                        self.data_dict["all_data"][key] = np.concatenate((self.data_dict["all_data"][key], data_dict[key]), axis=0)

              
            for key in gridded_keys:
                if key in special_keys:
                    # If the key is in the special keys, then we need to do something special with it
                    if key == "data_source":
                        for _ in range(len(data_dict_gridded["S_grid"])):
                            self.data_dict["gridded_data"][key].append(data_dict_gridded[key])
                else:
                    #print(f"key: {key}")
                    #print(f"data_dict_gridded[key].shape: {data_dict_gridded[key].shape}")
                    #print(f"self.data_dict['gridded_data'][key].shape: {self.data_dict['gridded_data'][key].shape}")
                    self.data_dict["gridded_data"][key] = np.concatenate((self.data_dict["gridded_data"][key], data_dict_gridded[key]), axis=0)
                    #print(f"self.data_dict['gridded_data'][key].shape: {self.data_dict['gridded_data'][key].shape}")

            batch = np.max(self.data_dict["all_data"]["batch"]) + 1
            self.data_dict["all_data"]["batch"] = np.concatenate((self.data_dict["all_data"]["batch"], np.repeat(batch, len(data_dict["S"]))), axis=0)
            self.data_dict["gridded_data"]["batch"] = np.concatenate((self.data_dict["gridded_data"]["batch"], np.repeat(batch, len(data_dict_gridded["S_grid"]))), axis=0)

  

    def refit_model(self, **kwargs):
        
        self.logger.log_info(f"[refit_model] Refitting the model")

        # IF we change some parameter in the model, we need to refit the model
        old_data_dict = self.data_dict
        
        # Set the old data to clean state
        self.clean_model()

        data_dict = {
            "S": old_data_dict["all_data"]["S"],
            "y": old_data_dict["all_data"]["y"],
            "T": old_data_dict["all_data"]["T"],
            "upper_bound": old_data_dict["all_data"]["upper_bound"],
            "lower_bound": old_data_dict["all_data"]["lower_bound"],
            "data_source": old_data_dict["all_data"]["data_source"]
        }

        # Add the data to the model
        self.add_data_to_model(data_dict)


    def clean_model(self):
        """
        Clean the model
        """
        f_name = "clean_model"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Cleaning the model")
        self.data_dict = {}
        self.data_dict["has_data"] = False
        self.data_dict["gridded_data"] = {}
        self.data_dict["all_data"] = {}
        self.timing.end_time_id(t_tok, return_time_stamp=True)



    def fit(self, **kwargs):
        """
        Fit the model to the data

        Fitting the model is done by using the gridded data
        this is because it greatly reduces the size of the data

        this should also happen on the transformed scale
        y* = f(y)
        where y* is the transformed data
        TODO:
        - Need to take into account covariance from boundary and 
        """

        f_name = "fit"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Fitting the model")

        ## Step 1: Load the data friom the gridded data
        if self.data_dict["has_data"] == False:
            raise ValueError("No data has been added to the model. Please add data before fitting the model.")
        
        prior_y = self.data_dict["gridded_data"]["prior_y_corrected"]   # We use the corrected prior
        prior_lower_bound = self.data_dict["gridded_data"]["prior_lower_bound"]
        prior_upper_bound = self.data_dict["gridded_data"]["prior_upper_bound"]
        y = self.data_dict["gridded_data"]["y_corrected"] # This is the corrected data
        S = self.data_dict["gridded_data"]["S_grid"]
        T = self.data_dict["gridded_data"]["T_grid"]
        upper_bound = self.data_dict["gridded_data"]["upper_bound"]
        lower_bound = self.data_dict["gridded_data"]["lower_bound"]

        # load the data that is only made after the fitting 
        m_y = self.data_dict["gridded_data"].get("m_y", None)           # Fitted mean for y
        m_lower_bound = self.data_dict["gridded_data"].get("m_lower_bound", None) # Fitted mean for lower bound
        m_upper_bound = self.data_dict["gridded_data"].get("m_upper_bound", None) # Fitted mean for upper bound


        # Getting number of points in the model and the new points
        n_fitted = len(m_y) if m_y is not None else 0 
        n_new = len(y) - n_fitted      
        n = n_fitted + n_new         

        self.logger.log_info(f"[{f_name}] Number of data total in model: {n}")              # Logging
        self.logger.log_info(f"[{f_name}] Number of data points fitted: {n_fitted}")        # Logging
        self.logger.log_info(f"[{f_name}] Number of new data points: {n_new}")              # Logging

        Sigma_y = self.data_dict["gridded_data"].get("Sigma_y", None)
        Sigma_bounds = self.data_dict["gridded_data"].get("Sigma_bounds", None)
        if Sigma_y is None:
            Sigma_y = self.covariance_y.make_covariance_matrix(S, T)
            Sigma_bounds = self.covariance_bounds.make_covariance_matrix(S, T)

            self.data_dict["gridded_data"]["Sigma_y"] = Sigma_y
            self.data_dict["gridded_data"]["Sigma_bounds"] = Sigma_bounds
        else:
            S_old = S[0:n_fitted]
            T_old = T[0:n_fitted]
            S_new = S[n_fitted:]
            T_new = T[n_fitted:]

            Sigma_y = self.covariance_y.update_covariance_matrix(Sigma_y, S_old, S_new, T_old, T_new)
            Sigma_bounds = self.covariance_bounds.update_covariance_matrix(Sigma_bounds, S_old, S_new, T_old, T_new)
            self.data_dict["gridded_data"]["Sigma_y"] = Sigma_y
            self.data_dict["gridded_data"]["Sigma_bounds"] = Sigma_bounds
        
        # Get conditional mean and covariance
        P = np.eye(len(y)) * self.get_tau()**2   # The noise is not necessarily the same for all sources

        # Matrix inverse This can be optimized using Sherman Woodbury formula
        tok_inv = self.timing.start_time(f"{f_name}.inverse")
        Sigma_y_P_inv = np.linalg.inv(Sigma_y + P)
        Sigma_bounds_P_inv = np.linalg.inv(Sigma_bounds + P)
        self.timing.end_time_id(tok_inv)

        # Matrix multiplications
        tok_mat_mul = self.timing.start_time(f"{f_name}.matrix_multiplication")
        Sigma_y_at_Sigma_y_P_inv = Sigma_y @ Sigma_y_P_inv
        Sigma_bounds_at_Sigma_bounds_P_inv = Sigma_bounds @ Sigma_bounds_P_inv
        Psi_y = Sigma_y - Sigma_y_at_Sigma_y_P_inv @ Sigma_y.T
        Psi_bounds = Sigma_bounds - Sigma_bounds_at_Sigma_bounds_P_inv @ Sigma_bounds.T
        m_y = prior_y + Sigma_y_at_Sigma_y_P_inv @ (y - prior_y)
        m_lower_bound = prior_y + Sigma_bounds_at_Sigma_bounds_P_inv @ (lower_bound - prior_lower_bound)
        m_upper_bound = prior_y + Sigma_bounds_at_Sigma_bounds_P_inv @ (upper_bound - prior_upper_bound)
        self.timing.end_time_id(tok_mat_mul)

        # Store the updated values in the data dictionary
        # Predicted mean
        self.data_dict["gridded_data"]["m_y"] = m_y
        self.data_dict["gridded_data"]["m_y_uncorrected"] = self.data_transformation_y_inv(m_y)
        self.data_dict["gridded_data"]["m_lower_bound"] = m_lower_bound
        self.data_dict["gridded_data"]["m_upper_bound"] = m_upper_bound

        # Conditional covariance
        self.data_dict["gridded_data"]["Psi_y"] = Psi_y
        self.data_dict["gridded_data"]["Psi_bounds"] = Psi_bounds
        
        # For faster computation later
        self.data_dict["gridded_data"]["Sigma_y_P_inv"] = Sigma_y_P_inv
        self.data_dict["gridded_data"]["Sigma_bounds_P_inv"] = Sigma_bounds_P_inv
        self.data_dict["gridded_data"]["Sigma_y"] = Sigma_y
        self.data_dict["gridded_data"]["Sigma_bounds"] = Sigma_bounds


        # Log the end of the fitting        
        time_end = self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time taken for fitting: {time_end - t1:.2f} seconds")

  

    def predict(self, S, T, **kwargs):
        """
        Predict the model

        Here we use _A for what is in the model and _B for what we want to predict
        S_A: grid points in model
        T_A: time points in model
        S_B: points we want to predict, must not be on the grid
        T_B: time points we want to predict, must not be on the grid
        """
        f_name = "predict"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)

        S_A = self.data_dict["gridded_data"]["S_grid"]
        T_A = self.data_dict["gridded_data"]["T_grid"]
        S_B = S
        T_B = T

        # Load model and estimated values
        Psi_y_A = self.data_dict["gridded_data"]["Psi_y"]
        Sigma_y_A = self.data_dict["gridded_data"]["Sigma_y"]
        Sigma_y_A_P_inv = self.data_dict["gridded_data"]["Sigma_y_P_inv"]
        Sigma_bounds_A = self.data_dict["gridded_data"]["Sigma_bounds"]
        Sigma_bounds_A_P_inv = self.data_dict["gridded_data"]["Sigma_bounds_P_inv"]

        # Get the estimated mean values
        m_y_A = self.data_dict["gridded_data"]["m_y"]
        m_lb_A = self.data_dict["gridded_data"]["m_lower_bound"]
        m_ub_A = self.data_dict["gridded_data"]["m_upper_bound"]

        # Get the prior values
        mu_y_A = self.data_dict["gridded_data"]["prior_y_corrected"]
        mu_lb_A = self.data_dict["gridded_data"]["prior_lower_bound"]
        mu_ub_A = self.data_dict["gridded_data"]["prior_upper_bound"]

        # Get priors for S_B and T_B
        mu_y_B_cor, mu_y_B, mu_ub_B, mu_lb_B = self.get_prior(S_B, T_B)


        # Get the cross covariance matrix
        # TODO: the spatial correlation for y, lower bound and upper bound could be the same, but
        # the covariance matric sould be different due to the scale in sigma!
        Sigma_y_AB = self.covariance_y.make_covariance_matrix_2(S_B, S_A, T_B, T_A)
        Sigma_y_B = self.covariance_y.make_covariance_matrix(S_B, T_B)
        Sigma_bounds_AB = self.covariance_bounds.make_covariance_matrix_2(S_B, S_A, T_B, T_A)
        Sigma_bounds_B = self.covariance_bounds.make_covariance_matrix(S_B, T_B)
        
        tok_mat_mul = self.timing.start_time(f"{f_name}.matrix_multiplication")
        # Matrix multiplication
        Sigma_y_AB_at_Sigma_y_A_P_inv = Sigma_y_AB @ Sigma_y_A_P_inv
        Psi_y_B = Sigma_y_B - Sigma_y_AB_at_Sigma_y_A_P_inv @ Sigma_y_AB.T
        m_y_B = mu_y_B_cor + Sigma_y_AB_at_Sigma_y_A_P_inv  @ (m_y_A - mu_y_A)

        Sigma_bounds_AB_at_Sigma_bounds_A_P_inv = Sigma_bounds_AB @ Sigma_bounds_A_P_inv
        Psi_bounds_B = Sigma_bounds_B - Sigma_bounds_AB_at_Sigma_bounds_A_P_inv @ Sigma_bounds_AB.T
        m_lb_B = mu_lb_B + Sigma_bounds_AB_at_Sigma_bounds_A_P_inv @ (m_lb_A - mu_lb_A)
        m_ub_B = mu_ub_B + Sigma_bounds_AB_at_Sigma_bounds_A_P_inv @ (m_ub_A - mu_ub_A)
        self.timing.end_time_id(tok_mat_mul)

        # Store the predictions in the data dictionary
        predictions = {}
        predictions["m_y"] = m_y_B
        predictions["m_lower_bound"] = m_lb_B
        predictions["m_upper_bound"] = m_ub_B
        predictions["prior_y_corrected"] = mu_y_B_cor
        predictions["prior_y"] = mu_y_B
        predictions["prior_lower_bound"] = mu_lb_B
        predictions["prior_upper_bound"] = mu_ub_B
        predictions["Psi_y"] = Psi_y_B
        predictions["Sigma_y"] = Sigma_y_B
        predictions["Psi_bounds"] = Psi_bounds_B
        predictions["Sigma_bounds"] = Sigma_bounds_B


        t2 = self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time taken for prediction: {t2 - t1:.2f} seconds")

        return predictions 
    

    def predict_field(self, **kwargs):
        """
        Predict the field
        """
        f_name = "predict_field"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)

        t = kwargs.get("t", time.time())
        full_sigma = kwargs.get("full_sigma", False)  # The mean prediction and dSigam will be the same
        S = self.grid.grid
        T = np.repeat(t, len(S))

        if full_sigma == True:
            # This is dangerous if the grid is large
            predictions = self.predict(S, T)
        else:   
            predictions = {
                "m_y": np.zeros(T.shape[0]),
                "m_lower_bound": np.zeros(T.shape[0]),
                "m_upper_bound": np.zeros(T.shape[0]),
                "prior_y_corrected": np.zeros(T.shape[0]),
                "prior_lower_bound": np.zeros(T.shape[0]),
                "prior_upper_bound": np.zeros(T.shape[0]),
                "dPsi_y": np.zeros(T.shape[0]),   # This is the main difference
                "dPsi_bounds": np.zeros(T.shape[0]),
            }

            

            # TODO: this can be done in parallel
            max_batch_size = kwargs.get("max_batch_size", 500)
            tok, t1 = self.timing.start_time(f"{f_name}.batch_size", return_time_stamp=True)
            batch_seperator = np.arange(0, len(S), max_batch_size, dtype=int)

            for i in range(len(batch_seperator) - 1):
                S_batch = S[batch_seperator[i]:batch_seperator[i + 1]]
                T_batch = T[batch_seperator[i]:batch_seperator[i + 1]]
                #print(f"S_batch.shape: {S_batch.shape}")
                #print(f"T_batch.shape: {T_batch.shape}")
                predictions_batch = self.predict(S_batch, T_batch)
                predictions["m_y"][batch_seperator[i]:batch_seperator[i + 1]] = predictions_batch["m_y"]
                predictions["m_lower_bound"][batch_seperator[i]:batch_seperator[i + 1]] = predictions_batch["m_lower_bound"]
                predictions["m_upper_bound"][batch_seperator[i]:batch_seperator[i + 1]] = predictions_batch["m_upper_bound"]
                predictions["prior_y_corrected"][batch_seperator[i]:batch_seperator[i + 1]] = predictions_batch["prior_y_corrected"]
                predictions["prior_lower_bound"][batch_seperator[i]:batch_seperator[i + 1]] = predictions_batch["prior_lower_bound"]
                predictions["prior_upper_bound"][batch_seperator[i]:batch_seperator[i + 1]] = predictions_batch["prior_upper_bound"]
                predictions["dPsi_y"][batch_seperator[i]:batch_seperator[i + 1]] = np.diag(predictions_batch["Psi_y"])
                predictions["dPsi_bounds"][batch_seperator[i]:batch_seperator[i + 1]] = np.diag(predictions_batch["Psi_bounds"])

                t2 = self.timing.end_time_id(tok, return_time_stamp=True)
                self.logger.log_info(f"[{f_name}] Time taken for batch size {max_batch_size}: {t2 - t1:.2f} seconds")


        t2 = self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time taken for prediction: {t2 - t1:.2f} seconds")
        return predictions, S, T
    

    def predict_inside_boundary(self, boundary_path, t=time.time()):
        # Predict the field inside a boundary for a given time
        f_name = "predict_inside_boundary"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Predicting inside boundary")

        # Get the boundary points

        t2 = self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time taken for prediction: {t2 - t1:.2f} seconds")
        pass
        #return predictions, S, T
    
    def get_prediction_top_xm(self, predictions, top_m=10):
        """
        Get the prediction for the top x meters
        """
        f_name = "get_prediction_top_xm"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)

        # Get the grid points
        m_y = predictions["m_y"]
        m_lower_bound = predictions["m_lower_bound"]
        m_upper_bound = predictions["m_upper_bound"]

        # 
        predictions_top_xm = np.zeros(m_y.shape)

        for i in range(len(m_y)):
            
            if m_upper_bound[i] >  top_m:
                predictions_top_xm[i] = 0
            elif m_lower_bound[i] <  top_m:
                predictions_top_xm[i] = m_y[i]
            else:
                # Get the value for the top m
                predictions_top_xm[i] = m_y[i] + (top_m - m_lower_bound[i]) / (m_upper_bound[i] - m_lower_bound[i]) * (m_y[i] - m_lower_bound[i])
            
        t2 = self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time taken for prediction: {t2 - t1:.2f} seconds")
        return predictions_top_xm

    

    def predict_field_prod(self, T, store_format="nc",  **kwargs):
        """
        Predict the field for several time steps and stores the results in a readable format
        """
        f_name = "predict_field_prod"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)

        full_sigma = kwargs.get("full_sigma", False)

        predictions_list = []



        for t in T:

            predictions, *S_T = self.predict_field(t=t, full_sigma=full_sigma)
            predictions_list.append(predictions)

        
        # Store the predictions in a formatted way

        if store_format == "nc":

            self.logger.log_info(f"[{f_name}] Storing predictions in netcdf format")
            t_tok2, t1 = self.timing.start_time(f"{f_name}.netcdf", return_time_stamp=True)

            # Store the predictions in a netcdf file
            x_ax = self.grid.get_grid_data_key("x_ax")
            y_ax = self.grid.get_grid_data_key("y_ax")

            
            # Create a netcdf file
            store_path = kwargs.get("store_path", ".")
            filname = kwargs.get("filename", "CASCADE_predictions.nc")
            nc_file = Dataset(filname, "w", format="NETCDF4")
            # Create dimensions
            nc_file.createDimension("time", len(T))
            nc_file.createDimension("xc", len(x_ax))
            nc_file.createDimension("yc", len(y_ax))

            # Create variables
            # Create coordinate variables
            time_var = nc_file.createVariable("time", float, ("time"))
            x_var = nc_file.createVariable("x", float, ("xc"))
            y_var = nc_file.createVariable("y", float, ("yc"))
        
            # Create coordinate variables
            lat_var = nc_file.createVariable("latitude", float, ("yc", "xc"))
            lon_var = nc_file.createVariable("longitude", float, ("yc", "xc"))

            S = self.grid.grid
            lon, lat = self.projection(S[:, 0], S[:, 1], inverse=True)
            lon_var[:] = lon.reshape((len(y_ax), len(x_ax)))
            lat_var[:] = lat.reshape((len(y_ax), len(x_ax)))


            # Create data variables
            m_y_var = nc_file.createVariable("biomass_estimate", float, ("time", "yc", "xc"))
            m_y_top_10m = nc_file.createVariable("biomass_estimate_top_10m", float, ("time", "yc", "xc"))
            dPsi_y_var = nc_file.createVariable("uncertainty_biomass", float, ("time","yc","xc"))
            information_gain = nc_file.createVariable("information_gain", float, ("time","yc","xc"))

            
            # Assign values to the variables
            time_var[:] = T
            x_var[:] = x_ax
            y_var[:] = y_ax

            
            for i, predictions in enumerate(predictions_list):
                m_y_var[i, :, :] = self.data_transformation_y_inv(predictions["m_y"]).reshape((len(y_ax), len(x_ax)))
                m_y_top_10m[i, :, :] = predictions["m_y"] # TODO: This should be the top 10m
                dPsi_y_var[i, :, :] = predictions["dPsi_y"].reshape((len(y_ax), len(x_ax)))
                information_gain[i, :, :] = predictions["dPsi_y"].reshape((len(y_ax), len(x_ax)))  # TODO: This should be the information gain

            # Set variable attributes
            m_y_var.units = "g/m^2"
            m_y_top_10m.units = "g/m^2"
            dPsi_y_var.units = "m"
            time_var.units = "seconds since 1970-01-01 00:00:00"
            x_var.units = "m"
            y_var.units = "m"
            time_var.long_name = "Time"
            x_var.long_name = "X coordinate"
            y_var.long_name = "Y coordinate"
            m_y_var.long_name = "Mean prediction for biomass"
            dPsi_y_var.long_name = "Uncertainty in Biomass prediction"


            # Setup from the sinmod file
            nc_file.grid_mapping = "grid_mapping"
            nc_file.grid_mapping_name = "polar_stereographic"
            nc_file.straight_vertical_longitude_from_pole = 58.0
            nc_file.horizontal_resolution =  160.0
            nc_file.coordinate_north_pole = [13201.0,  11643.5]
            nc_file.latitude_of_projection_origin =  90.0
            nc_file.standard_parallel = 60.0
            nc_file.barotropic_timestep = 0.4
            nc_file.baroclinic_timestep = 10.0
            nc_file._FillValue =  -32768
            nc_file.setup = "midnor"
            nc_file.relax_e = "T"
            nc_file.nested = "F"
            nc_file.tidal_input = "F"
            nc_file.DHA =  1.0
            nc_file.smagorin = "T"
            nc_file.biharmonic = "F"
            nc_file.KBi = 0.0
            nc_file.COLDSTART = "F"
            nc_file.ATMODATA = 5
            nc_file.CM = 1.0
            nc_file.CM2D = 1.0
            nc_file.CH = 0.3
            nc_file.CI = 5.0


            # Add global attributes
            nc_file.description = "Predictions from the CASCADE model"
            nc_file.history = f"Created {datetime.datetime.now()}"
            nc_file.source = "CASCADE model"
            nc_file.author = "André Olaisen"
            nc_file.institution = "NTNU IMF"
            nc_file.contact = "andre.j.h.olaisen@ntnu.no"
            nc_file.version = "1.0"
            nc_file.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            nc_file.title = "Predictions from the CASCADE model"

            nc_file.close()
            t2 = self.timing.end_time_id(t_tok2, return_time_stamp=True)
            self.logger.log_info(f"[{f_name}] Predictions stored in {filname}")

        self.timing.end_time_id(t_tok)
        return predictions_list
    








    def __log_determinant(K, tau, sigma):
        n = len(K)
        L = np.linalg.cholesky(K + np.eye(n) * tau**2)
        determinant = 2 * np.sum(np.log(np.diag(L))) + n * np.log(sigma ** 2) 
        return determinant

    def log_likelihood(self, **kwargs):
        """
        Compute the log likelihood of the fitted model
        """

        f_name = "log_likelihood"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Computing log likelihood")

        # Get the data
        if self.data_dict["has_data"] == False:
            raise ValueError("No data has been added to the model. Please add data before fitting the model.")
        
        y = self.data_dict["gridded_data"]["y_corrected"]
        m_y = self.data_dict["gridded_data"]["m_y"]
        Psi = self.data_dict["gridded_data"]["Psi"]
        Sigma_P_inv = self.data_dict["gridded_data"]["Sigma_P_inv"]

        # Compute the log likelihood
        log_likelihood = -0.5 * (np.log(np.linalg.det(Psi)) + (y - m_y).T @ Sigma_P_inv @ (y - m_y))
        
        t2 = self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time taken for log likelihood: {t2 - t1:.2f} seconds")
        return log_likelihood

          
    def get_prior(self, S, T, **kwargs):
        """
        Gets the values from the prior model
        """
        f_name = "get_prior"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)
        single_val = False
        if not hasattr(T, "__len__"):
            T = np.array([T, T])
            S = np.concatenate((S, S), axis=0)
            single_val = True


        uncorrected_prior = self.prior.get_prior_S_T(S, T)
        # Correct the prior values
        corrected_prior = self.data_transformation_sinmod(S, T, uncorrected_prior)

        # Get the prior bounds
        # TODO: Implement the prior bounds
        upper_bound = np.repeat(10, len(T))
        lower_bound = np.repeat(30, len(T))

        t2 = self.timing.end_time_id(t_tok, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time taken for getting prior: {t2 - t1:.2f} seconds")

        if single_val == True:
            self.logger.log_debug(f"[{f_name}] Single value prior: {corrected_prior}")
            corrected_prior = corrected_prior[0]
            uncorrected_prior = uncorrected_prior[0]
            upper_bound = upper_bound[0]
            lower_bound = lower_bound[0]
        return corrected_prior, uncorrected_prior, upper_bound, lower_bound


    def print_data_shape(self):
        """
        Print the shape of the data
        """
        print("Data shape:")
        for key in self.data_dict["all_data"].keys():
            print("\t", end="")
            if isinstance(self.data_dict["all_data"][key], np.ndarray):
                print(f"{key}: {self.data_dict['all_data'][key].shape}")
            else:
                if isinstance(self.data_dict["all_data"][key], list):
                    print(f"{key}: {len(self.data_dict['all_data'][key])}")
                else:
                    if hasattr(self.data_dict["all_data"][key], "__len__"):
                        print(f"{key}: {len(self.data_dict['all_data'][key])}")
                    else:
                        print(f"{key}: {self.data_dict['all_data'][key]}")
        print("Gridded data shape:")
        for key in self.data_dict["gridded_data"].keys():
            print("\t", end="")
            if isinstance(self.data_dict["gridded_data"][key], np.ndarray):
                print(f"{key}: {self.data_dict['gridded_data'][key].shape}")
            else:
                if isinstance(self.data_dict["gridded_data"][key], list):
                    print(f"{key}: {len(self.data_dict['gridded_data'][key])}")
                else:
                    if hasattr(self.data_dict["gridded_data"][key], "__len__"):
                        print(f"{key}: {len(self.data_dict['gridded_data'][key])}")
                    else:
                        print(f"{key}: {self.data_dict['gridded_data'][key]}")

    ################################################
    ######## # Getters and setters # ##############
    ################################################


    ############ Parameters ############
    def get_tau(self, vehicle=None, variable=None):
        """
        Get the tau parameter
        - vehicle: vehicle to get the tau for; AUV thor, ASV greta, Autonaught
        - variable: variable to get the tau for; y, lower bound, upper bound
        """
        return self.model_parameters["tau"]
    

    def set_tau(self, tau, vehicle=None, variable=None):

        if vehicle is not None and variable is not None:
            # Set the tau for the specific vehicle and variable
            self.model_parameters["tau"][vehicle][variable] = tau
        
        self.model_parameters["tau"] = tau
        self.covariance.set_tau(tau)
        self.logger.log_info(f"Set tau to {tau}")

    def get_sigma(self, variable="y"):
        """
        Get the sigma parameter
        - variable: variable to get the sigma for; y, lower bound, upper bound
        """
        return self.model_parameters["sigma"][variable]
    
    def set_sigma(self, sigma, variable="y"):
        """
        Set the sigma parameter
        - variable: variable to set the sigma for; y, lower bound, upper bound
        """
        self.model_parameters["sigma"][variable] = sigma
        if variable == "y":
            self.covariance_y.set_sigma(sigma)
        if variable == "bound":
            self.covariance_bounds.set_sigma(sigma)
        self.logger.log_info(f"Set sigma for variable {variable} to {sigma}")

    def set_phi_xy(self, phi_xy):
        """
        Set the phi_xy parameter
        - phi_xy: phi_xy parameter
        """
        self.model_parameters["phi_xy"] = phi_xy
        self.covariance_y.set_phi_xy(phi_xy)
        self.covariance_bounds.set_phi_xy(phi_xy)
        self.logger.log_info(f"Set phi_xy to {phi_xy}")

    def get_phi_xy(self):
        """
        Get the phi_xy parameter
        """
        return self.model_parameters["phi_xy"]
    def set_phi_z(self, phi_z):
        """
        Set the phi_z parameter
        - phi_z: phi_z parameter
        """
        self.model_parameters["phi_z"] = phi_z
        self.covariance_y.set_phi_z(phi_z)
        self.covariance_bounds.set_phi_z(phi_z)
        self.logger.log_info(f"Set phi_z to {phi_z}")
    

    
    ############ Data ############
    # For several of the getters we have the option to get the gridded data or the non gridded data
    # THe fitted model is only done on the gridded data
    # For several we can also get the corrected data or the non corrected data
    # The corrected data is the data that is used for the fitting
    def get_data_T_grid_inds(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                return self.data_dict["gridded_data"]["T_grid_inds"]
            else:
                return self.data_dict["all_data"]["T_grid_inds"]
        
    def get_data_T_grid(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                return self.data_dict["gridded_data"]["T_grid"]
            else:
                return self.data_dict["all_data"]["T_grid"]
        
    def get_data_S_grid_inds(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                return self.data_dict["gridded_data"]["S_grid_inds"]
            else:
                return self.data_dict["all_data"]["S_grid_inds"]
        

    def get_data_S_grid(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                return self.data_dict["gridded_data"]["S_grid"]
            else:
                return self.data_dict["all_data"]["S_grid"]
            
    def get_data_S(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                # Here we return the S grid points
                return self.data_dict["gridded_data"]["S_grid"]
            else:
                # Here we return the S points that are not gridded
                return self.data_dict["all_data"]["S"]
            
    def get_data_T(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                # Here we return the T grid points
                return self.data_dict["gridded_data"]["T_grid"]
            else:
                # Here we return the T points that are not gridded
                return self.data_dict["all_data"]["T"]
            
    def get_data_m_y(self, corrected=True):
        if self.data_dict["has_data"] == False:
            return None
        else:
            # The model is only fitted to the gridded data
            if corrected == True:
                return self.data_dict["gridded_data"]["m_y"]
            else:
                transfomed_m_y = self.data_transformation_y(self.data_dict["gridded_data"]["m_y"])
                return transfomed_m_y
        
    def get_data_prior_y(self, gridded=True, corrected=True):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                # The model is only fitted to the gridded data
                if corrected == True:
                    return self.data_dict["gridded_data"]["prior_y_corrected"]
                else:
                    return self.data_dict["gridded_data"]["prior_y"]
            else:
                # The model is only fitted to the gridded data
                if corrected == True:
                    return self.data_dict["all_data"]["prior_y_corrected"]
                else:
                    return self.data_dict["all_data"]["prior_y"]

    
    def get_data_S_T_grid_inds(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                return self.data_dict["gridded_data"]["S_T_grid_inds"]
            else:
                return self.data_dict["all_data"]["S_T_grid_inds"]
    
    def get_data_S_T_grid(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                return self.data_dict["gridded_data"]["S_T_grid"]
            else:
                return self.data_dict["all_data"]["S_T_grid"]
            
    def get_data_batch(self, gridded=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                return self.data_dict["gridded_data"]["batch"]
            else:
                return self.data_dict["all_data"]["batch"]
            
    def get_data_y(self, gridded=False, corrected=False):
        if self.data_dict["has_data"] == False:
            return None
        else:
            if gridded == True:
                if corrected == True:
                    return self.data_dict["gridded_data"]["y_corrected"]
                else:
                    return self.data_dict["gridded_data"]["y"]
            else:
                if corrected == True:
                    return self.data_dict["all_data"]["y_corrected"]
                else:
                    return self.data_dict["all_data"]["y"]
                
    def get_data_Psi_y(self):
        if self.data_dict["has_data"] == False:
            return None
        else:
            return self.data_dict["gridded_data"]["Psi_y"]
    
    def get_data_Psi_bounds(self):
        if self.data_dict["has_data"] == False:
            return None
        else:
            return self.data_dict["gridded_data"]["Psi_bounds"]
                


    def store_data(self, path, csv=True, do_pickle=True, fit_model=True):
        """
        Store the data in a file
        """
        f_name = "store_data"
        t_tok, t1 = self.timing.start_time(f_name, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Storing data to {path}")

        # Perhaps be a bit more careful about the naming of the files


        # Store the data in a csv file
        if csv == True:
            all_data_df = pd.DataFrame({"batch": self.data_dict["all_data"]["batch"]})
            for key in self.data_dict["all_data"].keys():
                if isinstance(self.data_dict["all_data"][key], np.ndarray):
                    if len(self.data_dict["all_data"][key].shape) == 1:
                        all_data_df[key] = self.data_dict["all_data"][key]
                    else:
                        if self.data_dict["all_data"][key].shape[0] == self.data_dict["all_data"][key].shape[1]:
                            # If the data is a matrix, we only store the diagonal
                            all_data_df["d" + key] = np.diag(self.data_dict["all_data"][key])
                        else:
                            # Make a list of the data
                            list_data = [v for v in self.data_dict["all_data"][key]]
                            all_data_df[key] = list_data
                else:
                    all_data_df[key] = self.data_dict["all_data"][key]

            # The gridded data can be stored in a few different ways
            #  - Matrices are not great for storing in csv

            gridded_data_df = pd.DataFrame({"batch": self.data_dict["gridded_data"]["batch"]})
            for key in self.data_dict["gridded_data"].keys():
                if isinstance(self.data_dict["gridded_data"][key], np.ndarray):
                    if len(self.data_dict["gridded_data"][key].shape) == 1:
                        gridded_data_df[key] = self.data_dict["gridded_data"][key]
                    else:
                        if self.data_dict["gridded_data"][key].shape[0] == self.data_dict["gridded_data"][key].shape[1]:
                            # If the data is a matrix, we only store the diagonal
                            gridded_data_df["d" + key] = np.diag(self.data_dict["gridded_data"][key])
                else:
                    gridded_data_df[key] = self.data_dict["gridded_data"][key]
            
            
            all_data_df.to_csv(path + "/all_data.csv", index=False)
            gridded_data_df.to_csv(path + "/gridded_data.csv", index=False)
            self.logger.log_info(f"[{f_name}] Data stored in {path}/all_data.csv and {path}/gridded_data.csv")

        if pickle == True:
            # Store the data in a pickle file
            with open(path + "/all_data.pkl", "wb") as f:
                pickle.dump(self.data_dict["all_data"], f)
            with open(path + "/gridded_data.pkl", "wb") as f:
                pickle.dump(self.data_dict["gridded_data"], f)
            self.logger.log_info(f"[{f_name}] Data stored in {path}/all_data.pkl and {path}/gridded_data.pkl")


    def turn_on_logging(self):
        self.logger.print_to_console = True
        self.covariance.logger.print_to_console = True
        self.prior.logger.print_to_console = True

    def turn_off_logging(self):
        self.logger.print_to_console = False
        self.covariance.logger.print_to_console = False
        self.prior.logger.print_to_console = False









if __name__=="__main__":

    from plotting.CascadePlotting import CascadePlotting

    folder = "figures/tests/Cascade"


    ###########################################
    # Setting up the sinmod model
    ###########################################
    # Example usage
    wdir = get_project_root()
    folder_test = os.path.join(wdir, "figures/tests/Cascade")
    sinmod_path = os.path.join(wdir, "data/sinmod/")
    files = ["BioStates_froshelf.nc", "BioStates_midnor.nc"]
    plot_test_path = "/figures/tests/Sinmod/"
    plot_test_path = os.path.join(wdir, plot_test_path)

    file_ind = 0
    logging_kwargs = { 
        "log_file": folder + "/Sinmod.log",
        "print_to_console": True,
        "overwrite_file": True
    }
    covariance_kwargs={
                     "phi_xy": 5000,
                     "phi_temporal": 60*60*4,
                     "sigma": 1,
                     "sigma_y": 1,
                     "sigma_bounds": 1,
                     "temporal_covariance_type": "exponential",
                     "xy_covariance_type": "matern_3_2",}

    boundary_kwargs = {
            "border_file": "/border_files/cascade_test_xy.csv",
            "file_type": "xy"}
    
    model_parameters = {
        "dxdy": 160,
        "dt": 60*60,
        "tau": 0.5
    }

    
    sinmod_c = Sinmod(sinmod_path + "/" + files[file_ind], 
                        plot_path=plot_test_path,
                        log_kwargs=logging_kwargs,
                        boundary_kwargs=boundary_kwargs,
                      print_while_running=True)

    
    logging_kwargs = {
        "log_file": folder + "/cascade_log.txt",
        "print_to_console": True,
        "overwrite_file": True
    }

    cascade = Cascade(prior=sinmod_c,
                      model_parameters=model_parameters,
                        covariance_kwargs=covariance_kwargs,
                        boundary_kwargs=boundary_kwargs,
                        log_kwargs=logging_kwargs)
    
    ##### Test inverse function
    CascadePlotting.test_inverse_functions(cascade)
    
    ##### Testing the grid
    CascadePlotting.plot_grid(cascade)
    

    ##### Testing the line data assignment
    CascadePlotting.line_data_assignmen(cascade)
    
    #### Test gap in timing
    #time.sleep(3) # REmove this later, just to check the timing plotting

    ###### Testing multiple line assignments
    CascadePlotting.multiple_line_assignments(cascade)

    ###### Predicting the field
    CascadePlotting.plot_predict_field(cascade)

    ###### Store predictions 
    T = [time.time() + 60*60*i for i in range(4)]
    cascade.predict_field_prod(T, store_format="nc", filename=folder + "/predictions3.nc")

    ##### Print the data shape
    cascade.print_data_shape()

    #### Store the data
    cascade.store_data(folder, csv=True, do_pickle=True)


    #### Test the predictions 
    CascadePlotting.plot_predicions(cascade)


    #### 
    cascade.predict_field(full_sigma=False)

    ##################################################  
    # Try meging the data 
    cascade.timing.merge_timing_data(cascade.prior.timing)
    cascade.timing.merge_timing_data(cascade.covariance.timing)
    cascade.timing.plot_timing_data(save_path=folder, plot_name="timing_data_cascade")