import time
from Cascade import Cascade
from Sinmod import Sinmod

import pandas as pd
from utilis.Logger import Logger
from utilis.Timing import Timing
from utilis.Grid import Grid
from utilis.utility_funcs import *
import os


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import datetime


class RunModel:
    """
    Class to run the model.
    """

    def __init__(self, config,  
                 cascade):
        """
        Initialize the RunModel class.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.cascade = cascade

        self.file_locations = {
            "data_files": config["file_locations"]["data_files"], # A list of directories to watch for new data files
            "prediction_requests": config["file_locations"]["prediction_requests"],
            "prediction_requests_filled": config["file_locations"]["predictions"],
        }
        self.files = {

            "all_files": {
                "data_files": [],
                "prediction_requests": [],
            },
            "processed_files": {
                "data_files": [],
                "prediction_requests": [],
            }   
        }


    def file_observer(self):
        """
        Set up the file observer to monitor directories for new files.

        # Returns:
            - None if there is nothing new
            - file path and file type if there is a new file

        """
        
        # S1 : check if there are any new data files

        for data_file_loc in self.file_locations["data_files"]:

            # List all files in the directory
            files = [f for f in os.listdir(data_file_loc) if os.path.isfile(os.path.join(data_file_loc, f))]
            # Check if there are any new files
            for file in files:
                file_path = os.path.join(data_file_loc, file)
                if file_path not in self.files["all_files"]["data_files"]:
                    self.files["all_files"]["data_files"].append(file_path)
                    return file_path, "data_files"
        
        # S2 : check if there are any new prediction requests
        for prediction_request_loc in self.file_locations["prediction_requests"]:
            # List all files in the directory
            files = [f for f in os.listdir(prediction_request_loc) if os.path.isfile(os.path.join(prediction_request_loc, f))]
            # Check if there are any new files
            for file in files:
                file_path = os.path.join(prediction_request_loc, file)
                if file_path not in self.files["all_files"]["prediction_requests"]:
                    self.files["all_files"]["prediction_requests"].append(file_path)
                    return file_path, "prediction_requests"


        
        return None, None

    
    def add_data_file_to_cascade(self, file):
        """
        Add a data file to the cascade model.

        Args:
            file (str): Path to the data file.
        """

        df = pd.read_csv(file)  
        
        print(df.head())    
        # TODO: add transfomation from x y to lon lat

        data_dict = {
            "Sx": df["x"].values,
            "Sy": df["y"].values,
            "y": np.clip(df["biomass"].values, 0, None),
            #"y": df["biomass"].values,
            "T": df["t"].values,
            "upper_bound": df["upper_bounds"].values,
            "lower_bound": df["lower_bounds"].values,
            "vehicle": df["vehicle"].values,
        }

        self.cascade.add_data_to_model(data_dict)

    def run(self):
        """
        Run the model.
        """

        # Set up the file observer

        sleep_counter = 0
        t0 = time.time()

        while True:

            # s1 observe and update the files
            file, file_type =  self.file_observer()

                

            if file:
                print("" + "-"*100)

                print(f"New file detected: {file}, type: {file_type}") 
                # s1:assimilate if there are any new data files then update the cascade model
                if file_type == "data_files":
                    self.add_data_file_to_cascade(file)
                    self.files["processed_files"]["data_files"].append(file)
                    print(f"Assimilated data file: {file}")

                # s1:predict if there are any new prediction requests then predict
                elif file_type == "prediction_requests":
                    #self.cascade.predict(file)
                    self.files["processed_files"]["prediction_requests"].append(file)
                    print(f"Predicted using request file: {file}")


                # Reset the sleep counter
                t0 = time.time()

                print("" + "-"*100)

            elif file is None:
                # s1: if there is nothing new then sleep for a while
                print(f"No new files for {str(datetime.timedelta(seconds=round(time.time()-t0)))} ", end = "\r")
                time.sleep(5)
            else:
                # s1: if there is nothing new then sleep for a while
                print(f"No new files for {str(datetime.timedelta(seconds=round(time.time()-t0)))} ", end = "\r")
                time.sleep(5)


if __name__ == "__main__":
    # Example configuration
    config = {
        "file_locations": {
            "data_files": ["data/simulated_data/"],
            "prediction_requests": ["data/requests/"],
            "predictions": ["data/predictions/"],
        }
    }

    folder = "figures/tests/RunModel"
    if not os.path.exists(folder):
        os.makedirs(folder)


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
    

    # Initialize the RunModel class
    run_model = RunModel(config, 
                         cascade=cascade)  # Replace with actual Cascade object
    
    run_model.run()










    