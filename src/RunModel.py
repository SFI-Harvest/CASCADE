import time
from Cascade import Cascade


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


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
        self.cascade = Cascade(config)

        self.file_locations = {
            "data_files": config["data_files"],
            "data_files_assimilated": config["data_files_assimilated"],
            "prediction_requests": config["prediction_requests"],
            "prediction_requests_filled": config["prediction_requests_filled"],
        }
        self.files = {
            "data_files": [],
            "data_files_assimilated": [],
            "prediction_requests": [],
            "prediction_requests_filled": [],
        }


    def file_observer(self):
        """
        Set up the file observer to monitor directories for new files.

        # Returns:
            - None if there is nothing new
            - file path and file type if there is a new file

        """


    def run(self):
        """
        Run the model.
        """

        # Set up the file observer
   
        # s1 observe and update the files
        file, file_type = self.file_observer()

        if file:
            # s1:assimilate if there are any new data files then update the cascade model
            if file_type == "data_files":
                self.cascade.assimilate(file)
                self.files["data_files"].append(file)
                print(f"Assimilated data file: {file}")

            # s1:predict if there are any new prediction requests then predict
            elif file_type == "prediction_requests":
                self.cascade.predict(file)
                self.files["prediction_requests"].append(file)
                print(f"Predicted using request file: {file}")
        elif file is None:
            # s1: if there is nothing new then sleep for a while
            print("No new files. Sleeping for a while...")
            time.sleep(5)
        else:
            # s1: if there is nothing new then sleep for a while
            print("No new files. Sleeping for a while...")
            time.sleep(5)











    