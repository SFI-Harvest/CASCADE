import datetime
import pandas as pd
import os

class Logger:
    def __init__(self, class_name, do_log=True, log_file=None, print_to_console=True, overwrite_file=False):
        self.class_name = class_name
        self.log_file = log_file
        self.do_log = do_log

        """
        TODO: implement the log modes
        Log modes:
        "all_p_l" - log all messages to the log file and print to console
        "all_p" - log all messages by printing to console
        "all_l" - log all messages to the log file
        "w_p_l" - log warning and error messages to the log file and print to console
        "w_p" - log warning and error messages by printing to console
        "w_l" - log warning and error messages to the log file
        "nd_p_l" - all messages except debug messages to the log file and print to console
        "nd_p" - all messages except debug messages by printing to console
        "nd_l" - all messages except debug messages to the log file

        """
        self.log_mode = "a" 

        self.print_to_console = print_to_console
        self.__log_data = {}
        self.__log_data["df"] = pd.DataFrame(columns=["time", "level", "class_name", "message"])
        self.__log_data["time"] = []
        self.__log_data["level"] = []
        self.__log_data["class_name"] = []
        self.__log_data["message"] = []

        # Check if the log file exists
        if log_file is not None:
            # Check if the log file exists
            if os.path.exists(log_file):
                if overwrite_file:
                    # Overwrite the log file
                    with open(log_file, "w") as f:
                        f.write("")
                else:
                    # Append to the log file
                    self.__rename_log_file_name(log_file)
            else:
                # Create the log file
                with open(log_file, "w") as f:
                    f.write("")
    @staticmethod
    def __rename_log_file_name(log_file, new_name=None):
        """
        This function renames the log file to a new name.
        This is done to avoid overwriting an existing log file.
        if new_name is None, the log file will be renamed to the current date and time
        """
        path_log_file = os.path.dirname(log_file)
        old_name = os.path.basename(log_file)
        old_name = old_name.split(".")[0]
        file_type = os.path.splitext(log_file)[1]
        if os.path.exists(log_file):
            if new_name is not None:
                new_log_file = os.path.join(path_log_file, old_name + new_name + file_type)
                os.rename(log_file, new_log_file)
            # Check if the log file is empty
            if os.path.getsize(log_file) == 0:
                return
            
            else:
                # Read the first line of the log file
                with open(log_file, "r") as f:
                    first_line = f.readline()
                    # Get the timestamp from the first line
                    timestamp_date = first_line.split(" ")[0]
                    timestamp_time = first_line.split(" ")[1]
                    time_stamp_date = timestamp_date.replace("-", "")
                    time_stamp_time = timestamp_time.replace(":", "")
                    timestamp = time_stamp_date + "_" + time_stamp_time
    
                    # Create the new log file name
                    new_log_file = os.path.join(path_log_file, old_name + "_" + timestamp + file_type)
                    os.rename(log_file, new_log_file)
        else:
            raise FileNotFoundError(f"Log file {log_file} does not exist.")
            
    

    def time_now_str(self):
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    def time_now(self):
        return datetime.datetime.now()
    
    def log_debug(self, message, **kwargs):
        """
        Log a debug message
        """
        self.log(message, level="DEBUG")
    
    def log_info(self, message, **kwargs):
        """
        Log an info message
        """
        self.log(message, level="INFO")
    
    def log_warning(self, message, **kwargs):
        """
        Log a warning message
        """
        self.log(message, level="WARNING")

    def log_error(self, message):
        """
        Log an error message
        """
        self.log(message, level="ERROR", print_to_console=True)

    def get_log_df(self):
        """
        Get the log data as a pandas DataFrame
        """
        if len(self.__log_data["df"]) == len(self.__log_data["time"]):
            return self.__log_data["df"]
        else:
            return pd.DataFrame({
                "time": self.__log_data["time"],
                "level": self.__log_data["level"],
                "class_name": self.__log_data["class_name"],
                "message": self.__log_data["message"]
            })


    
    def log(self, message, level="INFO", print_to_console=True, **kwargs):
        if not self.do_log:
            return

        # Add the message to the log data
        self.__log_data["time"].append(self.time_now())
        self.__log_data["level"].append(level)
        self.__log_data["class_name"].append(self.class_name)
        self.__log_data["message"].append(message)

        if self.log_file is None and not self.print_to_console:
            return  # No logging is required
        
        # Log the message with the class name
        if self.print_to_console:
            lvl = "[" + level + "]"
            print(f"{self.time_now_str()} {lvl:>9} [{self.class_name}] {message}")

       
        if self.log_file is not None:
            # Write the message to the log file
         
            with open(self.log_file, "a") as f:
                f.write(f"{self.time_now_str()} [{level}] [{self.class_name}] {message}\n")


    def print_log(self, levels = ["INFO", "ERROR", "WARNING", "DEBUG"],last_n_lines=10):
        """
        Print the last n lines of the log data
        """
        log_df = self.get_log_df() # Need to do this to update the log data
        if len(log_df) == 0:
            print("No log data available.")
            return
        if len(log_df) < last_n_lines:
            last_n_lines = len(log_df)
        for i, row in log_df.tail(last_n_lines).iterrows():
            print(f"{row['time']} [{row['level']}] [{row['class_name']}] {row['message']}")
      

    def print_warnings(self):
        """
        This function prints all the warnings in the log data
        """
        log_df = self.get_log_df()
        if len(log_df) == 0:
            print(self.time_now_str(), "[Logger]", "[Info]", "No log data available.")
            return
        if len(log_df[log_df["level"] == "WARNING"]) == 0:
            print(self.time_now_str(), "[Logger]", "[Info]", "No warnings in the log data.")
            return
        for i, row in log_df[log_df["level"] == "WARNING"].iterrows():
            print(f"{row['time']} [{row['level']}] [{row['class_name']}] {row['message']}")


    def merge_log(self, log, write_to_file=False):
        """
        Merge another logger's log data into this logger's log data.
        """
        if not isinstance(log, Logger):
            raise ValueError("log must be an instance of Logger")
        
        # Merge the log data
        self.__log_data["time"].extend(log.__log_data["time"])
        self.__log_data["level"].extend(log.__log_data["level"])
        self.__log_data["class_name"].extend(log.__log_data["class_name"])
        self.__log_data["message"].extend(log.__log_data["message"])
        
        # Update the DataFrame
        self.__log_data["df"] = pd.DataFrame({
            "time": self.__log_data["time"],
            "level": self.__log_data["level"],
            "class_name": self.__log_data["class_name"],
            "message": self.__log_data["message"]
        })

        if write_to_file and self.log_file is not None:
            # Write the merged log data to the log file
            # Sort the log data by time
            sorted_indices = sorted(range(len(log.__log_data["time"])), key=lambda i: log.__log_data["time"][i])
            sorted_log_data = {
                "time": [log.__log_data["time"][i] for i in sorted_indices],
                "level": [log.__log_data["level"][i] for i in sorted_indices],
                "class_name": [log.__log_data["class_name"][i] for i in sorted_indices],
                "message": [log.__log_data["message"][i] for i in sorted_indices]
            }
            # Write the sorted log data to the log file

            with open(self.log_file, "a") as f:
                for i in range(len(sorted_log_data["time"])):
                    f.write(f"{sorted_log_data['time'][i]} [{sorted_log_data['level'][i]}] [{sorted_log_data['class_name'][i]}] {sorted_log_data['message'][i]}\n")




if __name__ == "__main__":
    path_log_file = "/Users/ajolaise/Library/CloudStorage/OneDrive-NTNU/PhD/code/CASCADE/figures/tests/Logger"
    log_file_1 = path_log_file + "/test1.log"
    log_file_2 = path_log_file + "/test2.log"
    log_file_3 = path_log_file + "/test3.log"

    # Delete log files from a previous day


    logger = Logger("TestLogger", log_file=log_file_1, print_to_console=True)
    print("Logger 1 will print to console and rename any existing log file")
    logger2 = Logger("TestLogger2", log_file=log_file_2, print_to_console=False, overwrite_file=True)
    print("Logger 2 will not print to console and will overwrite the log file")
    logger3 = Logger("TestLogger3", log_file=log_file_3, print_to_console=True)
    # Example usage
    logger3.log_info("This is going to be a merged of logger1 and logger2")
    logger.log("This is a test message.")
    logger2.log("This is another test message.")
    logger.log_info("This is an info message.")
    logger2.log_info("This is another info message.")
    logger.log_warning("This is a warning message.")
    logger2.log_warning("This is another warning message.")
    logger.log_error("This is an error message.")
    logger2.log_error("This is another error message.")
    logger.log_debug("This is a debug message.")
    logger2.log_debug("This is another debug message.")

    print("Printing the last 2 lines of logger2")
    logger2.print_log(last_n_lines=2)
    print("Printing all warnings in logger2")
    logger2.print_warnings()

    print(logger.get_log_df())
    
    logger3.merge_log(logger, write_to_file=True)
    logger3.merge_log(logger2, write_to_file=True)

    print("Remeber to delete the log files after testing")
    print("or there will be a lot of log files in the folder")

    # Clean up log files
    all_files = os.listdir(path_log_file)
    n_keep = 2
    n_kept = 0
    print("Delete som old log files, but keep some of them to dempontrate the logger")
    for file in all_files:
        if file.endswith(".log") and file not in ["test1.log", "test2.log", "test3.log"]:
            if n_kept < n_keep:
                n_kept += 1
                continue
            os.remove(os.path.join(path_log_file, file))