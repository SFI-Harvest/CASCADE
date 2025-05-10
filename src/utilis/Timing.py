import time
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import pickle

class Timing:
    def __init__(self, class_name=None):
        """
        A class to keep track of the timing of different functions and classes 
        in the simulation.
        The class keeps track of the time stamps, function names, positions, and extra text.
        This timer is probably not great for very fast functions, but it is good for functions that take a bit of time to run.


        TODO:
        1. what to do with unclosed functions? When a function is not closed there will be no end time, this might be an 
        issue when we make the dataframes.

        # Example usages:
        1. Start a timer for a function:
        timing = Timing()
        function_id = timing.start_time("function_name", class_name="class_name", extra_text="extra_text")
        class.function()
        timing.end_time_id(function_id)  # Always end with an id


        2. Use in a class:
        class MyClass:
            def __init__(self):
                self.myclass_timing = Timing(class_name="MyClass")
            
            def my_function(self):
                function_id = self.myclass_timing.start_time("my_function")
                # Do something
                self.myclass_timing.end_time_id(function_id)

            def my_function2(self):
                # One can also get the time stamp
                function_id, time_start = self.myclass_timing.start_time("my_function2", return_time_stamp=True)
                # Do something
                time_inter = self.myclass_timing.add_intermittent_time_id(function_id, return_time_stamp=True)
                # Do something else
                time_end = self.myclass_timing.end_time_id(function_id, return_time_stamp=True)

                # Then we can print the same timing data
                print("time to do something: ", time_inter - time_start)
                print("time to do something else: ", time_end - time_inter)

            def my_function3(self, n):
                function_id = self.myclass_timing.start_time("my_function3")
                for i in range(n):
                    # Do something
                    self.myclass_timing.add_intermittent_time_id(function_id)
                self.myclass_timing.end_time_id(function_id)

            def my_function4(self):
                function_id = self.myclass_timing.start_time("my_function4")
                # Do something
                function_id_other = self.myclass_timing.start_time("my_function4_other", class_name="Other")
                other_class.do_something()
                self.myclass_timing.end_time_id(function_id_other)
                self.myclass_timing.end_time_id(function_id)

        3. Use in a function:
        global_timing = Timing()

        def my_function():
            function_id = global_timing.start_time("my_function")
            # Do something
            global_timing.end_time_id(function_id)
        
        4. Use in a function with a class:
        global_timing = Timing()

        """

        # Rewrite the class to using dicts

        self.timing_data = {} # A dictionary with the timing data for each function
        self.timing_data["counter"] = 0 # The number of function calls
        self.timing_data["lowest_unused_id"] = 0 # The lowest unused id
        self.timing_data["class_names"] = [] # The names of the classes
        self.timing_data["function_names"] = [] # The names of the functions
        self.timing_data["lists"] = {} # A dictionary with the lists for each function
        self.timing_data["arrays"] = {} # A dictionary with the arrays for each function
        """
        time_stamps: The time stamps for when something happened
        positions: The position of the time stamp (start, end, intermittent)
        positions_numeric: The position of the time stamp (1, -1, 0)
        function_list: A list with the ids of the functions
        function_name_list: The names of the functions
        class_list: The ids of the classes
        class_name_list: The names of the classes
        unique_id_list: The unique ids for each function call
        extra_text: Extra text to add to the time stamp, for example if a function is called in many loops 
        """
        self.numeric_lists = ["time_stamps", "positions_numeric", "function_list", "class_list", "unique_id_list"]
        self.string_lists = ["positions", "function_name_list", "class_name_list", "extra_text"]
        for list_name in self.numeric_lists:
            self.timing_data["lists"][list_name] = []
            self.timing_data["arrays"][list_name] = np.array([])
        for list_name in self.string_lists:
            self.timing_data["lists"][list_name] = []
        if class_name is not None:
            self.timing_data["class_name"] = class_name
            self.timing_data["class_names"].append(class_name)
        self.timing_data["is_array_updated"] = True
        self.timing_data["is_df_updated"] = False
        self.color_list = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

        self.dt = 1 # The time step to use for the gaps, we are not interested in tiny gaps
        self.__df = None  # A dataframe with the timing data. Should only be accessed using the get_all_timing_df() function

    def set_dt(self, dt):
        self.dt = dt

    def get_counter(self):
        return self.timing_data["counter"]

    def get_function_names(self):
        return self.timing_data["function_names"]
    
    def get_class_names(self):
        return self.timing_data["class_names"]

    def get_function_name_list(self):
        return self.timing_data["lists"]["function_name_list"]
    
    def get_class_name_list(self):
        return self.timing_data["lists"]["class_name_list"]
    
    def get_positions(self):
        return self.timing_data["lists"]["positions"]
    
    def get_extra_text(self):
        return self.timing_data["lists"]["extra_text"]
    
    def get_time_stamps(self, array=False):
        if array:
            self.__update_arrays()
            return self.timing_data["arrays"]["time_stamps"]
        else:
            return self.timing_data["lists"]["time_stamps"]
    
    def get_positions_numeric(self, array=False):
        if array:
            self.__update_arrays()
            return self.timing_data["arrays"]["positions_numeric"]
        else:
            return self.timing_data["lists"]["positions_numeric"]       

    def get_function_list(self, array=False):
        if array:
            self.__update_arrays()
            return self.timing_data["arrays"]["function_list"]
        else:
            return self.timing_data["lists"]["function_list"]

    def get_class_list(self, array=False):
        if array:
            self.__update_arrays()
            return self.timing_data["arrays"]["class_list"]
        else:
            return self.timing_data["lists"]["class_list"]   

    def get_unique_id_list(self, array=False):
        if array:
            self.__update_arrays()
            return self.timing_data["arrays"]["unique_id_list"]
        else:
            return self.timing_data["lists"]["unique_id_list"]

    def get_unique_ids(self, array=False):
        if array:
            self.__update_arrays()
            return np.arange(self.timing_data["lowest_unused_id"])
        else:
            return [i for i in range(self.timing_data["lowest_unused_id"])]
        
    def __get_inds_uniques_ids(self, unique_ids):
        """
        Get the indices of the unique ids.
        """
        if type(unique_ids) == np.int64 or type(unique_ids) == int:
            unique_ids = [unique_ids]
        unique_ids_list = self.get_unique_id_list(array=True)
        inds = []
        for i in unique_ids:
            idx = np.where(unique_ids_list == i)[0]
            if len(idx) == 0:
                continue
            inds.append(idx)
        return inds

    
    def get_all_timing_df_alt(self):
        """
        Get all the timing data in a pandas dataframe.
        """
        # TODO: Optimize this function, it is a bit slow
        # With 50000 entries it takes about 18 min
        # The smarter way to do this is to predefine the length of the df
        # this we can do because we know the number of unique ids
        # Then we just append the data to the df by itterating over 
        # all the arrays 
        unique_ids = self.get_unique_ids(array=True)
        self.__update_arrays()
        #if self.timing_data["is_df_updated"]:
        #    return self.__df
        c_names = []
        c_ids = []
        f_names = []
        f_ids = []
        start_times = []
        end_times = []
        intermittent_times = []
        extra_texts = []
        for i in unique_ids:
            inds = self.__get_inds_uniques_ids(i)[0]
            #self.print_data_inds(inds)
            if len(inds) == 0:
                continue
            start_times.append(self.timing_data["arrays"]["time_stamps"][inds[0]])
            end_times.append(self.timing_data["arrays"]["time_stamps"][inds[-1]])
            if len(inds) > 2:
                intermittent_times_ui = []
                for j in range(1, len(inds) - 1):
                    if self.timing_data["lists"]["positions_numeric"][inds[j]] == 0:
                        intermittent_times_ui.append(self.timing_data["arrays"]["time_stamps"][inds[j]])
    
                intermittent_times.append(intermittent_times_ui)
            else:
                intermittent_times.append(None)
            extra_texts.append(self.timing_data["lists"]["extra_text"][inds[0]])
            f_ids.append(self.timing_data["arrays"]["function_list"][inds[0]])
            c_ids.append(self.timing_data["arrays"]["class_list"][inds[0]])
            f_names.append(self.timing_data["lists"]["function_name_list"][inds[0]])
            c_names.append(self.timing_data["lists"]["class_name_list"][inds[0]])
        data = {
            "unique_id": unique_ids,
            "start_time": start_times,
            "end_time": end_times,
            "intermittent_time": intermittent_times,
            "extra_text": extra_texts,
            "function_id": f_ids,
            "class_id": c_ids,
            "function_name": f_names,
            "class_name": c_names, 
            "duration": np.array(end_times) - np.array(start_times),
        }
        self.__df = pd.DataFrame(data)
        self.timing_data["is_df_updated"] = True
        return pd.DataFrame(data)
    

    def get_all_timing_df(self):
        """
        Get all the timing data in a pandas dataframe.
        """
        self.__update_arrays()
        if self.timing_data["is_df_updated"]:
            return self.__df
        unique_ids = self.get_unique_ids(array=True)
        c_names = [None for i in range(len(unique_ids))]
        c_ids = [None for i in range(len(unique_ids))]
        f_names = [None for i in range(len(unique_ids))]
        f_ids = [None for i in range(len(unique_ids))]
        start_times = [None for i in range(len(unique_ids))]
        end_times = [None for i in range(len(unique_ids))]
        intermittent_times = [None for i in range(len(unique_ids))]
        extra_texts = [None for i in range(len(unique_ids))]

        # Load the arrays
        class_names_list = self.get_class_name_list()
        class_id_list = self.get_class_list(array=True)
        function_names_list = self.get_function_name_list()
        function_id_list = self.get_function_list(array=True)
        time_stamps_list = self.get_time_stamps(array=True)
        unique_ids_list = self.get_unique_id_list(array=True)
        positions_list = self.get_positions()
        positions_numeric_list = self.get_positions_numeric(array=True)
        extra_text_list = self.get_extra_text()

        for i in range(len(class_names_list)):
            j = unique_ids_list[i]
            if c_names[j] is None:
                c_names[j] = class_names_list[i]
            if c_ids[j] is None:
                c_ids[j] = class_id_list[i]
            if f_names[j] is None:
                f_names[j] = function_names_list[i]
            if f_ids[j] is None:
                f_ids[j] = function_id_list[i]
            if positions_list[i] == "start":
                start_times[j] = time_stamps_list[i]
            elif positions_list[i] == "end":
                end_times[j] = time_stamps_list[i]
            elif positions_list[i] == "intermittent":
                if intermittent_times[j] is None:
                    intermittent_times[j] = []
                intermittent_times[j].append(time_stamps_list[i])
            if extra_text_list[j] is not None:
                extra_texts[j] = extra_text_list[i]

        durations = []
        for i in range(len(unique_ids)):
            if start_times[i] is None or end_times[i] is None:
                durations.append(None)
            else:
                durations.append(end_times[i] - start_times[i])
        data = {
            "unique_id": unique_ids,
            "start_time": start_times,
            "end_time": end_times,
            "intermittent_time": intermittent_times,
            "extra_text": extra_texts,
            "function_id": f_ids,
            "class_id": c_ids,
            "function_name": f_names,
            "class_name": c_names, 
            "duration": durations,
        }
        self.__df = pd.DataFrame(data)
        self.timing_data["is_df_updated"] = True
        return pd.DataFrame(data)



        
    def __get_class_id(self, class_name, add_if_not_found=True) -> int:
        if class_name in self.timing_data["class_names"]:
            return self.timing_data["class_names"].index(class_name)
        else:
            if add_if_not_found:
                self.timing_data["class_names"].append(class_name)
                return len(self.timing_data["class_names"]) - 1
            return -1

    def __get_function_id(self, function_name: str, add_if_not_found=True) -> int:
        """
        Get the id of a function by its name.
        """
        function_names = self.timing_data["function_names"]
        if function_name in function_names:
            return function_names.index(function_name)
        else:
            if add_if_not_found:
                self.timing_data["function_names"].append(function_name)
                return len(self.timing_data["function_names"]) - 1
            return -1
        
    def __get_class_name(self, class_id: int) -> str:
        if class_id < len(self.timing_data["class_names"]):
            return self.timing_data["class_names"][class_id]
    
        else:
            return None
    
    def __get_function_name(self, function_id: int) -> str:
        if function_id < len(self.timing_data["function_names"]):
            return self.timing_data["function_names"][function_id]
        else:
            return None

    def __update_arrays(self):
        if self.timing_data["is_array_updated"]:
            return
        for list_name in self.numeric_lists:
            self.timing_data["arrays"][list_name] = np.array(self.timing_data["lists"][list_name])
        


    def __get_start_inds_class(self, class_name=None):
        """
        Get the start time for a function call with a unique id.
        """
        class_id = self.__get_class_id(class_name)
        if class_id is None:
            return []
        else:
            start_times_array = np.array(self.start_times)
            positions_array = np.array(self.positions_numeric)
            start_inds = np.where((positions_array == 1) & (start_times_array == class_id))[0]
            return start_inds
        
    def __get_start_times_class(self, class_name=None):
        """
        Get the start time for a function call with a unique id.
        """
        start_inds = self.__get_start_inds_class(class_name)
        if len(start_inds) == 0:
            return []
        else:
            start_times_array = np.array(self.time_stamps)
            return start_times_array[start_inds]
        
    def get_function_name_by_index(self, idx):
        """
        The index is the position in the function_name_list
        """
        if idx < len(self.timing_data["lists"]["function_name_list"]):
            return self.timing_data["lists"]["function_name_list"][idx]
        else:
            return None


    def get_data(self, data_types, **kwargs):
        """
        Get the data for a function call with a unique id.
        """
        data = {}
        if isinstance(data_types, str):
            data_types = [data_types]
        for data_type in data_types:
            if data_type in self.timing_data["lists"].keys():
                data[data_type] = self.timing_data["lists"][data_type]
            elif data_type in self.timing_data["arrays"].keys():
                data[data_type] = self.timing_data["arrays"][data_type]
            else:
                raise ValueError(f"Data type {data_type} not found")

    def __get_new_unique_id(self) -> int: 
        """
        Get a new unique id for the function call.
        This is not used yet in the code, but it might be useful in the future.
        """
        new_id = self.timing_data["lowest_unused_id"]
        self.timing_data["lowest_unused_id"] += 1
        return new_id
    
    def __get_unique_ids_func_class(self, function_name: str, class_name=None):
        """
        Get the unique ids for a function call with a unique id.
        """
        function_id = self.__get_function_id(function_name)
        class_id = self.__get_class_id(class_name)
        if function_id is None:
            return []
        else:
            functions_array = np.array(self.function_list)
            class_array = np.array(self.class_list)
            unique_ids_array = np.array(self.timing_data["unique_id_list"])
            positions_array = np.array(self.positions_numeric)
            start_inds = np.where((positions_array == 1) & (functions_array == function_id) & (class_array == class_id))[0]
            if len(start_inds) == 0:
                return []
            else:
                return start_inds, unique_ids_array[start_inds]
                


    def start_time(self, function_name: str, **kwargs) -> int:
        """
        Start the timer for a function.
        """
        
        class_name = self.timing_data.get("class_name", None)   # If the class name is not given, use the class name of the Timing object
        class_name = kwargs.get("class_name", class_name)
        extra_text = kwargs.get("extra_text", "")
        time_stamp = kwargs.get("time_stamp", time.time())
        # Get a unique id for the function call, not used yet in the code
        unique_id = self.__get_new_unique_id()
        self.__add_time(time_stamp, function_name, unique_id=unique_id, position="start", class_name=class_name, extra_text=extra_text)

        if kwargs.get("return_time_stamp", False):
            return unique_id, time_stamp
        return unique_id

    def end_time_id(self, unique_id: int, **kwargs):
        """
        This ends the time for a function with a unique id.
        This might be more useful when the functions are called in parallel
        
        Args:
            unique_id: The unique id for the function call, this is generated using start_time()
            kwargs: Extra arguments to pass to the function
        """

        # Get the index of the function call
        idx = self.timing_data["lists"]["unique_id_list"].index(unique_id)
        function_name = self.timing_data["lists"]["function_name_list"][idx]
        class_name = self.timing_data["lists"]["class_name_list"][idx]
        extra_text = kwargs.get("extra_text", "")
        time_stamp = kwargs.get("time_stamp", time.time())
        self.__add_time(time_stamp, function_name, unique_id=unique_id, position="end", class_name=class_name, extra_text=extra_text)

        if kwargs.get("return_time_stamp", False):
            return time_stamp

    def intermittent_time_id(self, unique_id, **kwargs):
        """
        Add some intermittent time for a function with a unique id.
        """
        idx = self.timing_data["lists"]["unique_id_list"].index(unique_id)
        function_name = self.timing_data["lists"]["function_name_list"][idx]
        class_name = self.timing_data["lists"]["class_name_list"][idx]
        extra_text = kwargs.get("extra_text", "")
        time_stamp = kwargs.get("time_stamp", time.time())
        self.__add_time(time_stamp, function_name, unique_id, position="intermittent", class_name=class_name, extra_text=extra_text)
        if kwargs.get("return_time_stamp", False):
            return time_stamp


    def __add_time(self, time_stamp, function_name, unique_id ,position="start", class_name=None, extra_text=""):
        # Getting the class id and function id
        function_id = self.__get_function_id(function_name, add_if_not_found=True)
        class_id = self.__get_class_id(class_name, add_if_not_found=True)

        self.timing_data["lists"]["positions"].append(position)

        if position == "start":
            self.timing_data["lists"]["positions_numeric"].append(1)
        elif position == "end":
            self.timing_data["lists"]["positions_numeric"].append(-1)
        else:
            self.timing_data["lists"]["positions_numeric"].append(0)

        self.timing_data["lists"]["time_stamps"].append(time_stamp) # The time stamps for when something happened
        self.timing_data["lists"]["function_list"].append(function_id) # The function ids
        self.timing_data["lists"]["function_name_list"].append(function_name)
        self.timing_data["lists"]["unique_id_list"].append(unique_id) # The unique ids for the function calls
        self.timing_data["lists"]["class_list"].append(class_id) # The class ids    
        self.timing_data["lists"]["class_name_list"].append(class_name)
        self.timing_data["lists"]["extra_text"].append(extra_text) # Extra text to add to the time stamp, for example if a function is called in many loops
        self.timing_data["counter"] += 1
        self.timing_data["is_array_updated"] = False
        self.timing_data["is_df_updated"] = False


    
    def print_timing_for_function(self, function_name, class_name=None):
        """
        Print the timing for a function.
        """
        function_id = self.__get_function_id(function_name)
        class_id = self.__get_class_id(class_name)
        if function_id is None:
            print(f"Function {function_name} not found")
            return
        else:
            print(f"Timing for function {function_name}:")
            for i in range(len(self.timing_data["lists"]["time_stamps"])):
                if self.timing_data["function_list"] == function_id:
                    if self.positions_numeric[i] == 1:
                        print(f"\tStart time: {self.time_stamps[i]}")
                    elif self.positions_numeric[i] == -1:
                        print(f"\tEnd time: {self.time_stamps[i]}")
                    else:
                        print(f"\tIntermittent time: {self.time_stamps[i]}")

    def print_data_inds(self, inds):
        """
        Print the data for a function call with a unique id.
        """
        for list in self.timing_data["lists"].keys():
            print(f"{list}: ", end=" ")
            for i in inds:
                print(f"{self.timing_data['lists'][list][i]}", end=" ")
            print()
        
    def get_total_time_for_function(self, function_name):
        # Get the total time for a function
        function_id = self.__get_function_id(function_name)
        if function_id is None:
            return 0
        else:
            time_stamps = self.get_time_stamps(array=True)
            positions_numeric = self.get_positions_numeric(array=True)
            unique_ids = self.get_unique_id_list(array=True)
            functions_array = self.get_function_list(array=True)

            idx = np.where(functions_array == function_id)[0]
            if len(idx) == 0:
                return 0
            time_stamps = time_stamps[idx]
            positions_numeric = positions_numeric[idx]
            unique_ids = unique_ids[idx]
        
            total_time = 0
            for unique_id in unique_ids:
                idx = np.where(unique_ids == unique_id)[0]
                if len(idx) == 0:
                    continue
                start_time = time_stamps[idx[0]]    
                end_time = time_stamps[idx[-1]]
                total_time += end_time - start_time
                if positions_numeric[idx[0]] != 1:
                    print(f"Warning: start time for function {function_name} is not 1")
                if positions_numeric[idx[-1]] != -1:
                    print(f"Warning: end time for function {function_name} is not -1")
            return total_time



    def get_average_time_for_function(self, function_name):
        # Get the average time for a function
        function_id = self.__get_function_id(function_name)
        if function_id is None:
            return 0
        else:
            total_time = 0
            n_calls = 0
            idx = np.where(self.function_list == function_id)[0]
            #starting

    
    @staticmethod
    def time_now_str():
        """
        Get the current time in a string format.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def time_stamp_to_str(time_stamp):
        pass

        
    def get_total_time_for_class(self, class_name):
        # Get the total time for a class
        class_id = self.__get_class_id(class_name)
        if class_id is None:
            return 0
        else:
            total_time = 0

            for i in range(len(self.time_stamps)):
                if self.class_list[i] == class_id:
                    if self.positions_numeric[i] == 1:
                        start_time = self.time_stamps[i]
                    elif self.positions_numeric[i] == -1:
                        end_time = self.time_stamps[i]
                        total_time += end_time - start_time
            return total_time
        
    def get_timing_chunks(self, function_name, class_name=None):
        """
        Get the timing chunks for a function.
        """
        function_id = self.__get_function_id(function_name)
        class_id = self.__get_class_id(class_name)
        if function_id is None:
            return []
        else:
            chunks = []
            pass
    
    def get_n_calls_for_function(self, function_name):
        """
        Get the number of calls for a function.
        """
        function_id = self.__get_function_id(function_name)
        if function_id is None:
            return 0
        else:
            functions_array = self.get_function_list(array=True)
            positions_array = self.get_positions_numeric(array=True)
            start_inds = np.where((positions_array == 1) & (functions_array == function_id))[0]
            return len(start_inds)

    def print_total_time_all_functions(self):
        """
        Print the total time for all functions.
        """
        print("Total time for all functions:")
        timing_df = self.get_all_timing_df()
        for class_name in timing_df["class_name"].unique():
            class_df = timing_df[timing_df["class_name"] == class_name]
            total_time = class_df["duration"].sum()
            print(f"Class: {class_name}")
            for function_name in class_df["function_name"].unique():
                function_df = class_df[class_df["function_name"] == function_name]
                total_time = function_df["duration"].sum()
                print(f"\t {function_name} \t: {total_time:.2f} s")


    def print_total_calls_all_functions(self):
        """
        Print the total number of calls for all functions.
        """
        print("Total number of calls for all functions:")
        for function_name in self.function_names:
            n_calls = self.get_n_calls_for_function(function_name)
            print(f"\t {function_name} \t: {n_calls:.2f} calls")
        print(f"Total number of calls for all functions: {self.counter:.2f} calls")


    def merge_timing_data(self, other_timing):
        """
        Merge the timing data from another Timing object into this one.
        """
        other_timing_df = other_timing.get_all_timing_df()
        for i, row in other_timing_df.iterrows():
            u_id = self.start_time(row["function_name"], class_name=row["class_name"], extra_text=row["extra_text"], time_stamp=row["start_time"])
            if row["intermittent_time"] is not None:
                for j in range(len(row["intermittent_time"])):
                    if row["intermittent_time"][j] is not None:
                        self.intermittent_time_id(u_id, time_stamp=row["intermittent_time"][j])
            self.end_time_id(u_id, time_stamp=row["end_time"])
   

    def save_timing_data(self, file_name):
        """
        Save the timing data to a file.
        """
        with open(file_name, "wb") as f:
            # Save the timing data to a file
            pickle.dump(self.timing_data, f)



    def load_timing_data(self, file_name):
        """
        Load the timing data from a file.
        """
        with open(file_name, "rb") as f:
            # Load the timing data from a file
            self.timing_data = pickle.load(f)
            self.timing_data["is_array_updated"] = False
            self.timing_data["is_df_updated"] = False
            self.__update_arrays()
            self.__df = None
            self.timing_data["is_df_updated"] = False


    def is_anything_happening(self, time_stamp):
        """
        Check if anything is happening at a given time stamp.
        This is useful for checking if the simulation is running or if it is stuck somewhere.
        """
        df = self.get_all_timing_df()
        starts = df["start_time"].values
        ends = df["end_time"].values
        if time_stamp < np.min(starts) or time_stamp > np.max(ends):
            return False
        for start, end in zip(starts, ends):
            if start <= time_stamp <= end:
                return True
        return False

    def get_gaps(self, dt=0.1):
        """
        Get the gaps in the timing data.
        This is where nothing happens
        This could be because it is not in use, but it could also be something unexpected that takes a long time
        dt: The time step to use for the gaps, we are not interested in tiny gaps
        that is for example i do not care if there is some gap lasting 0.1 seconds
        """
        dt = self.dt
        df = self.get_all_timing_df()
        timeline = np.arange(df["start_time"].min(), df["end_time"].max(), dt)
        starts = []
        ends = []
        for i in range(len(timeline) - 1):
            if self.is_anything_happening(timeline[i]) == False:
                start = timeline[i]
                start_ind = i
                for j in range(i, len(timeline)):
                    if self.is_anything_happening(timeline[j]) == True:

                        end = timeline[j]
                        end_ind = j
                        break
                if end_ind - start_ind < 2:
                    continue
                starts.append(start)
                ends.append(end)
        return starts, ends



    def plot_timing_data(self, **kwargs):
        """
        Plot the timing data. based on the pandas dataframe
        """
        df = self.get_all_timing_df()
        fig, (line_ax, tot_time_ax, avg_time_ax) = plt.subplots(1,3, figsize=(10, 5), sharey=True, width_ratios=[2, 1, 1])

        plot_hights = {}
        current_hight = 0
        y_axis_labels = []
        y_axis_labels_hights = []

        longest_function_name = 0
        for function_name in df["function_name"].unique():
            if len(function_name) > longest_function_name:
                longest_function_name = len(function_name)
        longest_class_name = 0
        for class_name in df["class_name"].unique():
            if class_name is None:
                continue
            if len(class_name) > longest_class_name:
                longest_class_name = len(class_name)       
        classes_ids = df["class_id"].unique()
        for class_id in classes_ids:
            j = 0
            plot_hights[class_id] = {}
            plot_hights[class_id]["hight"] = current_hight
            functions = df[df["class_id"] == class_id]["function_id"].unique()
            for function_id in functions:
                plot_hights[class_id][function_id] = current_hight
                function_name = self.__get_function_name(function_id)
                class_name = self.__get_class_name(class_id)
                if j == 0 and class_name is not None:
                    #label_str = "{:<longest_class_name} {:>longest_function_name}".format(class_name, function_name)
                    label_str = class_name + " " * (longest_class_name - len(class_name)) + " " * (longest_function_name + 1) + function_name
                    label_str = class_name + " " * (longest_class_name - len(class_name)) + " "  + " " * (longest_function_name - len(function_name)) + function_name
                    y_axis_labels.append(label_str)
                else:
                    #label_str = "{class_name:<longest_class_name} {function_name:>longest_function_name}".format(class_name="", function_name=function_name,longest_class_name=longest_class_name,longest_function_name=longest_function_name)
                    label_str = "" + " " * (longest_class_name - len("")) + " "  + " " * (longest_function_name - len(function_name)) + function_name
                    #label_str = " " * (longest_class_name + 1) + function_name + " " * (longest_function_name - len(function_name))
                    y_axis_labels.append(label_str)
                y_axis_labels_hights.append(current_hight)
                current_hight += 1
                j += 1
            current_hight += 1
        y_axis_labels_hights.append(current_hight-1)
        y_axis_labels.append(f"gaps (>{self.dt:.2f} s)")
     

        for i in range(len(df)):
            start_time = df["start_time"][i]
            end_time = df["end_time"][i]
            class_name = df["class_name"][i]
            function_name = df["function_name"][i]
            class_id = df["class_id"][i]
            function_id = df["function_id"][i]

            plot_hight = plot_hights[class_id][function_id]
            plot_color = self.color_list[class_id % len(self.color_list)]
            line_ax.fill_between([start_time, end_time], [plot_hight, plot_hight],[plot_hight+1, plot_hight+1], alpha=0.4, color=plot_color)

            if intermittent_time := df["intermittent_time"][i]:
                for j in range(len(intermittent_time)):

                    line_ax.plot([intermittent_time[j], intermittent_time[j]], [plot_hight, plot_hight + 1], color="black", alpha=1)
            # Add the function name to the y axis of the plot
            #ax.text((start_time + end_time) / 2, current_hight-1, function_name)

        # Add the functions on the y-axis
        # Change the tics on the y-axis to be the function names
        tick_hights = np.array(y_axis_labels_hights) + 0.5

        for ax in (line_ax, tot_time_ax, avg_time_ax):
            ax.set_yticks(list(tick_hights))
            ax.set_ylim(0, current_hight + 2)
      
        line_ax.set_yticks(list(tick_hights))
        line_ax.set_yticklabels(list(y_axis_labels))
        line_ax.set_xlabel("Time stamps")
        # Flip the y axis
        line_ax.invert_yaxis()
        #line_ax.set_ylim(0, current_hight)
        line_ax.set_ylabel("Function")

        # Plot the total time for each function
        for key in plot_hights.keys():
            for function_id in plot_hights[key].keys():
                if function_id == "hight":
                    continue
                total_time = df[df["function_id"] == function_id]["duration"].sum()
                n_calls = df[df["function_id"] == function_id]["unique_id"].nunique()
                plot_hight = plot_hights[key][function_id]
                plot_color = self.color_list[key % len(self.color_list)]
                tot_time_ax.barh(plot_hight+0.5, total_time, color=plot_color)
                text = "calls" if n_calls > 1 else "call"
                tot_time_ax.text(total_time, plot_hight+0.2, f"{n_calls} {text}")
        tot_time_ax.set_yticks(list(tick_hights))
        tot_time_ax.set_xlabel("Total time (s)")
        # Flip the y axis
        tot_time_ax.invert_yaxis()
        tot_time_ax.set_ylim(0, current_hight)

        # Plot the average time for each function
        for key in plot_hights.keys():
            for function_id in plot_hights[key].keys():
                if function_id == "hight":
                    continue
                avg_time = df[df["function_id"] == function_id]["duration"].mean()
                max_time = df[df["function_id"] == function_id]["duration"].max()
                plot_hight = plot_hights[key][function_id]
                plot_color = self.color_list[key % len(self.color_list)]
                # Box plot with the time pr function
                avg_time_ax.barh(plot_hight+0.5, avg_time, color=plot_color)
                text = "s max"
                avg_time_ax.text(avg_time, plot_hight+0.2, f"{max_time:.2f} {text}")
        avg_time_ax.set_yticks(list(tick_hights))
        avg_time_ax.set_xlabel("Average time (s)")
        # Flip the y axis
        avg_time_ax.invert_yaxis()
        avg_time_ax.set_ylim(0, current_hight)


        # Plotting the gaps
        gaps = self.get_gaps(dt=0.01)
        for i in range(len(gaps[0])):
            if len(gaps[0]) == 0:
                break
            start = gaps[0][i]
            end = gaps[1][i]
            line_ax.fill_between([start, end], [current_hight, current_hight],[current_hight-1, current_hight-1], alpha=0.4, color="black")

        fig.tight_layout()

        if "save_path" in kwargs.keys():
            save_path = kwargs["save_path"]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plot_name = kwargs.get("plot_name", "timing_plot")
            plt.savefig(os.path.join(save_path, f"{plot_name}.png"), dpi=300)
            plt.close()
        else:
            plt.show()




if __name__ == "__main__":


    from utility_funcs import *
    wdir = get_project_root()
    plot_path = os.path.join(wdir, "figures", "tests", "Timing")
    timing = Timing()

    # Example usage
    classes = ["A", "B", "C", None]
    functions = [["A_f1", "Af_2", "Af_3", "AB_f"], ["B_f1", "Bf_2", "Bf_3", "AB_f"], ["C_f1", "Cf_2", "Cf_3"], ["None_f1", "None_f2", "None_f3"]]

    t_start = time.time()
    for i in range(len(classes)):
        class_name = classes[i]
        for j in range(len(functions[i])):
            n_function_calls = np.random.randint(1, 5)
            for n in range(n_function_calls):
                start, end = np.sort(np.random.randint(10, 1000, 2))
                f_id = timing.start_time(functions[i][j], class_name=class_name, time_stamp=t_start + start)
                if np.random.rand() > 0.8:
                    timing.intermittent_time_id(f_id, time_stamp=t_start + np.random.randint(start, end))
                if np.random.rand() > 0.8:
                    timing.intermittent_time_id(f_id, time_stamp=t_start + np.random.randint(start, end))
                timing.end_time_id(f_id, time_stamp=t_start + end)

    # Print the timing data

    print("Timing data:")
    for key in timing.timing_data.keys():
        if key in ["lists", "arrays"]:
            for list_key in timing.timing_data[key].keys():
                print(key, list_key, timing.timing_data[key][list_key])
        else:
            print(key, timing.timing_data[key])

    pdf = timing.get_all_timing_df()
    print(pdf)
    pdf_alt = timing.get_all_timing_df_alt()
    print(pdf_alt)

    # Save the timing data to a file
    pdf.to_csv(os.path.join(plot_path, "timing_data.csv"), index=False)
    pdf_alt.to_csv(os.path.join(plot_path, "timing_data_alt.csv"), index=False)
    #timing.plot_timing_data()
    timing.plot_timing_data(save_path=plot_path, plot_name="simulated_timings_test")


    
    ##### 
    class TestTiming:
        def __init__(self):
            self.timing = Timing("TestTiming")
        


        def function_1(self):
            t_id = self.timing.start_time("function_1")
            time.sleep(np.random.uniform(0.1, 0.5))
            self.timing.end_time_id(t_id)
        
        def function_2(self):
            t_id = self.timing.start_time("function_2")
            time.sleep(np.random.uniform(0.1, 0.5))
            self.timing.intermittent_time_id(t_id)
            time.sleep(np.random.uniform(0.1, 0.5))
            self.timing.end_time_id(t_id)
        
        def function_3(self):
            t_id = self.timing.start_time("function_3")
            time.sleep(np.random.uniform(0.1, 0.5))
            self.function_1()
            self.timing.end_time_id(t_id)


        def function_4(self):
            t_id = self.timing.start_time("function_4")
            time.sleep(np.random.uniform(0.1, 0.5))
            t_2 = self.timing.start_time("function_other", class_name="Other")
            time.sleep(np.random.uniform(0.1, 0.5))
            self.timing.end_time_id(t_2) 
            time.sleep(np.random.uniform(0.1, 0.5))
            self.timing.end_time_id(t_id)

    def do_nothing():
        time.sleep(np.random.uniform(0.2, 0.5))

    ######################################
    ## Test and example usage
    ######################################
    test_timing = TestTiming()
    function_call = np.random.randint(1, 6, 20)
    for i in range(20):
        if function_call[i] == 1:
            test_timing.function_1()
        elif function_call[i] == 2:
            test_timing.function_2()
        elif function_call[i] == 3:
            test_timing.function_3()
        elif function_call[i] == 4:
            test_timing.function_4()
        elif function_call[i] == 5:
            do_nothing()
    print(test_timing.timing.get_all_timing_df())
    test_timing.timing.plot_timing_data(save_path=plot_path, plot_name="before_merge")

    print("Gaps in the timing data:")
    print(test_timing.timing.get_gaps(dt=0.1))

    timing_alt = Timing()
    timing_alt.merge_timing_data(test_timing.timing)
    timing_alt.plot_timing_data(save_path=plot_path, plot_name="after_merge") # This should be the same as the one above

    test_timing2 = TestTiming()
    function_call = np.random.randint(2, 6, 20)
    for i in range(20):
        if function_call[i] == 1:
            test_timing2.function_1()
        elif function_call[i] == 2:
            test_timing2.function_2()
        elif function_call[i] == 3:
            test_timing2.function_3()
        elif function_call[i] == 4:
            test_timing2.function_4()
        elif function_call[i] == 5:
            do_nothing()
    print(test_timing2.timing.get_all_timing_df())
    test_timing2.timing.plot_timing_data(save_path=plot_path, plot_name="before_merge_2")

    # Merge the timing data
    test_timing.timing.merge_timing_data(test_timing2.timing)
    print(test_timing.timing.get_all_timing_df())
    test_timing.timing.plot_timing_data(save_path=plot_path, plot_name="after_merge_2")


    # Save the timing data to a file
    timing.save_timing_data(os.path.join(plot_path, "timing_data.pkl"))
    # Load the timing data from a file
    timing_loaded = Timing()
    timing_loaded.load_timing_data(os.path.join(plot_path, "timing_data.pkl"))
    timing_loaded.plot_timing_data(save_path=plot_path, plot_name="loaded_timing_data")

    print(timing_loaded.get_gaps(dt=0.1))



    # Test if there is a significat overhead 
    print("Testing overhead time and if it is increasing")
    n = 10000
    t_sleep = 0.001
    print(f"Testing overhead time for {n} iterations")
    print(f"This will take around {n * (t_sleep * 1.1) * 2 / 60} minutes")
    t_control = []
    t_timing = []
    timing = Timing()
    for i in range(n):
        if i % (n // 10) == 0:
            print(f"Iteration {i} of {n}")
        t1_start = time.time()
        time.sleep(t_sleep)
        t1_end = time.time()

        t2_start = time.time()
        t_id = timing.start_time("test_function")
        time.sleep(t_sleep)
        timing.end_time_id(t_id)
        t2_end = time.time()
        t_control.append(t1_end - t1_start)
        t_timing.append(t2_end - t2_start)
    t_control = np.array(t_control)
    t_timing = np.array(t_timing)
    error = np.abs(t_control - t_timing)
    mean_error = np.mean(error)

    rolling_mean = np.convolve(error, np.ones(100)/100, mode='valid')
    # Fit a linar regression
    x = np.arange(len(error))
    coeffs = np.polyfit(x, error, 1)
    linear_fit = np.polyval(coeffs, x)
    plt.plot(error)
    plt.plot(linear_fit, color="green")
    plt.axhline(np.mean(error))
    plt.xlabel("Iteration")
    plt.ylabel("Extra time (s) pr iteration")
    plt.plot(rolling_mean, color="red")
    plt.title(f"Mean error: {mean_error:.4f} seconds \n Line y= {coeffs[0]:.6f}x + {coeffs[1]:.6f}")
    plt.savefig(plot_path + "/timing_overhead.png")
    plt.close()


    # Then we can check the time to get the df
    t1_start = time.time()
    timing.get_all_timing_df()
    t1_end = time.time()
    print(f"Time to get the df: {t1_end - t1_start:.6f} seconds")

    t1_start = time.time()
    df = timing.get_all_timing_df()
    t1_end = time.time()
    print(f"Time to get the df second time: {t1_end - t1_start:.4f} seconds")

    # The alternative way to get the df
    t1_start = time.time()
    df_alt = timing.get_all_timing_df_alt()
    t1_end = time.time()
    print(f"Time to get the df alt: {t1_end - t1_start:.4f} seconds")

    # Compare the two dataframes
    print("Comparing the two dataframes")
    print("Are the two dataframes equal: ", df.equals(df_alt))


    
