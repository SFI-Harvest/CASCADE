from scipy.spatial import distance_matrix
import numpy as np
from utilis.Logger import Logger
from utilis.Timing import Timing


class Covariance:
    """
    Covariance class to calculate covariance between two lists of numbers.
    """

    def __init__(self, covariance_params: dict,
                 logger_kwargs: dict = {}):
        """
        Initialize the Covariance class with two lists of numbers.

        Args:
            covariance_params (dict): Dictionary containing the covariance parameters.
                - phi_xy: The range parameter for the xy covariance function.
                - phi_temporal: The range parameter for the temporal covariance function.
                - sigma: The standard deviation of the noise.
                - temporal_covariance_type: The type of covariance function to use for the temporal dimension.
                - xy_covariance_type: The type of covariance function to use for the xy dimension.
            logger_kwargs (dict): Dictionary containing the logger parameters.
        """
        self.timing = Timing("Covariance")
        self.logger = Logger("Covariance", **logger_kwargs)

        # Implemented parameters
        self.cov_func_inmplemented = ["matern_3_2", "matern_5_2", "exponential", "gaussian"]
        self.spaces_included = ["temporal", "spatial"]


        self.model_parameters = covariance_params
        self.__check_model_parameters()
        self.logger.log_info("Covariance model initialized")

    

    def __check_model_parameters(self):
        """
        Check the model parameters
        """
        if "phi_xy" not in self.model_parameters:
            self.logger.log_warning("phi_xy not in model parameters")
        if "phi_temporal" not in self.model_parameters:
            self.logger.log_warning("phi_temporal not in model parameters")
        if "sigma" not in self.model_parameters:
            self.logger.log_warning("sigma not in model parameters")
        if "temporal_covariance_type" not in self.model_parameters:
            self.logger.log_warning("temporal_covariance_type not in model parameters")
        if "xy_covariance_type" not in self.model_parameters:
            self.logger.log_warning("xy_covariance_type not in model parameters")


    def set_print_to_console(self, print_to_console):
        """
        Set the print to console parameter for the logger.

        Args:
            print_to_console (bool): Whether to print to console or not.
        """
        self.logger.log_info(f"Setting Print to console to {print_to_console}")
        self.logger.print_to_console = print_to_console
        self.logger.log_info(f"Print to console set to {print_to_console}")

    def set_sigma(self, sigma):
        self.model_parameters["sigma"] = sigma
        self.logger.log_info(f"Sigma set to {sigma}")


    def set_phi_xy(self, phi_xy):
        self.model_parameters["phi_xy"] = phi_xy
        self.logger.log_info(f"Phi xy set to {phi_xy}")


    def set_phi_temporal(self, phi_temporal):
        self.model_parameters["phi_temporal"] = phi_temporal
        self.logger.log_info(f"Phi temporal set to {phi_temporal}")


    def set_temporal_covariance_type(self, covariance_type):
        if covariance_type not in self.cov_func_inmplemented:
            raise ValueError(f"Covariance type must be one of {self.cov_func_inmplemented}, but got {covariance_type}")
        self.model_parameters["temporal_covariance_type"] = covariance_type
        self.logger.log_info(f"Temporal covariance type set to {covariance_type}")


    def set_xy_covariance_type(self, covariance_type):
        if covariance_type not in self.cov_func_inmplemented:
            raise ValueError(f"Covariance type must be one of {self.cov_func_inmplemented}, but got {covariance_type}")
        self.model_parameters["xy_covariance_type"] = covariance_type
        self.logger.log_info(f"XY covariance type set to {covariance_type}")


    def set_phi_from_h_and_c(self, h, c, space):

        if space not in self.spaces_included:
            raise ValueError(f"space must be either {self.spaces_included}, but got {space}")
        
        if space == "temporal":
            phi = self.get_phi_from_range(h, c, self.model_parameters["temporal_covariance_type"])
            self.set_phi_temporal(phi)
            return
        if space == "spatial":
            phi = self.get_phi_from_range(h, c, self.model_parameters["xy_covariance_type"])
            self.set_phi_xy(phi)
    


    @staticmethod
    def distance_matrix_one_dimension(vec_1, vec_2) -> np.ndarray:
        return distance_matrix(vec_1.reshape(-1,1), vec_2.reshape(-1,1))
    
    @staticmethod
    def matern_3_2(h, params={"phi": 1}):
        """
        Matern covariance function with nu=3/2
        """
        phi = params["phi"]
        return (1 + np.sqrt(3) * h  / phi) * np.exp(-np.sqrt(3) * h / phi)
    
    @staticmethod
    def matern_5_2(h, params={"phi": 1}):
        """
        Matern covariance function with nu=5/2
        """
        phi = params["phi"]
        return (1 + np.sqrt(5) * h / phi + 5 * h**2 / (3 * phi**2)) * np.exp(-np.sqrt(5) * h / phi)
    
    @staticmethod
    def exponential(h, params={"phi": 1}):
        """
        Exponential covariance function
        """
        phi = params["phi"]
        return np.exp(-h / phi)
    
    @staticmethod
    def gaussian(h, params={"phi": 1}):
        """
        Gaussian covariance function
        """
        phi = params["phi"]
        return np.exp(-h**2 / (2 * phi**2))
    
    @staticmethod
    def phi_to_beta(phi, cov_type = "matern_3_2"):
        """
        Convert beta to phi
        """
        if cov_type == "exponential":
            return 1 / phi
        elif cov_type == "gaussian":
            return 1 / (2 * phi**2)
        elif cov_type == "matern_3_2":
            return np.sqrt(3) / phi
        elif cov_type == "matern_5_2":
            return np.sqrt(5) / phi
        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")
        
    @staticmethod
    def beta_to_phi(beta, cov_type = "matern_3_2"):
        """
        Convert phi to beta
        """
        if cov_type == "exponential":
            return 1 / beta
        elif cov_type == "gaussian":
            return 1 / np.sqrt(2 * beta)
        elif cov_type == "matern_3_2":
            return np.sqrt(3) / beta
        elif cov_type == "matern_5_2":
            return np.sqrt(5) / beta
        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")
        
    @staticmethod
    def dC_dbeta(h, beta, cov_type = "matern_3_2"):
        """
        Derivative of the covariance function with respect to beta
        """
        if cov_type == "exponential":
            return -h * np.exp(-h * beta)
        elif cov_type == "gaussian":
            return -h**2 * np.exp(-h**2 * beta)
        elif cov_type == "matern_3_2":
            return - h**2 * beta * np.exp(-h * beta)
        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")



    
    def covariance_function(self, h, params, cov_type="matern_3_2"):
        """
        Covariance function for the model
        """
        if cov_type == "matern_3_2":
            return self.matern_3_2(h, params)
        elif cov_type == "matern_5_2":
            return self.matern_5_2(h, params)
        elif cov_type == "exponential":
            return self.exponential(h, params)
        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")
    

    def covariance_function_temporal(self, h):
        phi_temporal = self.model_parameters["phi_temporal"]
        covariance_type = self.model_parameters["temporal_covariance_type"]
        return self.covariance_function(h, {"phi": phi_temporal}, covariance_type)
    
    def covariance_function_yx(self, h):
        phi_xy = self.model_parameters["phi_xy"]    
        covariance_type = self.model_parameters["xy_covariance_type"]
        return self.covariance_function(h, {"phi": phi_xy}, covariance_type)


    def make_covariance_matrix(self, S: np.ndarray, T = np.empty((1))) -> np.ndarray:
        f_name = "make_covariance_matrix"
        tok_t, time_start = self.timing.start_time(f_name,return_time_stamp=True)
        
        sigma = self.model_parameters["sigma"]
        
        # Split the yx and z
        #S_z = S[:,2]
        S_yx = S[:,0:2]

        # This function makes the covariance matrix for the model
        #Dz_matrix = self.distance_matrix_one_dimension(S_z,S_z)
        Dyx_matrix = distance_matrix(S_yx,S_yx)
        Dt_matrix = self.distance_matrix_one_dimension(T,T)

        Sigma = self.covariance_function_yx(Dyx_matrix)
        Sigma = Sigma * self.covariance_function_temporal(Dt_matrix)
        Sigma = Sigma * sigma**2


        time_end = self.timing.end_time_id(tok_t, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time to make matix with size {len(S)} x {len(S)}: {time_end - time_start:.4f} seconds")
        return Sigma
      
    
    def make_covariance_matrix_2(self, S_1: np.ndarray, S_2: np.ndarray, T_1 = np.empty(1), T_2 = np.empty(1)) -> np.ndarray:
        f_name = "make_covariance_matrix_2"
        tok_t, time_start = self.timing.start_time(f_name,return_time_stamp=True)
        # TODO: able to add just one time point
        sigma = self.model_parameters["sigma"]

        #S_z_1 = S_1[:,2]
        S_yx_1 = S_1[:,0:2]
        #S_z_2 = S_2[:,2]
        S_yx_2 = S_2[:,0:2]

        Dyx_matrix = distance_matrix(S_yx_1,S_yx_2)
        #Dz_matrix = self.distance_matrix_one_dimension(S_z_1,S_z_2)
        Dt_matrix = self.distance_matrix_one_dimension(T_1,T_2) 
        #Sigma = self.covariance_function_z(Dz_matrix)
        Sigma = self.covariance_function_yx(Dyx_matrix)
        Sigma = Sigma * self.covariance_function_temporal(Dt_matrix)
        Sigma = Sigma * sigma**2 

        time_end = self.timing.end_time_id(tok_t, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time to make matrix with size {len(S_1)}x{len(S_2)} : {time_end - time_start:.4f} seconds")
        return Sigma

    
    def update_covariance_matrix(self, old_cov_matrix: np.ndarray, S_old: np.ndarray, S_new: np.ndarray, T_old: np.ndarray, T_new: np.ndarray) -> np.ndarray:
        # This function updates the
        # covariance matrix by adding new points

        f_name = "update_covariance_matrix"
        tok_t, time_start = self.timing.start_time(f_name,return_time_stamp=True)
        n = len(T_old)
        m = len(T_new)

        new_cov_matrix = np.zeros((n + m, n + m))
        new_cov_matrix[0:n, 0:n] = old_cov_matrix

        # Get the covariance matrix for the new points
        Sigma_cross = self.make_covariance_matrix_2(S_old, S_new, T_old, T_new)
        new_cov_matrix[n:(n+m),0:n] = Sigma_cross.T
        new_cov_matrix[0:n,n:(n+m)] = Sigma_cross
        new_cov_matrix[(n):(n+m),(n):(n+m)] = self.make_covariance_matrix(S_new, T_new)
        
        # Logging
        time_end = self.timing.end_time_id(tok_t, return_time_stamp=True)
        self.logger.log_info(f"[{f_name}] Time to update matrix, old size {n}x{n} updated size {n+m}x{n+m}: {time_end - time_start:.4f} seconds")
        return new_cov_matrix
    


    """
    fuctions for solving the problems 
    C(h ; phi) = c
    provided the covariance function C
    we want to provide x and h to get phi
    or phi and x to get h
    """


    def get_h_from_phi_and_c(self, phi, c, cov_type):
        """
        Solves the equation C(h ; phi) = c for h

        Args:
            phi (float): Phi parameter
            c (float): Constant
        want to set h such that c(h) = c
        """
        # TODO: Optimize the search range for h

        if cov_type=="exponential":
            return -phi * np.log(c)
        elif cov_type=="gaussian":
            return np.sqrt(-2 * phi**2 * np.log(c))
        elif cov_type in ["matern_3_2", "matern_5_2"]:
            exp_solution = self.get_h_from_phi_and_c(phi, c, "exponential")
            gaussian_solution = self.get_h_from_phi_and_c(phi, c, "gaussian")
            min_solution = min(exp_solution, gaussian_solution)
            max_solution = max(exp_solution, gaussian_solution)
            hh = np.linspace(min_solution, max_solution, 1000)
            if cov_type=="matern_3_2":
                c_values = self.matern_3_2(hh, {"phi": phi})
            elif cov_type=="matern_5_2":
                c_values = self.matern_5_2(hh, {"phi": phi})
            return self.__solve_decreasing_function(hh, c_values, c)
        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")
    

    def get_phi_from_range(self, h, co, cov_type):
        """
        solves the equation C(h ; phi) = c for phi

        Args:
            h (float): Range
            c (float): Constant
        want to set phi such that C(h;phi) = co
        """

        if cov_type=="exponential":
            return -h / np.log(co)
        elif cov_type=="gaussian":
            return np.sqrt(- h**2 /(2 * np.log(co)))
        elif cov_type in ["matern_3_2", "matern_5_2"]:
            exp_solution = -h / np.log(co)
            gaussian_solution = np.sqrt(-h**2 / (2 * np.log(co)))
            # We want to find the minimum of the two solutions
            # The solution will be between the two solutions (i think) TODO: check
            min_solution = min(exp_solution, gaussian_solution)
            max_solution = max(exp_solution, gaussian_solution)
            phi_values = np.linspace(min_solution, max_solution, 1000)
            if cov_type=="matern_3_2":
                c_values = self.matern_3_2(h, {"phi": phi_values})
            elif cov_type=="matern_5_2":
                c_values = self.matern_5_2(h, {"phi": phi_values})
            return self.__solve_decreasing_function(phi_values, c_values, co)
        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")


    def __solve_decreasing_function(self, x, y, c):
        # This solves the equation for what x is when y = c
        # y is a monetonically decreasing function
        dist = y-c
        abs_dist = np.abs(dist)
        if not np.any(dist < 0) or not np.any(dist > 0):
            self.logger.log_warning("No solution found")
        min_index = np.argmin(abs_dist)
        return x[min_index]


      
    


if __name__ == "__main__":
    from utilis.Timing import Timing
    from utilis.Logger import Logger
    from utilis.utility_funcs import *
    import numpy as np
    import os
    import time

    import matplotlib.pyplot as plt

    wdir = get_project_root()
    stor_path = os.path.join(wdir, "figures/tests/Covariance")

    if not os.path.exists(stor_path):
        os.makedirs(stor_path)
    

    logger_kwargs = {
        "log_file": os.path.join(stor_path, "Covariance.log"),
        "print_to_console": True,
        "overwrite_file": True,
    }

    covariance_params = {
        "phi_xy": 1.0,
        "phi_temporal": 1.0,
        "sigma": 1.0,
        "temporal_covariance_type": "matern_3_2",
        "xy_covariance_type": "matern_3_2",
    }
    cov = Covariance(covariance_params, logger_kwargs=logger_kwargs)


    cov_functions = {
        "matern_3_2": {
            "function": cov.matern_3_2,
            "params": {"phi": 1.0},
            "plot_color": "blue",
            "plot_label": "Matern 3/2",
        },
        "matern_5_2": {
            "function": cov.matern_5_2,
            "params": {"phi": 1.0},
            "plot_color": "green",
            "plot_label": "Matern 5/2",
        },
        "exponential": {
            "function": cov.exponential,
            "params": {"phi": 1.0},
            "plot_color": "red",
            "plot_label": "Exponential",
        },
        "gaussian": {
            "function": cov.gaussian,
            "params": {"phi": 1.0},
            "plot_color": "orange",
            "plot_label": "Gaussian",
        }}

    # Run some tests on the covariance matrix
    h = np.linspace(0, 6, 100)
    for cov_type, cov_func in cov_functions.items():
        plt.plot(h, cov_func["function"](h, cov_func["params"]), label=f"{cov_func['plot_label']}, phi={cov_func['params']['phi']}")
    plt.xlabel("h")
    plt.ylabel("C(h;phi)")
    plt.title("Covariance functions")
    plt.ylim(0, 1.1)
    plt.xlim(0, 6)
    plt.legend()
    plt.savefig(os.path.join(stor_path, "covariance_functions.png"))
    plt.close()


    # Test with C(h;phi) = c
    range = 4000
    c = 0.3
    hh = np.linspace(0, range * 1.5, 100)
    for cov_type, cov_func in cov_functions.items():
        phi = cov.get_phi_from_range(range, c, cov_type)
        cov_functions[cov_type]["params"]["phi"] = phi
        print(f"Covariance type: {cov_type}, phi: {phi:.2f}")
        C = cov_func["function"](hh, {"phi": phi})
        plt.plot(hh, C, label=f"{cov_type}, phi={phi:.2f}")
    plt.axhline(y=c, color="r", linestyle="--", label="C(h;phi)=c")
    plt.axvline(x=range, color="r", linestyle="--", label="h")
    plt.xlabel("h")
    plt.ylabel("C(h;phi)")
    plt.title(f"Covariance functions with C({range};phi)={c}")
    plt.ylim(0, 1.1)
    plt.xlim(0, range * 1.5)
    plt.legend()
    plt.savefig(os.path.join(stor_path, "covariance_functions_c.png"))
    plt.close()

    # Now we can see how the covariance functi
    c = 0.7
    hh = np.linspace(0, range * 2, 100)
    for cov_type, cov_func in cov_functions.items():
        phi = cov_functions[cov_type]["params"]["phi"]
        h = cov.get_h_from_phi_and_c(phi, c, cov_type)
        print(f"Covariance type: {cov_type}, h: {h:.2f}")
        C = cov_func["function"](hh, {"phi": phi})
        plt.plot(hh, C, label=f"{cov_type}, h={h:.2f}", color=cov_func["plot_color"])
        plt.plot([h, h], [0, c], color=cov_func["plot_color"])
    plt.axhline(y=c, color="r", linestyle="--", label="C(h;phi)=c")
    plt.xlim(0, range * 2)
    plt.ylim(0, 1.1)
    plt.xlabel("h")
    plt.ylabel("C(h;phi)")
    plt.title(f"Covariance functions with C(h;phi)={c}")
    plt.legend()
    plt.savefig(os.path.join(stor_path, "covariance_functions_h.png"))
    plt.close()


    # Now we can get the spatial covariance matrix
    Sx = np.linspace(0, 10, 100)
    Sy = np.linspace(0, 10, 100)
    S = np.array([Sx, Sy]).T
    T = np.repeat(0, len(Sx))
    Sigma = cov.make_covariance_matrix(S, T)
    plt.imshow(Sigma, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Covariance matrix with line from [0,0] to [10,10], T=0")
    plt.savefig(os.path.join(stor_path, "covariance_matrix.png"))
    plt.close()

    # Now with random points
    Sx = np.random.uniform(0, 10, 100)
    Sy = np.random.uniform(0, 10, 100)
    S = np.array([Sx, Sy]).T
    T = np.repeat(0, len(Sx))
    Sigma = cov.make_covariance_matrix(S, T)
    plt.imshow(Sigma, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Covariance matrix with random points, T=0")
    plt.savefig(os.path.join(stor_path, "covariance_matrix_random.png"))
    plt.close()

    # Test if making the cov and updating gives the same result
    n = np.random.randint(100, 200)
    m = np.random.randint(100, 200)
    Sx = np.random.uniform(0, 10, n+m)
    Sy = np.random.uniform(0, 10, n+m)
    S_small = np.array([Sx[:n], Sy[:n]]).T
    S_big = np.array([Sx, Sy]).T
    S_new = np.array([Sx[n:], Sy[n:]]).T
    T = np.random.uniform(0, 10, len(Sx))
    T_small = T[:n]
    T_big = T   
    T_new = T[n:]

    Cov_small = cov.make_covariance_matrix(S_small, T_small)
    Cov_big = cov.make_covariance_matrix(S_big, T_big)
    Cov_new = cov.make_covariance_matrix(S_new, T_new)
    Cov_updated = cov.update_covariance_matrix(Cov_small, S_small, S_new, T_small, T_new)
    # Check if the updated covariance matrix is equal to the new covariance matrix
    if np.allclose(Cov_updated, Cov_big):
        print("Covariance matrix updated correctly")
    else:
        print("Covariance matrix updated incorrectly")
        raise ValueError("Covariance matrix updated incorrectly") # Must be the same


    # Test the speed of the covariance matrix
    nn = 10**np.linspace(1, 4, 20)
    time_list = []
    for n in nn:
        n = int(n)
        Sx = np.random.uniform(0, 10, n)
        Sy = np.random.uniform(0, 10, n)
        S = np.array([Sx, Sy]).T
        T = np.random.uniform(0, 10, n)
        time_start = time.time()
        Sigma = cov.make_covariance_matrix(S, T)
        time_end = time.time()
        time_list.append(time_end - time_start)

    plt.plot(nn, time_list)
    plt.xlabel("Number of points")
    plt.ylabel("Time (s)")
    plt.title("Time to calculate covariance matrix")
    plt.xscale("log")
    plt.savefig(os.path.join(stor_path, "covariance_matrix_speed.png"))
    plt.close()


    # Test the relative speed of making from scratch and updating
    import math
    nn = 10**np.linspace(2, 3.5, 10)
    mm = 10**np.linspace(2, 3.5, 10)
    max_points = math.ceil(max(nn)) + math.ceil(max(mm))
    Sx = np.random.uniform(0, 10, max_points )
    Sy = np.random.uniform(0, 10, max_points )
    S = np.array([Sx, Sy]).T
    T = np.random.uniform(0, 10, max_points )

    timing_matrix = np.zeros((len(nn), len(mm), 4))

    for i, n in enumerate(nn):
        time_list_small = []
        time_list_big = []
        time_list_update = []
        n = int(n)
        for j, m in enumerate(mm):
            m = int(m)
            S_small = S[:n]
            S_big = S[:n+m]
            S_new = S[n:n+m]
            T_small = T[:n]
            T_big = T[:n+m]
            T_new = T[n:n+m]
            # Make the covariance matrix from scratch
            time_start = time.time()
            Sigma_small = cov.make_covariance_matrix(S_small, T_small)
            time_end = time.time()
            time_list_small.append(time_end - time_start)
            # Make the covariance matrix from scratch
            time_start = time.time()
            Sigma_big = cov.make_covariance_matrix(S_big, T_big)
            time_end = time.time()
            time_list_big.append(time_end - time_start)
            # Update the covariance matrix
            time_start = time.time()
            Sigma_updated = cov.update_covariance_matrix(Sigma_small, S_small, S_new, T_small, T_new)
            time_end = time.time()
            time_list_update.append(time_end - time_start)
        
        timing_matrix[i, :, 0] = np.array(time_list_small)
        timing_matrix[i, :, 1] = np.array(time_list_big)
        timing_matrix[i, :, 2] = np.array(time_list_update)
        timing_matrix[i, :, 3] = np.array(time_list_big) / np.array(time_list_update)


    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    vmin = np.min(timing_matrix[:, :, 1])
    vmax = np.max(timing_matrix[:, :, 1])
    ax[0].imshow(timing_matrix[:, :, 1], cmap="hot", interpolation="nearest", 
                extent=[mm[0], mm[-1], nn[-1], nn[0]], aspect="auto", vmin=vmin, vmax=vmax)
    ax[1].imshow(timing_matrix[:, :, 2], cmap="hot", interpolation="nearest",
                extent=[mm[0], mm[-1], nn[-1], nn[0]], aspect="auto", vmin=vmin, vmax=vmax)
    # Add colorbars
    cbar1 = plt.colorbar(ax[1].imshow(timing_matrix[:, :, 2], cmap="hot", interpolation="nearest", 
                extent=[mm[0], mm[-1], nn[-1], nn[0]], aspect="auto", vmin=vmin, vmax=vmax), ax=ax[1])
    cbar1.set_label("Time (s)")
    ax[0].set_title("Time to calculate covariance matrix from scratch")
    ax[1].set_title("Speedup of updating covariance matrix")
    ax[0].set_xlabel("Number of points in big matrix")
    ax[0].set_ylabel("Number of points in small matrix")
    plt.savefig(os.path.join(stor_path, "covariance_matrix_speedup.png"))
    plt.close()


    # For fun plot the timing data
    cov.timing.plot_timing_data(save_path=stor_path, plot_name="timing_data_covariance")


        

    

    










