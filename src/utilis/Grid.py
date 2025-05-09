import numpy as np
import math
from scipy.interpolate import griddata
from scipy import sparse
import time


class Grid:
    """
    Gridd class that handles grid operations.
    """

    """

    the basic idea is to make a grid that looks something like this:

    #######################
    # * * * * * * * * * * #
    # * * * * * * * * * * #
    # * * * * * * * * * * #
    # * * * * * * * * * * #
    # * * * * * * * * * * #
    # * * * * * * * * * * #
    #######################

    then we can use the assignment of the grid points to the grid points to get the closest point
    (xi, yi) -> grid_ind

    ############################################
    # (0,4)  (1,4)  (2,4)  (3,4)  (4,4)  (5,4) #
    # (0,3)  (1,3)  (2,3)  (3,3)  (4,3)  (5,3) #
    # (0,2)  (1,2)  (2,2)  (3,2)  (4,2)  (5,2) #
    # (0,1)  (1,1)  (2,1)  (3,1)  (4,1)  (5,1) #
    # (0,0)  (1,0)  (2,0)  (3,0)  (4,0)  (5,0) #
    ############################################

    to make this even easier we want to use the notion of a grid index
    gi = yi * n_x + xi

    ############################
    # (20) (21) (22) (23) (24) #
    # (15) (16) (17) (18) (19) #
    # (10) (11) (12) (13) (14) #
    # (5 ) (6 ) (7 ) (8 ) (9 ) #
    # (0 ) (1 ) (2 ) (3 ) (4 ) #
    ############################

    Vocabulary:
    - s: the point in space
    - S: the set of points in space
    - grid: the grid points, which is a 2D array of shape (n_grid_points, 2) same as S
    - grid_ind / gi : the grid inde
    - grid_point / gs : the grid point
    - x_ind, xi: the x index of the grid point 
    - y_ind, yi: the y index of the grid point
    - x_point, xs: the x point of the grid point
    - y_point, ys: the y point of the grid point
    - x_ax: the x axis of the grid
    - y_ax: the y axis of the grid

    """

    def __init__(self, grid_file=None, logger_kwargs={}):
        """
        Initialize the Gridd class with a grid file.
        """
        self.grid = None
        self.grid_file = grid_file
        self.__grid_data = {}

        #self.logger = Logger("Grid", **logger_kwargs)
        #self.logger.log_info("Grid initialized")
    
    def __len__(self):
        """
        Get the length of the grid.
        """
        if self.grid is None:
            raise ValueError("Grid not initialized.")
        return self.__grid_data["n"]
    
    def __getitem__(self, index):
        """
        Get the grid point for a given index.
        """
        if self.grid is None:
            raise ValueError("Grid not initialized.")
        if index < 0 or index >= len(self.grid):
            raise IndexError("Index out of range.")
        return self.grid[index]


    def make_regular_grid(self, xmin, xmax, ymin, ymax, dx, dy):
        """
        Create a regular grid.
        """
        x_ax = np.arange(xmin, xmax, dx)
        y_ax = np.arange(ymin, ymax, dy)

        self.make_regular_grid_from_axes(x_ax, y_ax)
     
    def make_regular_grid_from_axes(self, x_ax, y_ax):
        """
        Create a regular grid from x and y axes.

        # TODO: minimize the storage of the grid
        # One only needs to store the x and y axes
        """
        xx, yy = np.meshgrid(x_ax, y_ax)
        xxi , yyi = np.meshgrid(np.arange(len(x_ax)), np.arange(len(y_ax)))
        self.grid = np.column_stack((xx.flatten(), yy.flatten()))
        self.__grid_data["x_ax"] = x_ax
        self.__grid_data["y_ax"] = y_ax
        self.__grid_data["n_grid_points"] = len(x_ax) * len(y_ax)
        self.__grid_data["xx"] = xx
        self.__grid_data["yy"] = yy
        self.__grid_data["xxi"] = xxi.flatten()
        self.__grid_data["yyi"] = yyi.flatten()
        self.__grid_data["dx"] = x_ax[1] - x_ax[0]
        self.__grid_data["dy"] = y_ax[1] - y_ax[0]
        self.__grid_data["nx"] = len(x_ax)
        self.__grid_data["ny"] = len(y_ax)
        self.__grid_data["n"] = len(x_ax) * len(y_ax)
        self.__grid_data["xmin"] = x_ax[0]
        self.__grid_data["xmax"] = x_ax[-1]
        self.__grid_data["ymin"] = y_ax[0]
        self.__grid_data["ymax"] = y_ax[-1]
        self.__grid_data["grid"] = self.grid

    def make_regular_grid_inside_boundary(self, dxdy, boundary, origin=None, origin_place=None):
        """
        Create a regular grid inside a boundary.
        """
        # Get the boundary points

        bounding_box = boundary.get_bounding_box()
        x_min = bounding_box[0]
        x_max = bounding_box[2]
        y_min = bounding_box[1]
        y_max = bounding_box[3]

        if origin is not None:
            x_ax = np.arange(origin[0], x_max, dxdy)
            y_ax = np.arange(origin[1], y_max, dxdy)
        else:
            x_ax = np.arange(x_min, x_max, dxdy)
            y_ax = np.arange(y_min, y_max, dxdy)
        self.make_regular_grid_from_axes(x_ax, y_ax)
        
    def make_time_grid(self, t0, dt):
        """
        Create a time grid.
        No box is needed here
        """
        self.__grid_data["dt"] = dt
        self.__grid_data["t0"] = t0

    def time_ind_to_t(self, ind):
        """
        Convert time index to time.
        """
        t0 = self.__grid_data["t0"]
        dt = self.__grid_data["dt"]
        return t0 + ind * dt


    def get_time_grid_ind(self, grid_ind, time_ind):
        n_grid_points = self.__grid_data["n_grid_points"]
        return time_ind * n_grid_points + grid_ind
    
    def get_timestamp_above(self, t):
        """
        Get the timestamp above a given time t.
        """
        t0 = self.__grid_data["t0"]
        dt = self.__grid_data["dt"]
        if t < t0:
            return t0
        nt = math.ceil((t - t0) / dt)
        return t0 + nt * dt
    
    def get_timestamps_interval(self, t0, t1):
        """
        Get the timestamps between t0 and t1.
        """
        timestamps = []
        t0 = self.get_timestamp_above(t0)
        t_next = t0
        while t_next <= t1:
            timestamps.append(t_next)
            t_next += self.__grid_data["dt"]
        return timestamps


    def get_closest_time_ind_t(self, t):
        """
        Get the closest time grid index for a given time t.
        """
        t0 = self.__grid_data["t0"]
        dt = self.__grid_data["dt"]
        if t < t0:
            return 0
        nt = int((t - t0) / dt)
        return nt
    
    
    def get_closest_time_inds_T(self, T):
        """
        Get the closest time grid indices for a given set of times T.
        """
        closest_T_ind = np.array([self.get_closest_time_ind_t(t) for t in T], dtype=int)
        return closest_T_ind


    def get_closest_time_t(self, t):
        """
        Get the closest time grid index for a given time t.
        """
        t0 = self.__grid_data["t0"]
        dt = self.__grid_data["dt"]
        if t < t0:
            return t0
        nt = self.get_closest_time_ind_t(t)
        return t0 + nt * dt
    
    def get_closest_time_T(self, T):
        """
        Get the closest time grid index for a given set of times T.
        """
        closest_T = np.array([self.get_closest_time_t(t) for t in T])
        return closest_T
 
    def get_surrounding_time_inds_t(self, t):
        """
        Get the surrounding time grid indices for a given time t.
        """
        t0 = self.__grid_data["t0"]
        dt = self.__grid_data["dt"]
        if t < t0:
            return [t0]
        time_below = math.floor((t - t0) / dt)
        time_above = math.ceil((t - t0) / dt)
        return [time_below, time_above]
    
    def get_surrounding_time_inds_T(self, T):
        """
        Get the surrounding time grid indices for a given set of times T.
        """
        surrounding_T = [self.get_surrounding_times_t(t) for t in T]
        return surrounding_T
    
    def get_surrounding_times_t(self, t):
        """
        Get the surrounding time grid indices for a given time t.
        """
        t0 = self.__grid_data["t0"]
        dt = self.__grid_data["dt"]
        surrounding_inds = self.get_surrounding_time_inds_t(t)
        surrounding_times = [self.time_ind_to_t(ind) for ind in surrounding_inds]
        return surrounding_times
    
    def get_surrounding_times_T(self, T):
        """
        Get the surrounding time grid indices for a given set of times T.
        """
        surrounding_times = [self.get_surrounding_times_t(t) for t in T]
        return surrounding_times




    def get_x_ax(self) -> np.ndarray:
        return self.__grid_data["x_ax"]

    def get_y_ax(self) -> np.ndarray:
        return self.__grid_data["y_ax"]

    def get_grid_data_key(self, key):
        if key in self.__grid_data:
            return self.__grid_data[key]
        else:
            raise KeyError(f"Key '{key}' not found in grid data.")
        
    def get_gs_from_xi_yi(self, x_ind, y_ind):
        """
        Get the grid point for a given index.
        """
        x = self.__grid_data["x_ax"][x_ind]
        y = self.__grid_data["y_ax"][y_ind]
        return np.array([x, y])
    
    def get_grid_ind(self, x_ind, y_ind):
        """
        Get the grid indices for a given point.
        """
        grid_ind = int(y_ind * len(self.__grid_data["x_ax"]) + x_ind) 
        # self.logger.log_debug(f"[get_grid_ind] x_ind: {x_ind}, y_ind: {y_ind}, grid_ind: {grid_ind}")
        return grid_ind

    def get_gi_from_xi_yi(self, x_ind, y_ind):
        """
        Get the grid index form the x and y indices.
        """
        if hasattr(x_ind, "__len__"):
            x_ind = np.array(x_ind, dtype=int)
            y_ind = np.array(y_ind, dtype=int)
            Gi = x_ind + y_ind * self.__grid_data["nx"]
            return Gi
        else:
            return int(x_ind + y_ind * self.__grid_data["nx"])
    
    def get_xi_yi_from_gi(self, grid_ind):
        """
        Get the x and y indices for a given grid index.
        """
        if hasattr(grid_ind, "__len__"):
            grid_ind = np.array(grid_ind, dtype=int)
            x_ind = grid_ind % self.__grid_data["nx"]
            y_ind = grid_ind // self.__grid_data["nx"]
            return x_ind, y_ind
        else:
            return int(grid_ind % self.__grid_data["nx"]), int(grid_ind // self.__grid_data["nx"])

    
    def get_x_ind_y_ind(self, grid_ind):
        """
        Get the x and y indices for a given grid index.
        """
        n_x = len(self.__grid_data["x_ax"])
        y_ind = int(grid_ind / n_x)  # int() rounds down
        x_ind = int(grid_ind % n_x)
        return x_ind, y_ind
    
    def get_grid_point_inds(self, grid_inds):
        """
        Get the grid points for a given set of indices.
        """
        # self.logger.log_debug(f"[get_grid_point_inds] grid_inds: {grid_inds}")
        if type(grid_inds) == np.float64:
            self.logger.log_error(f"[get_grid_point_inds] grid_inds is a float64: {grid_inds}")
            raise ValueError("grid_inds is a float64, not an int.")
        if type(grid_inds) == int or type(grid_inds) == np.int64:
            grid_inds = [grid_inds]
        if len(grid_inds) == 1:
            return self.grid[grid_inds[0]]
        return self.grid[grid_inds]
    

    
    
    def get_surrounding_inds_s(self, s):
        """
        Get the surrounding indices for a given point s.
        """
        x = s[0]
        y = s[1]
        x_inds_above = np.where(self.__grid_data["x_ax"] >= x)[0]
        y_inds_above = np.where(self.__grid_data["y_ax"] >= y)[0]
        x_inds = []
        y_inds = []
        if x > self.__grid_data["x_ax"][-1]:
            x_inds.append(len(self.__grid_data["x_ax"]) - 1)
        elif x < self.__grid_data["x_ax"][0]:
            x_inds.append(0)
        else:
            x_inds.append(x_inds_above[0])
            x_inds.append(x_inds_above[0] - 1)
        if len(y_inds_above) == 0:
            y_inds.append(len(self.__grid_data["y_ax"]) - 1)
        elif y < self.__grid_data["y_ax"][0]:
            y_inds.append(0)
        else:
            y_inds.append(y_inds_above[0])
            y_inds.append(y_inds_above[0] - 1)
        inds = []
        for x_ind in x_inds:
            for y_ind in y_inds:
                inds.append([x_ind, y_ind])
        # Inds are in the form [x_ind, y_ind]
        # Convert to grid indices
        grid_inds = []
        for ind in inds:
            grid_ind = self.get_grid_ind(ind[0], ind[1])
            grid_inds.append(grid_ind)
        # np.array(inds, dtype=int)
        return np.array(grid_inds, dtype=int)

    def get_surrounding_gi_gi(self, grid_ind, d=1,
                               boundary_condition="",
                               status=None):

        """
        x - ind
        o - surrounding points
        * - grid points

        d=1
        #######################
        # * * * * * * o x o * #
        # * * o * * * * o * * #
        # * o x o * * * * * * #
        # * * o * * * * * * o #
        # * * * * * * * * o x #
        #######################

        d=2
        #######################
        # * * o * * o o x o o #
        # * o o o * * o o o * #
        # o o x o o * * o * * #
        # * o o o * * * * * o #
        # * * o * * * * * o o #
        # * * * * * * * o o x #
        #######################

        and so on...
        """
        if status is None:
            status = np.zeros(len(self.grid), dtype=int)
        # status[grid_ind] = 0 -> not discovered
        # status[grid_ind] = 1 -> closed
        # status[grid_ind] = 2 -> open
        # status[grid_ind] = 3 -> discovered
        inds = []
        surrounding_inds = []
        inds_temp = []
        
        # Getting the surrounding grid indices for the grid index
        inds_temp.append(self.get_gi_to_left(grid_ind, boundary_condition))
        inds_temp.append(self.get_gi_to_right(grid_ind, boundary_condition))
        inds_temp.append(self.get_gi_to_above(grid_ind, boundary_condition))
        inds_temp.append(self.get_gi_to_below(grid_ind, boundary_condition))
        # Note that some of these indices may be None

        for ind in inds_temp:
            if ind is not None: 
                inds.append(ind)
                surrounding_inds.append(ind)
                if status[ind] == 0:
                    if d > 1:
                        status[ind] = 2   # The surrounding point is open
                    else:
                        pass
                        #status[ind] = 1  # The surrounding point is closed


        # Close the grid index
        status[grid_ind] = 1

        
        # If d=1, we are done
        # If d=2, we need to get the surrounding points of the surrounding points
        # This is done recursively using a depth first search
        if d > 1:
            for ind in inds:
                surrounding_inds_temp = self.get_surrounding_gi_gi(ind, d - 1, boundary_condition, status)
                for ind_temp in surrounding_inds_temp:
                    if ind_temp not in inds and ind_temp not in surrounding_inds:
                        surrounding_inds.append(ind_temp)
                        if status[ind_temp] == 0:
                            status[ind_temp] = 2
        # Recursive is stupid 
        # We know the pattern like this 
        ########################
        #                           (xi  , yi+2)
        #              (xi-1, yi+1) (xi  , yi+1) (xi+1, yi+1)
        # (xi-2, yi  ) (xi-1, yi  ) (xi  , yi  ) (xi+1, yi  ) (xi+2, yi  )
        #              (xi-1, yi-1) (xi  , yi-1) (xi+1, yi-1)
        #                           (xi  , yi-2)
        #  ######################
        # Then we can just keep the points that are in the grid

        """
        xi, yi = self.get_x_ind_y_ind(grid_ind)
        surrounding_inds_alt = []
        sur_xi = []
        sur_yi = []
        for i in range(-d, d + 1):
            for j in range(-d, d + 1):
                if abs(i) + abs(j) <= d:
                    xi_temp = xi + i
                    yi_temp = yi + j
                    if xi_temp < 0 or xi_temp >= self.__grid_data["nx"] or yi_temp < 0 or yi_temp >= self.__grid_data["ny"]:
                        continue
                    sur_xi.append(xi + i)
                    sur_yi.append(yi + j)
                    grid_ind_temp = self.get_grid_ind(xi + i, yi + j)
                    surrounding_inds_alt.append(grid_ind_temp)
        
        # DEBUG
        for i in surrounding_inds_alt:
            if i not in surrounding_inds:
                print(f"i: {i} not in surrounding_inds")
        for i in surrounding_inds:
            if i not in surrounding_inds_alt:
                print(f"i: {i} not in surrounding_inds_alt")
        """

        
        return surrounding_inds
    
    def __is_gi_internal(self, grid_inds, d=1):
        """
        Check if the grid index is internal.

        d=2
        #######################

        """
        x_ind, y_ind = self.get_x_ind_y_ind(grid_ind)
        if x_ind >= 0 + d and x_ind < self.__grid_data["nx"] - d and y_ind >= 0 + d and y_ind < self.__grid_data["ny"] - d:
            return True
        else:
            return False



    def get_surronding_gi_matrix(self, d=1, boundary_condition="", diagonal=False, make_sparse=False):
        """
        Get the surrounding grid indices for a given grid index.
        TODO: exploit the fact that the grid is a regular grid a bit more
        this is not a big issue if d is small, but if d is large, this can be a problem
        when d=1 and the grid is 500x500 this will take 5s
        when d=2 and the grid is 500x500 this will take 9s
        when d=3 and the grid is 500x500 this will take 10s
        """
        if self.grid is None:
            raise ValueError("Grid not initialized.")
        if d < 1:
            raise ValueError("d must be greater than 0.")
        surronding_matrix = np.zeros((len(self.grid), len(self.grid)), dtype=int)
        #surronding_matrix_alt = np.zeros((len(self.grid), len(self.grid)), dtype=int)

        pattern_size = sum([i * 4 for i in range(1, d + 1)])
        basic_pattern = None
        # The basic pattern is a 1D array of size pattern_size
        # The basic pattern appears for the inner grid points

        t1 = time.time() # REMOVE
        for i in range(len(self.grid)):
            xi, yi = self.get_x_ind_y_ind(i)
            if xi >= 0+d and xi < self.__grid_data["nx"] - d and yi >= 0 + d and yi < self.__grid_data["ny"] - d:
                # The grid point is in the middle of the grid
                # The surrounding points are the same as the basic pattern
                if basic_pattern is None:
                    surronding_inds = np.array(self.get_surrounding_gi_gi(i, d, boundary_condition), dtype=int)
                    basic_pattern = surronding_inds - i
                surronding_matrix[i, basic_pattern + i] = 1
            else:
                surronding_inds = self.get_surrounding_gi_gi(i, d, boundary_condition)
                for j in range(len(surronding_inds)):
                    surronding_matrix[i, surronding_inds[j]] = 1

        #t2 = time.time() # REMOVE
        #for i in range(len(self.grid)):
        #    surronding_inds = self.get_surrounding_gi_gi(i, d, boundary_condition)
        #    for j in range(len(surronding_inds)):
        #        surronding_matrix[i, surronding_inds[j]] = 1
        #t3 = time.time() # REMOVE

        #print(f"Time taken to get the surrounding grid indices (alt): {t2 - t1:.2f} seconds")
        #print(f"Time taken to get the surrounding grid indices: {t3 - t2:.2f} seconds")

        # Set if the diagonal is considered      
        for i in range(len(self.grid)):
            if diagonal:
                surronding_matrix[i, i] = 1
        #        surronding_matrix_alt[i, i] = 1
            else:
                surronding_matrix[i, i] = 0
        #        surronding_matrix_alt[i, i] = 0

        # Check if the matrix is the same
        #if np.array_equal(surronding_matrix, surronding_matrix_alt):
        #    print("The two matrices are the same")
        #else:
        #    print("The two matrices are NOT the same")

    
        # This matrix will be very sparse, so we can use a sparse matrix
        if make_sparse:
            surronding_matrix = sparse.csr_matrix(surronding_matrix)
            
        return surronding_matrix

    

    def get_gi_to_left(self, grid_ind, boundary_condition=""):
        """
        Get the index to the left of a given index.
        """
        x_ind, y_ind = self.get_x_ind_y_ind(grid_ind)
        if x_ind == 0:
            if boundary_condition == "periodic":
                x_ind = len(self.__grid_data["x_ax"]) - 1
            elif boundary_condition == "reflective":
                x_ind = 1
            else:
                return None
            return self.get_grid_ind(x_ind, y_ind)
        else:
            return self.get_grid_ind(x_ind - 1, y_ind)
        

    def get_gi_to_right(self, grid_ind, boundary_condition=""):
        """
        Get the index to the right of a given index.
        """
        x_ind, y_ind = self.get_x_ind_y_ind(grid_ind)

        if x_ind == len(self.__grid_data["x_ax"]) - 1:
            if boundary_condition == "periodic":
                x_ind = 0
            elif boundary_condition == "reflective":
                x_ind = len(self.__grid_data["x_ax"]) - 2
            else:
                return None
            return self.get_grid_ind(x_ind, y_ind)
        else:
            return self.get_grid_ind(x_ind + 1, y_ind)
        

    def get_gi_to_above(self, grid_ind, boundary_condition=""):
        """
        Get the index above a given index.
        """
        x_ind, y_ind = self.get_x_ind_y_ind(grid_ind)
        if y_ind == len(self.__grid_data["y_ax"]) - 1:
            if boundary_condition == "periodic":
                y_ind = 0
            elif boundary_condition == "reflective":
                y_ind = len(self.__grid_data["y_ax"]) - 2
            else:
                return None
            return self.get_grid_ind(x_ind, y_ind)
        else:
            return self.get_grid_ind(x_ind, y_ind + 1)
        
        
    def get_gi_to_below(self, grid_ind, boundary_condition=""):
        """
        Get the index below a given index.
        """
        x_ind, y_ind = self.get_x_ind_y_ind(grid_ind)
        if y_ind == 0:
            if boundary_condition == "periodic":
                y_ind = len(self.__grid_data["y_ax"]) - 1
            elif boundary_condition == "reflective":
                y_ind = 1
            else:
                return None
            return self.get_grid_ind(x_ind, y_ind)
        else:
            return self.get_grid_ind(x_ind, y_ind - 1)


    def get_closest_ind_s(self, s):
        """
        Get the closest grid index for a given point s.
        """
        surrounding_inds = self.get_surrounding_inds_s(s)
        if len(surrounding_inds) == 0:
            raise ValueError("No surrounding points found.")
        if len(surrounding_inds) == 1:
            # self.logger.log_debug(f"[get_closest_ind_s] len(s_inds)==1 surrounding_inds: {surrounding_inds}") # REMOVE
            return surrounding_inds[0]
        surrounding_points = self.grid[surrounding_inds]
        dist = np.sqrt((surrounding_points[:, 0] - s[0]) ** 2 + (surrounding_points[:, 1] - s[1]) ** 2)
        ind = np.argmin(dist)
        # self.logger.log_info(f"s: {s}") # REMOVE
        # self.logger.log_debug(f"surrounding_inds: {surrounding_inds}")
        # self.logger.log_debug(f"ind: {ind}") # REMOVE
        # self.logger.log_debug(f"dist: {dist}")
        # self.logger.log_debug(f"surrounding_points: {surrounding_points}")
        # self.logger.log_debug(f"closest_ind: {ind}")
        # self.logger.log_debug(f"closest_point, surrounding_inds[ind]: {surrounding_points[ind]}")
        return surrounding_inds[ind]
    
    def get_closest_ind_S(self, S):
        """
        Get the closest grid indices for a given set of points S.
        """
        closest_inds = []
        for s in S:
            closest_inds.append(self.get_closest_ind_s(s))
        return np.array(closest_inds, dtype=int)
    
    
    def get_surrounding_inds_S(self, S):
        """
        Get the surrounding indices for a given point S.
        """
        inds = []
        for s in S:
            inds.append(self.get_surrounding_inds_s(s))
        return np.array(inds, dtype=int)

    
    def get_surrounding_points_s(self, s):
        """
        Get the surrounding points for a given point s.
        """
        surrounding_inds = self.get_surrounding_inds_s(s)
        if len(surrounding_inds) == 0:
            raise ValueError("No surrounding points found.")
        if len(surrounding_inds) == 1:
            return [self.grid[surrounding_inds[0]]]
        return self.grid[surrounding_inds]
    
    def get_surrounding_points_S(self, S):
        """
        Get the surrounding points for a given set of points S.
        """
        surrounding_points = []
        for s in S:
            surrounding_points.append(self.get_surrounding_points_s(s))
        return surrounding_points



    def get_closest_point_s(self, s):
        """
        Get the closest point for a given point s.
        """
        closest_ind = self.get_closest_ind_s(s)
        # self.logger.log_debug(f"[get_closest_point_s] closest_ind: {closest_ind}") # REMOVE
        return self.get_grid_point_inds(closest_ind)
        
 
    
    def get_closest_points_S(self, S):
        """
        Get the closest points for a given set of points S.
        """
        closest_points = []
        for s in S:
            # self.logger.log_debug(f"[get_closest_points_S] s: {s}") # REMOVE
            closest_points.append(self.get_closest_point_s(s))
        return np.array(closest_points)


    def get_random_grid_inds(self, n , replace=False):
        """
        Get n random grid indices.
        """
        if replace:
            return np.random.choice(len(self.grid), n, replace=True)
        else:
            return np.random.choice(len(self.grid), n, replace=False)
        

    def get_internal_gi(self, d=1):
        """
        Get the internal grid indices for a given grid index.
        """
        if self.grid is None:
            raise ValueError("Grid not initialized.")
        if d < 1:
            raise ValueError("d must be greater than 0.")
        internal_gi = []
        for i in range(len(self.grid)):
            xi, yi = self.get_x_ind_y_ind(i)
            if xi >= 0 + d and xi < self.__grid_data["nx"] - d and yi >= 0 + d and yi < self.__grid_data["ny"] - d:
                internal_gi.append(i)
        return np.array(internal_gi, dtype=int)
    

      
    
    def get_weights(self, dists):
        """
        Get the weights for a given point s.
        """
        weights = 1 / dists
        weights /= np.sum(weights)
        return weights
    
    
    def get_dist(self, S1, S2):
        """
        Get the distance between two lists of points S1 and S2.
        There are 4 cases:
        1. S1 and S2 are both 1D arrays meaning they are points
        2. S1 is a 1D array and S2 is a 2D array meaning S2 is a list of points
        3. S1 is a 2D array and S2 is a 1D array meaning S1 is a list of points
        4. S1 and S2 are both 2D arrays meaning they are lists of points
        """
        S1 = np.array(S1)
        S2 = np.array(S2)
        
        if len(S1.shape) == 1 and len(S2.shape) == 1:
            # Case 1
            return self.__get_dist_s_s(S1, S2)
        elif len(S1.shape) == 1 and len(S2.shape) == 2:
            # Case 2
            return self.__get_dist_s_S(S1, S2)
        elif len(S1.shape) == 2 and len(S2.shape) == 1:
            # Case 3
            return self.__get_dist_s_S(S2, S1)
        elif len(S1.shape) == 2 and len(S2.shape) == 2:
            # Case 4
            return self.__get_dist_S_S(S1, S2)
    

    
    def __get_dist_s_S(self, s, S):
        """
        Get the distance between a point s and a list of points S.
        """
        return np.sqrt((s[0] - S[:, 0]) ** 2 + (s[1] - S[:, 1]) ** 2)

    def __get_dist_S_S(self, S1, S2):
        """
        Get the distance between two lists of points S1 and S2.
        """
        return np.sqrt((S1[:, 0] - S2[:, 0]) ** 2 + (S1[:, 1] - S2[:, 1]) ** 2)
    
    def __get_dist_s_s(self, s1, s2):
        """
        Get the distance between two points s1 and s2.
        """
        return np.sqrt((s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2)
    

    def get_grid_assignment(self, S):
        """
        Get the grid assignment for a given set of points S.
        """
        closses_inds = self.get_closest_ind_S(S)
        closest_points = self.get_grid_point_inds(closses_inds)
        dist_to_closest = self.get_dist(S, self.get_grid_point_inds(closses_inds))
        return closses_inds, closest_points, dist_to_closest
    
    def get_time_assignment(self, T):
        """
        Get the time assignment for a given set of times T.
        """
        closses_time_inds = self.get_closest_time_inds_T(T)
        closest_times = self.get_closest_time_T(T)
        dist_to_closest = np.abs(closest_times.flatten() - T.flatten())
        return closses_time_inds, closest_times, dist_to_closest
    
    def get_assignment(self, S, T):
        """
        Get the grid assignment for a given set of points S and times T.
        """
        closses_inds, closest_points, dist_to_closest = self.get_grid_assignment(S)
        closses_time_inds, closest_times, dist_to_closest_time = self.get_time_assignment(T)
        grid_time_inds = self.get_time_grid_ind(closses_inds, closses_time_inds)
        assignments = {
            "grid_inds": closses_inds,
            "grid_points": closest_points,
            "dist_to_closest": dist_to_closest,
            "time_inds": closses_time_inds,
            "time_points": closest_times,
            "dist_to_closest_time": dist_to_closest_time,
            "grid_time_inds": grid_time_inds
        }
        return assignments
    
    def flatten_y(self, y):
        """
        Flatten the y array to a 1D array.
        """
        if len(y.shape) == 2:
            return y.flatten()
        elif len(y.shape) == 1:
            return y
        else:
            raise ValueError("y must be a 1D or 2D array.")
    
    def reshape_y(self, y):
        """
        Reshape the y array to the grid shape.
        """
        if len(y.shape) == 1:
            return y.reshape((self.__grid_data["ny"], self.__grid_data["nx"]))
        elif len(y.shape) == 2:
            return y
        else:
            raise ValueError("y must be a 1D or 2D array.")

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Logger import Logger
    plot_path = "figures/tests/Grid/"

    logging_kwargs = {
        "print_to_console": True,
        "log_file": "figures/tests/Grid/grid_test.log",
        "overwrite_file": True
    }

    grid = Grid(logger_kwargs=logging_kwargs)
    grid.make_regular_grid(0, 3, 0, 6, 1, 2)
    print("grid", grid.grid )
    print("grid[0]", grid.grid[0])
    print("grid[[2,3]]", grid.grid[[2,3]])
    print("grid[[3,5,7,8]]", grid.grid[[3,5,7,8]])  

    # Test grid inds is correct
    random_ind = grid.get_random_grid_inds(1)[0]
    print("random_ind", random_ind)
    x_ind, y_ind = grid.get_x_ind_y_ind(random_ind)
    print("x_ind", x_ind)
    print("y_ind", y_ind)
    grid_ind = grid.get_grid_ind(x_ind, y_ind)
    print("grid_ind", grid_ind)
    if grid_ind != random_ind:
        print("grid_ind != random_ind")
        raise ValueError("grid_ind != random_ind")



    grid = Grid(logger_kwargs=logging_kwargs)
    grid.make_regular_grid(0, 10, 0, 10, 1.4, 4.4)
    surrounding_inds = grid.get_surrounding_inds_s([5.4, 4.5])
    print("surrounding_inds", surrounding_inds)
    surrounding_points = grid.grid[surrounding_inds]
    print("surrounding_points", surrounding_points)
    closest_ind = grid.get_closest_ind_s([5.4, 4.5])
    closest_points = grid.grid[closest_ind]
    plt.plot(grid.grid[:, 0], grid.grid[:, 1], "o", color="blue", markersize=1, label="Grid")
    for i in range(len(surrounding_points)):
        plt.plot(surrounding_points[i, 0], surrounding_points[i, 1], "ro")
        plt.plot([5.4, surrounding_points[i, 0]], [4.5, surrounding_points[i, 1]], color="green")
    plt.plot([5.4, closest_points[0]], [4.5, closest_points[1]], color="red", label="Closest point")
    plt.scatter(closest_points[0], closest_points[1], color="red", s=10)
    plt.title("Plotting surrounding points")
    plt.savefig(plot_path + "grid_test_surrounding_points.png")
    plt.close()

    # Write the grid_ind to the plot
    xxi = grid.get_grid_data_key("xxi")
    yyi = grid.get_grid_data_key("yyi")
    ggi = grid.get_gi_from_xi_yi(xxi, yyi)
    for i in range(len(grid.grid)):
        plt.text(xxi[i], yyi[i]+0.03, str(ggi[i]), fontsize=8)
        plt.text(xxi[i], yyi[i]-0.04, f"({xxi[i]},{yyi[i]})", fontsize=8)
    plt.plot(xxi, yyi, "o", color="blue", markersize=1, label="Grid")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.title("Grid indices")
    plt.savefig(plot_path + "grid_test_grid_inds.png", dpi=300)
    plt.close()


    grid = Grid(logger_kwargs=logging_kwargs)
    grid.make_regular_grid_from_axes(np.arange(0, 10, 1.4), np.arange(0, 10, 4.4))
    surrounding_inds = grid.get_surrounding_inds_s([5.4, 4.5])
    print("surrounding_inds", surrounding_inds)
    surrounding_points = grid.grid[surrounding_inds]
    print("surrounding_points", surrounding_points)
    closest_ind = grid.get_closest_ind_s([5.4, 4.5])
    closest_points = grid.grid[closest_ind]
    plt.plot(grid.grid[:, 0], grid.grid[:, 1], "o", color="blue", markersize=1, label="Grid")
    for i in range(len(surrounding_points)):
        plt.plot(surrounding_points[i, 0], surrounding_points[i, 1], "ro")
        plt.plot([5.4, surrounding_points[i, 0]], [4.5, surrounding_points[i, 1]], color="green")
    plt.plot([5.4, closest_points[0]], [4.5, closest_points[1]], color="red", label="Closest point")
    plt.scatter(closest_points[0], closest_points[1], color="red", s=10)
    plt.title("Plotting surrounding points")
    plt.savefig(plot_path + "grid_test_surrounding_points_alt.png")
    plt.close()

    # Some test
    x_lim = [np.random.uniform(0, 100), np.random.uniform(200, 300)]
    y_lim = [np.random.uniform(0, 100), np.random.uniform(200, 300)]
    dx = np.random.uniform(10, 20)
    dy = np.random.uniform(10, 20)
    grid.make_regular_grid(x_lim[0], x_lim[1], y_lim[0], y_lim[1], dx, dy)
    n = 10
    plt.plot(grid.grid[:, 0], grid.grid[:, 1], "o", color="blue", markersize=1)
    for i in range(n):
        x = np.random.uniform(x_lim[0] - 20, x_lim[1]+ 20)
        y = np.random.uniform(y_lim[0] - 20, y_lim[1] + 20)
        s = (x, y)
        surrounding_points = grid.get_surrounding_points_s(s)
        closest_ind = grid.get_closest_ind_s(s)
        print(f"closest_ind: {closest_ind}")
        closest_points = grid.get_closest_point_s(s)
        plt.plot(x, y, "ro")
        for sp in surrounding_points:
            plt.plot([x, sp[0]], [y, sp[1]], color="black", linestyle="--")
        plt.plot([x, closest_points[0]], [y, closest_points[1]], color="green", linestyle="--")
        plt.scatter(closest_points[0], closest_points[1], color="red", s=4)

    plt.title("Plotting random points and their closest points")
    plt.savefig(plot_path + "grid_test_random_points.png")
    plt.close()


    # Line test 
    x_start = np.random.uniform(x_lim[0] - 20, x_lim[1] + 20)
    y_start = np.random.uniform(y_lim[0] - 20, y_lim[1] + 20)
    x_end = np.random.uniform(x_lim[0] - 20, x_lim[1] + 20)
    y_end = np.random.uniform(y_lim[0] - 20, y_lim[1] + 20)
    line_x = np.linspace(x_start, x_end, 100)
    line_y = np.linspace(y_start, y_end, 100)
    line_S = np.column_stack((line_x, line_y))
    #grid.logger.log_debug(f"[outside] line_S: {line_S}") # REMOVE
    closest_points = grid.get_closest_points_S(line_S)
    print(f"closest points shape", closest_points.shape)
    plt.plot(grid.grid[:, 0], grid.grid[:, 1], "o", color="blue", markersize=1)
    for i in range(len(line_S)):
        plt.plot(line_S[i, 0], line_S[i, 1], "ro")
        plt.plot([line_S[i, 0], closest_points[i, 0]], [line_S[i, 1], closest_points[i, 1]], color="green")
    plt.scatter(closest_points[:, 0], closest_points[:, 1], color="red", s=10)
    plt.title("Plotting line and their closest points")
    plt.savefig(plot_path + "grid_test_line.png")
    plt.close()


    # Using griddata
    grid_inds_close = grid.get_closest_ind_S(line_S)
    grid_points = grid.grid[grid_inds_close]
    plt.plot(grid.grid[:, 0], grid.grid[:, 1], "o", color="blue", markersize=1)
    for i in range(len(line_S)):
        plt.plot(line_S[i, 0], line_S[i, 1], "ro")
        plt.plot([line_S[i, 0], grid_points[i, 0]], [line_S[i, 1], grid_points[i, 1]], color="green")
    plt.scatter(grid_points[:, 0], grid_points[:, 1], color="red", s=10)
    plt.title("Plotting line and their closest grid points")
    plt.savefig(plot_path + "grid_test_line_griddata.png")
    plt.close()



    # Check some grid inds
    for _ in range(10):
        x_ind = np.random.randint(0, len(grid.get_x_ax()))
        y_ind = np.random.randint(0, len(grid.get_y_ax()))
        print(f"x_ind: {x_ind}, y_ind: {y_ind}")
        ind = grid.get_grid_ind(x_ind, y_ind)
        print(f"grid_ind: {ind}")
        print(f"grid[ind]: {grid.grid[ind]}")
        xg, yg = grid.get_gs_from_xi_yi(x_ind, y_ind)
        print(f"xg: {xg}, yg: {yg}")

    

    # Test neigbour points
    random_grid_inds = grid.get_random_grid_inds(3)
    plt.plot(grid.grid[:, 0], grid.grid[:, 1], "o", color="blue", markersize=1)
    for i in random_grid_inds:
        random_point = grid.grid[i] 
        surrounding_inds = grid.get_surrounding_gi_gi(i, d=1)
        surrounding_points = grid.grid[surrounding_inds]
        plt.plot(random_point[0], random_point[1], "ro")
        plt.scatter(surrounding_points[:, 0], surrounding_points[:, 1], color="green", s=10)

    plt.title(f"Plotting random points and their surrounding points d={1}")
    plt.savefig(plot_path + "grid_test_random_points_surrounding_d1.png")
    plt.close()

    # Test neigbour points
    for d in [1,2,3]:
        random_grid_inds = grid.get_random_grid_inds(2)
        plt.plot(grid.grid[:, 0], grid.grid[:, 1], "o", color="blue", markersize=1)
        for i in random_grid_inds:
            random_point = grid.grid[i] 
            surrounding_inds = grid.get_surrounding_gi_gi(i, d=d)
            surrounding_points = grid.grid[surrounding_inds]
            plt.plot(random_point[0], random_point[1], "ro")
            plt.scatter(surrounding_points[:, 0], surrounding_points[:, 1], color="green", s=10)
        plt.title(f"Plotting random points and their surrounding points d={d}")
        plt.savefig(plot_path + f"grid_test_random_points_surrounding_d{d}.png")
        plt.close()

    # Surronding matrix
    grid2 = Grid(logger_kwargs=logging_kwargs)
    grid2.make_regular_grid(0, 4, 0, 4, 1, 1)
    surronding_matrix = grid2.get_surronding_gi_matrix(d=1, boundary_condition="", diagonal=True)
    surronding_matrix_d3 = grid2.get_surronding_gi_matrix(d=3, boundary_condition="", diagonal=False)
    surronding_matrix_d2 = grid2.get_surronding_gi_matrix(d=2, boundary_condition="")
    fig, ax = plt.subplots(2,2, figsize=(10, 5))

    ax[0,0].imshow(surronding_matrix, cmap="hot", interpolation="nearest")
    ax[0,0].set_title(f"Surrounding matrix d={1}, diagonal=True")
    ax[1,0].imshow(surronding_matrix_d2, cmap="hot", interpolation="nearest")
    ax[1,0].set_title(f"Surrounding matrix d={2}, diagonal=False")
    ax[0,1].scatter(grid2.grid[:, 0], grid2.grid[:, 1], c="blue", s=1)
    for i in range(len(grid2.grid)):
        ax[0,1].text(grid2.grid[i, 0], grid2.grid[i, 1], str(i), fontsize=8)
    ax[0,1].set_title(f"Grid points")
    ax[1,1].imshow(surronding_matrix_d3, cmap="hot", interpolation="nearest")
    ax[1,1].set_title(f"Surrounding matrix d={3}, diagonal=False")

    plt.savefig(plot_path + "grid_test_surrounding_matrix.png")
    plt.close()





    # Test some interpolation
    def function(x, y):
        return np.sin(x / 20)* 5  + np.cos(y/ 30) * 4
    closest_inds = grid.get_closest_ind_S(line_S)
    closest_points = grid.grid[closest_inds]
    dists = grid.get_dist(line_S, closest_points)
    k = 0
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(line_S[:, 0], line_S[:, 1], c="red", label="Line")
    ax[0].scatter(grid.grid[:, 0], grid.grid[:, 1], c="blue", label="Grid")
    ax[0].scatter(closest_points[:, 0], closest_points[:, 1], c="green", label="Closest points")
    ax[0].scatter(line_S[:, 0], line_S[:, 1], c=np.arange(len(line_S)), label="Line points", cmap="jet")
    for i in range(len(line_S)):
        ax[0].plot([line_S[i, 0], closest_points[i, 0]], [line_S[i, 1], closest_points[i, 1]], color="green")
    for i in np.unique(closest_inds):
        j = np.where(closest_inds == i)[0]
        dist_j = dists[j]
        f_j = function(line_S[j, 0], line_S[j, 1])
        weights = grid.get_weights(dist_j)
        f_i = np.sum(weights * f_j)

        ax[1].scatter(j, f_j, c=j, vmin=0, vmax = len(closest_inds), label="Function values", cmap="jet") 

        ax[1].plot([j[0], j[-1]], [f_i, f_i], color="green", label="Function")
        print(sum(weights))
        ax[2].scatter(np.repeat(len(weights),len(weights)), weights, c=j, vmin=0, vmax=len(closest_inds), label="Weights", cmap="jet")
    plt.savefig(plot_path + "grid_test_line_weights.png")
    plt.close()
    
    
    x_ax = np.linspace(x_lim[0]-20, x_lim[1]+20, 100)
    y_ax = np.linspace(y_lim[0]-20, y_lim[1]+20, 100)
    xx, yy = np.meshgrid(x_ax, y_ax)
    zz_true = function(xx, yy)
    plt.imshow(zz_true, extent=(x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]), origin="lower")
    plt.scatter(grid.grid[:, 0], grid.grid[:, 1], color="red", s=1)
    plt.colorbar()
    plt.title("Function")
    plt.savefig(plot_path + "interpolation_true_func.png")
    plt.close()

    xx, yy = grid.get_grid_data_key("x_ax"), grid.get_grid_data_key("y_ax")
    xx, yy = np.meshgrid(xx, yy)
    zz_grid = function(xx, yy)
    zz_true_flatten = grid.flatten_y(zz_grid)
    zz_true_reshaped = grid.reshape_y(zz_true_flatten)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))   
    ax[0].scatter(xx.flatten(), yy.flatten(), c=zz_true_flatten, s=20)
    ax[1].scatter(xx.flatten(), yy.flatten(), c=zz_true_reshaped.flatten(), s=20)
    plt.title("Function")
    plt.savefig(plot_path + "shape_reshape.png")
    plt.close()


    # Interpolate from surrounding points
    xx, yy = np.meshgrid(x_ax, y_ax)
    xx, yy = xx.flatten(), yy.flatten()
    zz = np.zeros(len(xx))
    S = np.column_stack((xx, yy))
    surrounding_points = grid.get_surrounding_points_S(S)
    for i in range(len(S)):
        surrounding_points_s = surrounding_points[i]
        dist_s = grid.get_dist(S[i], surrounding_points_s)
        weights = grid.get_weights(dist_s) 
        if len(surrounding_points_s) == 1:
            f_j = function(surrounding_points_s[0][0], surrounding_points_s[0][1])
        else:
            f_j = function(surrounding_points_s[:, 0], surrounding_points_s[:, 1])
        zz[i] = np.sum(weights * f_j)
    zz = zz.reshape(len(x_ax), len(y_ax))
    plt.imshow(zz, extent=(x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]), origin="lower")
    plt.scatter(grid.grid[:, 0], grid.grid[:, 1], color="red", s=1)
    plt.colorbar()
    plt.title("Interpolated function")
    plt.savefig(plot_path + "linear_interpolation_func.png")
    plt.close()


    # Plot error
    plt.imshow(np.abs(zz - zz_true), extent=(x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]), origin="lower")
    plt.colorbar()
    plt.title("Error")
    plt.savefig(plot_path + "linear_interpolation_error.png")
    plt.close()




    # Getting the timing as well
    t0 = np.random.uniform(-50,50)
    dt = np.random.uniform(200, 250)
    grid.make_time_grid(t0=t0, dt=dt)
    # Getting the time grid
    start_t = np.random.uniform(-200, -50)
    end_t = np.random.uniform(1000, 2000)
    time_grid = grid.get_timestamps_interval(start_t, end_t)
    T = np.linspace(start_t, end_t, 100)
    closest_inds = grid.get_closest_time_inds_T(T)
    closest_times = grid.get_closest_time_T(T)
    plt.plot(time_grid, np.zeros(len(time_grid)), "o", color="blue", markersize=1, label="Time grid")
    for i in range(len(T)):
        plt.plot(T[i], 1, "ro", )
        plt.plot([T[i], closest_times[i]], [1, 0], color="green")
    #plt.scatter(closest_times, np.zeros(len(closest_times)) + 2, color="red", s=10)
    plt.axvline(x=t0, color="black", linestyle="--", label="start time")
    plt.title("Plotting time grid and their closest points")
    plt.legend()
    plt.savefig(plot_path + "grid_test_time.png")
    plt.close()



    # Test all assignments in 3d
    x_start = np.random.uniform(x_lim[0] - 20, x_lim[1] + 20)
    y_start = np.random.uniform(y_lim[0] - 20, y_lim[1] + 20)
    x_end = np.random.uniform(x_lim[0] - 20, x_lim[1] + 20)
    y_end = np.random.uniform(y_lim[0] - 20, y_lim[1] + 20)
    line_x = np.linspace(x_start, x_end, 100)
    line_y = np.linspace(y_start, y_end, 100)
    line_S = np.column_stack((line_x, line_y))

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the grid points
    for t in time_grid:
        ax.scatter(grid.grid[:, 0], grid.grid[:, 1], t, color="blue", s=1)

    # Plot the line
    ax.plot(line_S[:, 0], line_S[:, 1], T, color="red", label="Line")

    assignments = grid.get_assignment(line_S, T)
    for i in range(len(line_S)):
        ax.plot([line_S[i, 0], assignments["grid_points"][i][0]], [line_S[i, 1], assignments["grid_points"][i][1]], [T[i], assignments["time_points"][i]], color="green")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Time axis')
    plt.savefig(plot_path + "grid_test_3d.png")
    plt.close()

    print("assignments[grid_inds]", assignments["grid_inds"])
    print("assignments[grid_points]", assignments["grid_points"])
    print("assignments[dist_to_closest]", assignments["dist_to_closest"])
    print("assignments[time_inds]", assignments["time_inds"])
    print("assignments[time_points]", assignments["time_points"])
    print("assignments[dist_to_closest_time]", assignments["dist_to_closest_time"])
    print("assignments[grid_time_inds]", assignments["grid_time_inds"])

    # unique tuples 
    S_ind = assignments["grid_inds"]
    T_inds = assignments["time_inds"]
    S_T_tuples = [(s_ind, t_ind) for s_ind, t_ind in zip(S_ind, T_inds)]
    tup_set = list(set(S_T_tuples))

    print("tup_set", tup_set)
    print("len(tup_set)", len(tup_set))
    print("np.unique(assignments[grid_time_inds])", np.unique(assignments["grid_time_inds"]))
    print("len(np.unique(assignments[grid_time_inds]))", len(np.unique(assignments["grid_time_inds"])))

    print(np.unique(assignments["grid_time_inds"]))




    # Test speed  

    # Make the neighbour points matrix

    print("Testing speed to make the neighbour points matrix")
    for n in [50, 100, 500]:
        grid = Grid(logger_kwargs=logging_kwargs)
        x_ax = np.linspace(0, 100, n)
        y_ax = np.linspace(0, 100, n)
        t1 = time.time()
        grid.make_regular_grid_from_axes(x_ax, y_ax)
        t2 = time.time()
        print(f"Time taken to make the grid of size {len(grid.grid)}: {t2 - t1:.2f} seconds")
        d_v = [1,2,3]
        times = []
        for d in d_v:
            t1 = time.time()
            M = grid.get_surronding_gi_matrix(d=d, boundary_condition="", diagonal=False, make_sparse=False)
            t2 = time.time()
            times.append(t2 - t1)

            if t2 - t1 > 60:
                break

        print(f"n x n: {n} x {n}", end=" ")
        for i in range(len(d_v)):
            print(f"d: {d_v[i]} time: {times[i]:.2f}", end=" ")
        print()

        if n == 50:
            surronding_matrix_d3 = grid.get_surronding_gi_matrix(d=5, boundary_condition="", diagonal=False, make_sparse=False)

            plt.imshow(surronding_matrix_d3[0:100,0:100], cmap="hot", interpolation="nearest")
            plt.savefig(plot_path + "grid_test_surrounding_matrix_d5_corner.png")
            plt.close()





    