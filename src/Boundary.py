"""
Boundary module for the poisson project.

Objective:
    - Create boundary for the field.
    - Create obstacles within the field.
"""
from utilis.WGS import WGS
from shapely.geometry import Point, LineString, Polygon
import numpy as np
from math import cos, sin, radians
import pandas as pd
import os
import matplotlib.pyplot as plt


class Boundary:

    def __init__(self,
                border_file = None,
                file_type = "xy") -> None:
        
        # s1, load csv files for all the polygons
        if border_file is not None:
        
            # If no path is given, then use the current working directory
            # This is useful for testing
            #self.border_file = os.getcwd()+"/src/" + border_file
            self.border_file = os.path.dirname(__file__) + border_file
            self.__polygon_border = pd.read_csv(self.border_file).to_numpy()
        else:
            self.__polygon_border = None
            print("No border file is given, so the operational area is the whole field.")
            return
        

        # s2, convert wgs to xy if needed
        if file_type == "latlon":
            x, y = WGS.latlon2xy(self.__polygon_border[:, 0], self.__polygon_border[:, 1])
            self.__polygon_border = np.stack((x, y), axis=1)
        else:
            self.__polygon_border = self.__polygon_border

        # s3, create shapely polygon objects
        self.__polygon_border_shapely = Polygon(self.__polygon_border)

        # s4, create shapely line objects
        self.__line_border_shapely = LineString(self.__polygon_border)


    @staticmethod
    def create_box_xy_border_file_from_loc(loc: np.ndarray, size:float, file_name: str, loc_type: str):

        if loc_type == "latlon":
            x, y = WGS.latlon2xy(loc[0], loc[1])
        else:
            x, y = loc[0], loc[1]

        # Create the box
        box = np.array([[x-size, y-size],
                        [x-size, y+size],
                        [x+size, y+size],
                        [x+size, y-size]])
        data_frame = pd.DataFrame({"x": box[:,0], "y": box[:,1]})

        # Save the box
        file = f"src/border_files/{file_name}_xy.csv"
        data_frame.to_csv(file, index=False)




    def is_loc_legal(self, loc: np.ndarray) -> bool:
        """
        Check if a point is legal:
        That means that the point is inside the operational area and not inside the obstacle
        """
        if self.__polygon_border is None:
            return True
        if len(loc) == 3:
            loc = loc[:2]
        if self.__is_loc_legal_border(loc):
            return True
        else:
            return False
        
    def __is_loc_legal_border(self, loc: np.ndarray) -> bool:
        """
        Check if a point is legal:
        That means that the point is inside the operational area and not inside the obstacle
        """
        if self.__polygon_border is None:
            # If there is no border, then the point is legal
            return True
        if self.__border_contains(loc):
            return True
        else:
            return False
    

    def __border_contains(self, loc: np.ndarray) -> bool:
        """ Test if point is within the border polygon """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_border_shapely.contains(point)


    def is_path_legal(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """
        Check if a path is legal:
        That means that the path does not intersect with the border or the obstacle
        """
        if self.__polygon_border is None:
            return True
        if len(loc_start) == 3:
            loc_start = loc_start[:2]
        if len(loc_end) == 3:
            loc_end = loc_end[:2]

        if self.__is_loc_legal_border(loc_start)==False or self.__is_loc_legal_border(loc_end)==False:
            return False
        if self.__is_border_in_the_way(loc_start, loc_end):
            return False
        else:
            return True
        
    def is_path_legal_S(self, S: np.ndarray) -> bool:
        """
        Check if a path is legal:
        That means that the path does not intersect with the border or the obstacle
        """
        if self.__polygon_border is None:
            return True
        if len(S) == 3:
            S = S[:2]

        for i in range(len(S)-1):
            S_i = S[i]
            S_ip1 = S[i+1]

            if self.is_path_legal(S_i, S_ip1) == False:
                return False
        return True

        
    def get_closest_legal_loc(self, loc: np.ndarray) -> np.ndarray:
        """
        Get the closest legal location to loc
        """
        if self.is_loc_legal(loc):
            return loc
        else:

            if len(loc) == 3:
                x, y = loc[:2]
            if len(loc) == 2:
                x, y = loc
            point = Point(x, y)
            closest_point = self.__polygon_border_shapely.exterior.interpolate(self.__polygon_border_shapely.exterior.project(point))
            return np.array([closest_point.x, closest_point.y])
            
    def __is_border_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_border_shapely.intersects(line)
    

    def get_closest_intersect_point(self, loc_start: np.ndarray, loc_end: np.ndarray) -> np.ndarray:

        if len(loc_start) == 3:
            loc_start = loc_start[:2]
        if len(loc_end) == 3:
            loc_end = loc_end[:2]
        
        # These are the intersect points
        intersect_points = self.__get_path_intersect(loc_start, loc_end)

        closest_point = np.empty(2)
        if len(intersect_points) > 0:
            dist_list = np.linalg.norm(loc_start - intersect_points, axis=1)

            idx = dist_list.argmin()
            closest_point = intersect_points[idx]

        return closest_point

    def __get_path_intersect(self,loc_start: np.ndarray, loc_end: np.ndarray) -> np.ndarray:
        """
        Returns all the intersection points with a path with the border or obstacle         
        """

        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])

        # The line intersects will eighter return 0, 1 or multilple points
        intersect_points_border = self.__line_border_shapely.intersection(line)

        # Turn the points into numpy arrays
        if intersect_points_border.geom_type == 'Point':
            intersect_points_border_np = np.array([[intersect_points_border.x, intersect_points_border.y]])
        elif intersect_points_border.geom_type == 'MultiPoint':
            intersect_points_border_np = np.array([[e.x, e.y] for e in intersect_points_border.geoms])
        else:
            intersect_points_border_np = np.array([])




        m = len(intersect_points_border_np.reshape(-1,1))/2
        intersect_points = np.empty((int(m),2))
        
        k = 0
        if len(intersect_points_border_np) > 0:
            if len(intersect_points_border_np.reshape(-1,1)) == 2:
                # This is a single point
                intersect_points[k] = intersect_points_border_np
                k += 1
            else:
                for p in intersect_points_border_np:
                    intersect_points[k] = p
                    k += 1

        #print("intersect points", intersect_points) #REMOVE
        return np.array(intersect_points)
    
    def get_polygon_border(self) -> np.ndarray:
        return self.__polygon_border

    
    def get_exterior_border(self) -> np.ndarray:
        return self.__polygon_border_shapely.exterior.xy
    
    def get_bounding_box(self) -> np.ndarray:
        """
        Get the bounding box of the polygon
        """
        if self.__polygon_border is None:
            return np.array([-np.inf, -np.inf, np.inf, np.inf])
        else:
            minx, miny, maxx, maxy = self.__polygon_border_shapely.bounds
            return np.array([minx, miny, maxx, maxy])
        
    def get_bounding_points(self) -> np.ndarray:
        """
        Get the bounding points of the polygon
        """
        if self.__polygon_border is None:
            return np.array([0, 0])
        else:
            minx, miny, maxx, maxy = self.__polygon_border_shapely.bounds
            # Create the points
            # The points are in the order: bottom left, top left, top right, bottom right and then back to bottom left
            points = np.array([[minx, miny],
                              [minx, maxy],
                              [maxx, maxy],
                              [maxx, miny], 
                              [minx, miny]])
            return points
    

    def get_random_loc(self) -> np.ndarray:
        """
        Get a random location within the operational area
        """
        if self.__polygon_border is None:
            return np.array([0, 0])
        else:
            # Get a random interiour point
            
            # define the bounding box
            minx, miny, maxx, maxy = self.__polygon_border_shapely.bounds
            while True:
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                if self.__polygon_border_shapely.contains(Point(x, y)):
                    break
            return np.array([x, y])
    
    def print_border_xy(self):
        border = self.get_exterior_border()
        xb, yb = border
        # Print the border
        for i in range(len(xb)):
            print(f"({xb[i]}, {yb[i]})")

    def print_border_latlon(self):
        border = self.get_exterior_border()
        lat, lon = WGS.xy2latlon(border[0], border[1])
        # Print the border
        for i in range(len(lat)):
            print(f"({lat[i]}, {lon[i]})")


    def get_regular_grid_inside_border(self, point_distance=160) -> np.ndarray:
        """
        Get a regular grid inside the border
        """
        if self.__polygon_border is None:
            return np.array([0, 0])
        else:
            # Get a random interiour point
            
            # define the bounding box
            minx, miny, maxx, maxy = self.__polygon_border_shapely.bounds
            x = np.arange(minx, maxx, point_distance)
            y = np.arange(miny, maxy, point_distance)
            xx, yy = np.meshgrid(x, y)
            points = np.array([xx.flatten(), yy.flatten()]).T
            points_inside = []
            for p in points:
                if self.__polygon_border_shapely.contains(Point(p[0], p[1])):
                    points_inside.append(p)
            return np.array(points_inside)
            
    



if __name__ == "__main__":

    plot_path = "figures/tests/Boundary/"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    #### Test creating a box xy border file
    loc = np.array([200, 300])
    size = 2000 # meters
    test_name1 = "box_test"
    Boundary.create_box_xy_border_file_from_loc(loc, size, test_name1, "xy") 

    ### Test create box latlon border file
    loc = np.array([63.4, 10.4])
    size = 1500 # meters
    test_name2 = "hitl_latlon"
    Boundary.create_box_xy_border_file_from_loc(loc, size, test_name2, "latlon")


    # Test the boundary class
    files = [("/border_files/simulation_border_xy.csv", "xy", "box_xy"),
             (f"/border_files/{test_name1}_xy.csv", "xy", "box_test_xy"),
            (f"/border_files/{test_name2}_xy.csv", "xy", "box_test_latlon"),
            ("/border_files/polygon_border.csv", "latlon", "box_latlon"),
            ("/border_files/simulation_border_latlon.csv", "latlon", "torndheim_latlon"),
            (None, None, "none"),
            ("/border_files/arrow_border_xy.csv", "xy", "arrow_xy"),
            ("/border_files/arrow_border_latlon.csv", "latlon", "arrow_latlon"),
            ("/border_files/mausund_mission_area.csv", "latlon", "mausund_latlon")]
        
    for file, file_type ,name in files:
        f = Boundary(file, file_type)



        #### Plotting legal and illegal locations
        n = 200
        random_x = np.random.uniform(0, 5000, size = n)
        random_y = np.random.uniform(-2000, 2000, size = n)
        random_z = np.random.uniform(0, 100, size = n)
        random_points = np.array([random_x,random_y, random_z]).T
        for s in random_points:
            if f.is_loc_legal(s):
                plt.scatter(s[1],s[0], c="green")
            else:
                plt.scatter(s[1],s[0], c="red")
        if file is not None:
            border = f.get_exterior_border()
            xb, yb = border
            plt.plot(yb, xb, label="border", c="green")
        #plt.legend()
        plt.title(f"Legal and illegal locations {name} in xy")
        plt.axis('scaled')
        plt.savefig(f"figures/tests/Boundary/loc_legal_xy_{name}.png")
        plt.close()
    


        #### Plotting legal and illegal points in latlon
        for s in random_points:
            if f.is_loc_legal(s):
                lat, lon = WGS.xy2latlon(s[0], s[1])
                plt.scatter(lon,lat, c="green")
            else:
                lat, lon = WGS.xy2latlon(s[0], s[1])
                plt.scatter(lon,lat, c="red")
        if file is not None:
            border = f.get_exterior_border()
            xb, yb = border
            lat, lon = WGS.xy2latlon(xb, yb)
            plt.plot(lon, lat, label="border", c="green")
        #plt.legend()
        plt.xlabel("Longitude"), plt.ylabel("Latitude")
        plt.title(f"Legal and illegal locations {name} in latlon \n File name: {name}  File type: {file_type}")
        plt.savefig(f"figures/tests/Boundary/loc_legal_latlon_{name}.png")
        plt.close()

        
        #### Plotting legal and illegal paths
        n = 40
        random_x = np.random.uniform(0, 5000, size = n)
        random_y = np.random.uniform(-2000, 2000, size = n)
        random_points_A = np.array([random_x,random_y]).T

        random_x = np.random.uniform(0, 5000, size = n)
        random_y = np.random.uniform(-2000, 2000, size = n)
        random_points_B = np.array([random_x,random_y]).T
        for i in range(n):
            loc_stat = random_points_A[i]
            loc_end = random_points_B[i]
            if f.is_path_legal(loc_stat, loc_end):
                plt.plot([loc_stat[1], loc_end[1]],[loc_stat[0], loc_end[0]], c="green")
            else:
                #intersect_points = f.__get_path_intersect(loc_stat, loc_end)
                closest_points = f.get_closest_intersect_point(loc_stat, loc_end)
                
                plt.plot([loc_stat[1], loc_end[1]],[loc_stat[0], loc_end[0]], c="red")
                plt.scatter(loc_stat[1], loc_stat[0], c="brown")
                #plt.scatter(intersect_points[:,1], intersect_points[:,0], c="black")
                plt.scatter(closest_points[1], closest_points[0], c="blue")
        if file is not None:
            border = f.get_exterior_border()
            xb, yb = border
            plt.plot(yb, xb, label="border", c="green")
        #plt.legend()
        plt.title(f"Legal and illegal paths {name} in xy")
        plt.axis('scaled')
        plt.savefig(f"figures/tests/Boundary/path_legal_xy_{name}.png")
        plt.close()


        # Testing the closest legal location
        if file is not None:
            n = 10
            while n > 0:
                random_x = np.random.uniform(0, 5000)
                random_y = np.random.uniform(-2000, 2000)
                random_z = np.random.uniform(0, 100)
                random_points = np.array([random_x,random_y, random_z])
                if not f.is_loc_legal(random_points):
                    closest_point = f.get_closest_legal_loc(random_points)
                    plt.scatter(random_points[1],random_points[0], c="green")
                    plt.scatter(closest_point[1],closest_point[0], c="blue")
                    plt.plot([random_points[1], closest_point[1]],[random_points[0], closest_point[0]], c="red")
                    n -= 1

            # Add the border
            if file is not None:
                border = f.get_exterior_border()
                xb, yb = border
                plt.plot(yb, xb, label="border", c="green")
            plt.savefig(f"figures/tests/Boundary/closest_legal_loc_{name}.png")
            plt.close()



        # Testing the random location

        if file is not None:
            n = 100
            for i in range(n):
                random_point = f.get_random_loc()
                plt.scatter(random_point[1], random_point[0], c="green")
            border = f.get_exterior_border()
            xb, yb = border
            plt.plot(yb, xb, label="border", c="green")
            plt.title(f"Random locations {name} in xy")
            plt.axis('scaled')
            plt.savefig(f"figures/tests/Boundary/random_loc_{name}.png")
            plt.close()


        # Testing a regular grid
        if file is not None:
            regular_grid = f.get_regular_grid_inside_border(160)
            print("Regular grid shape", regular_grid.shape)
            plt.scatter(regular_grid[:,1], regular_grid[:,0], c="green")
            border = f.get_exterior_border()
            xb, yb = border
            plt.plot(yb, xb, label="border", c="green")
            plt.title(f"Regular grid locations {name} in xy")
            plt.axis('scaled')
            plt.savefig(f"figures/tests/Boundary/regular_grid_loc_{name}.png")
            plt.close()
                
