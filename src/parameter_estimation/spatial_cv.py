
#### This file contains the code for the spatial cross-validation of the parameter estimation

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error



###### 


