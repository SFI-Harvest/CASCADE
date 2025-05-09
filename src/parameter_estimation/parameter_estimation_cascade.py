import numpy as np
import random
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy import sparse
from scipy.sparse import diags
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression



# List of parameter and functions to test
"""
- Measurment noises
- tau_y
- tau_upper_bound and tau_lower_bound
- tau_amplitude
- sigma_y
- sigma_upper_bound and sigma_lower_bound
- sigma_amplitude
- phi_xy
- phi_t
- transformation_y
- transformation_sigma
- transformation_y_asv
- transformation_t
- transition_matrix


for each of this we need a search space 
a and a set function 
and a way to evaluate the fitness of the model
"""

def spatial_cross_validation(data, n_splits=5):
    """
    Perform spatial cross-validation on the data.
    
    Args:
        data (np.ndarray): The data to be used for cross-validation.
        n_splits (int): The number of splits for cross-validation.
        
    Returns:
        list: A list of tuples containing the train and test indices for each split.
    """


    # Create a KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Create a list to store the train and test indices
    splits = []
    
    # Iterate through the splits
    for train_index, test_index in kf.split(data):
        splits.append((train_index, test_index))
    
    return splits



def estimate_tau(data_files): 
    """ 
    Estmating tau depends on considering that two consecutive points are correlated
    in time and space.
    
    Args:
        data_files (list): list of data files to be used for estimation
    Returns:
        tau_dict: a dictionary with the estimated tau for each vehicle and each variable
    """
    variables = ["y", "bounds"]
    tau_dict = {}
    for data_file in data_files:
        # Load the data
        data = np.load(data_file)
        
        # Extract the variables
        x = data['x']
        y = data['y']
        
        # Estimate tau for each variable
        tau_x = estimate_tau_y(x)
        tau_y = estimate_tau_y(y)
        
        # Store the results in the dictionary
        tau_dict[data_file] = {'tau_x': tau_x, 'tau_y': tau_y}
    return tau_dict


def estimate_tau_y(y):
    """
    Estimate the tau parameter from the y data.
    Args:
        y (np.ndarray): The y data.
    Returns:
        float: The estimated tau.

    This will overestimate the tau when the c(s_1, t_1, s_2, t_2) is much smaller than 1
    if the correlation is low this will overestimate the tau
    """
    diff = y[1:] - y[:-1]
    v = np.var(diff)
    return np.sqrt(v/2)


if __name__ == "__main__":
    x = np.linspace(0, 10, 5000)
    true_tau = 0.2
    y = np.sin(x)*1 + np.random.normal(0, true_tau, len(x))
    tau = estimate_tau_y(y)
    print(f"Estimated tau: {tau}")

    tau_list = []
    n_list = 10**np.linspace(1, 5, 20)
    for n in n_list :
        x = np.linspace(0, 10, int(n))
        true_tau = 0.2
        y = np.sin(x)* 5
        y_measurement = y + np.random.normal(0, true_tau, len(x))   
        tau = estimate_tau_y(y_measurement)
        print(f"Estimated tau for n={n}: {tau}")
        tau_list.append(tau)
    plt.plot(n_list, tau_list)
    plt.hlines(true_tau, min(n_list), max(n_list), colors='r', linestyles='dashed', label='True tau')
    plt.xlabel("n")
    # Log scale for n
    plt.xscale('log')
    plt.ylabel("Estimated tau")
    plt.title("Estimated tau vs n")
    plt.show()


    





