import numpy as np

x_mean = 3.088352e+00
y_mean = -1.905240e-16	
x_std =  1.051818e+01
y_std = 8.089716e+00
T_mean = 7.550000e+01

U_std = 3.204348e-01
T_std = x_std / U_std

nu = 0.01

def forward_transform_input(X):
    """
    Give this function to the train function and also when plotting to format the input into a normalized format
    """

    # data from the cylinder.csv file, get it separately, no need to compute if everytime
    t,x,y = X[0], X[1], X[2]
    # Non-dimensionalize velocity and pressure
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    t = (t - T_mean) / T_std

    return np.array([t, x, y])
forward_transform_input = np.vectorize(forward_transform_input, signature='(n)->(n)')

def forward_transform_output(y):
    """
    Give this function to the output data before training to format the output and compare the results
    """

    # data from the cylinder.csv file, get it separately, no need to compute if everytime
    

    u,v,p = y[0], y[1], y[2]
    # Non-dimensionalize velocity and pressure
    u = u / U_std
    v = v / U_std
    p = p / U_std**2 ## non-dimension is either U^2 or nu/T, in our case the first one is more suitable
    return np.array([u, v, p])
forward_transform_output = np.vectorize(forward_transform_output, signature='(n)->(n)')

def get_non_dim_transform():
    """
    Get the non-dimensionalization functions for input and output data.
    Returns
    -------
    tuple
        A tuple containing the non-dimensionalization functions for input and output data.
    """
    return forward_transform_input, forward_transform_output

def get_Reynolds():
    """
    Re = U * L / nu
    """
    return U_std * x_std / nu

