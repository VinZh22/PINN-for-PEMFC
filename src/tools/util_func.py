import numpy as np

def forward_transform_input(X):
    """
    Give this function to the train function and also when plotting to format the input into a normalized format
    """

    # data from the cylinder.csv file, get it separately, no need to compute if everytime
    x_mean = 9.113882e-01
    y_mean = 3.184817e-05	
    x_std =  3.204348e-01
    y_std = 1.304893e-01
    T_mean = 7.550000e+01
    T_std = 4.330032e+01
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
    U_mean = 9.113882e-01
    V_mean = 3.184817e-05	
    U_std =  3.204348e-01
    V_std = 1.304893e-01
    P_mean = -3.673176e-02
    P_std = 1.302342e-01

    u,v,p = y[0], y[1], y[2]
    # Non-dimensionalize velocity and pressure
    u = (u - U_mean) / U_std
    v = (v - V_mean) / V_std
    p = (p - P_mean) / P_std
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