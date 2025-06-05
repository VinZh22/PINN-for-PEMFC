import numpy as np
import pandas as pd
import os

from src.data_process.load_data import import_data

def get_ND_non_dim(data_path, df, nu = 0.01):
    """
    Get the non-dimensionalization parameters for 2D data.
    Parameters
    ----------
    data_path : str
        Path to the data file.
    nu : float, optional
        Kinematic viscosity. The default is 0.01.

    Returns
    -------
    tuple
        (forward_transform_input, forward_transform_output, inverse_transform_input, inverse_transform_output, Re)
    """
    # Load the data
    input,output = import_data(data_path, df=df)
    
    pos_mean = np.mean(input[:,1:], axis=0)
    x_mean = np.mean(input[:,1])
    y_mean = np.mean(input[:,2])
    pos_std = np.std(input[:,1:], axis=0)
    x_std = np.std(input[:,1])
    y_std = np.std(input[:,2])
    L = np.sqrt(np.sum(pos_mean**2)) ## L is the length scale, which is the diagonal of the cylinder
    T_mean = np.mean(input[:,0])
    speed_std = np.std(output[:,:-1], axis=0)
    U_std = np.std(output[:,0])
    V_std = np.std(output[:,1])

    U_hat = np.sqrt(np.sum(speed_std**2))  # Non-dimensional velocity scale

    # L = x_std
    # U_hat = U_std

    T_hat = L / U_hat

    print("Non-dimensionalization parameters:")
    print(f"Length scale (L): {L}")
    print(f"Velocity scale (U_hat): {U_hat}")
    print(f"Time scale (T_hat): {T_hat}")
    print(f"Reynolds number (Re): {U_hat * L / nu}")

    def forward_transform_input(X):
        """
        Give this function to the train function and also when plotting to format the input into a normalized format
        """
        t,position = X[0], X[1:]
        # Non-dimensionalize velocity and pressure
        position = (position - pos_mean) / pos_std
        t = (t - T_mean) / T_hat

        return np.array([t] + list(position))
    forward_transform_input = np.vectorize(forward_transform_input, signature='(n)->(n)')

    def forward_transform_output(y):
        """
        Give this function to the output data before training to format the output and compare the results
        """

        speed,p = y[:-1], y[-1]
        # Non-dimensionalize velocity and pressure
        speed = (speed - speed_std) / U_hat
        p = p / U_hat**2 ## non-dimension is either U^2 or nu/T, in our case the first one is more suitable
        return np.array(list(speed) + [p])
    forward_transform_output = np.vectorize(forward_transform_output, signature='(n)->(n)')

    def inverse_transform_input(X):
        """
        Inverse transform the input data to original scale.
        """
        t, position = X[0], X[1:]
        position = position * pos_std + pos_mean
        t = t * T_hat + T_mean

        return np.array([t] + list(position))
    inverse_transform_input = np.vectorize(inverse_transform_input, signature='(n)->(n)')

    def inverse_transform_output(y):
        """
        Inverse transform the output data to original scale.
        """
        speed, p = y[:-1], y[-1]
        speed = speed * U_hat + speed_std
        p = p * U_hat**2

        return np.array(list(speed) + [p])
    inverse_transform_output = np.vectorize(inverse_transform_output, signature='(n)->(n)')


    Re = U_hat * L / nu

    return forward_transform_input, forward_transform_output, inverse_transform_input, inverse_transform_output, Re



def save_args(args, save_dir):
    """
    Save the arguments to a file.
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")