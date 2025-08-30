import numpy as np
import pandas as pd
import os
import dill

from src.data_process.load_data import import_data

def get_ND_non_dim(data_path, df:pd.DataFrame, nu = 0.01):
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
    pos_std = np.std(input[:,1:], axis=0)
    T_mean = np.mean(input[:,0])
    T_std = np.std(input[:,0])
    if T_std == 0:
        T_std = 1. ## only one time point ie no time in the data
    L = np.sqrt(np.sum(pos_std**2)) ## L is the length scale used for Re (in order to have the same for different axis)
    ## MSE of the speed, which is the mean of the speed at all points
    speeds =  np.sqrt(np.sum(output[:,:-1]**2, axis=1))
    speed_std = np.std(output[:,:-1], axis=0)
    U_hat = np.mean(speeds)  # Non-dimensional velocity scale (represented by mean speed)

    Re = U_hat * L / nu

    p_mean = np.mean(output[:,-1])  # Mean pressure
    p_hat = p_mean  # Non-dimensional pressure scale (using dynamic pressure)
    # p_hat = U_hat**2  # Non-dimensional pressure scale (using dynamic pressure)

    T_hat = L / U_hat # useful for the equation in loss

    print("Non-dimensionalization parameters:")
    print(f"Position standard deviation (pos_std): {pos_std}")
    print(f"Length scale (L): {L}")
    print(f"Speed standard deviation (speed_std): {speed_std}")
    print(f"Time std (T_std): {T_std}")
    print(f"Velocity scale (U_hat): {U_hat}")
    print(f"Time scale (T_hat): {T_hat}")
    print(f"Pressure scale (p_hat): {p_hat}")
    print(f"Reynolds number (Re): {Re}")

    def forward_transform_input(X):
        """
        Give this function to the train function and also when plotting to format the input into a normalized format
        """
        t,position = X[0], X[1:]
        # Non-dimensionalize velocity and pressure
        position = (position - pos_mean) / pos_std
        t = (t - T_mean) / T_std

        return np.array([t] + list(position))
    forward_transform_input = np.vectorize(forward_transform_input, signature='(n)->(n)')

    def forward_transform_output(y):
        """
        Give this function to the output data before training to format the output and compare the results
        """

        speed,p = y[:-1], y[-1]
        # Non-dimensionalize velocity and pressure
        speed = speed / U_hat
        p = p / p_hat ## non-dimension is either U^2 or nu/T, in our case the first one is more suitable
        return np.array(list(speed) + [p])
    forward_transform_output = np.vectorize(forward_transform_output, signature='(n)->(n)')

    def inverse_transform_input(X):
        """
        Inverse transform the input data to original scale.
        """
        t, position = X[0], X[1:]
        position = position * pos_std + pos_mean
        t = t * T_std + T_mean

        return np.array([t] + list(position))
    inverse_transform_input = np.vectorize(inverse_transform_input, signature='(n)->(n)')

    def inverse_transform_output(y):
        """
        Inverse transform the output data to original scale.
        """
        speed, p = y[:-1], y[-1]
        speed = speed * U_hat
        p = p * p_hat

        return np.array(list(speed) + [p])
    inverse_transform_output = np.vectorize(inverse_transform_output, signature='(n)->(n)')


    return (forward_transform_input, forward_transform_output, inverse_transform_input, inverse_transform_output, Re, 
            [pos_mean, pos_std, T_mean, T_std, U_hat, T_hat, p_hat]) ## we don't give L because it can be inferred as T_hat * U_hat
            # The constants are unpacked only in loss.py in the NS functions

def save_functions(save_dir, forward_transform_input, forward_transform_output, inverse_transform_input, inverse_transform_output):
    """
    Save the transformation functions to a file.
    """ 
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'transformation_functions.pkl')
    with open(path, 'wb') as f:
             dill.dump((forward_transform_input, forward_transform_output,
                      inverse_transform_input, inverse_transform_output), f)

def save_args(args, save_dir):
    """
    Save the arguments to a file.
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

def force_2D(df, axis_to_remove, has_duplicates = True):
    name_axis_to_remove = f'Points:{axis_to_remove-1}'
    name_speed_to_remove = f'U:{axis_to_remove-1}'
    if has_duplicates:
        z_values = df[name_axis_to_remove].values
        z_sampled = z_values[0]
        df = df[df[name_axis_to_remove] == z_sampled]
    df = df.drop(columns=[name_axis_to_remove, name_speed_to_remove])

    return df