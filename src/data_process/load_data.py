import pandas as pd
import numpy as np


def convert_to_numpy(data, nondim_input = None, nondim_output = None):
    """
    Convert the data from a pandas DataFrame to numpy arrays and apply non-dimensionalization if needed.
    Parameters
    ----------
    data : pd.DataFrame
        The input data containing columns for x, y, t, u, v, and p.
    nondim_input : function, optional
        A function to apply non-dimensionalization to the input data.
    nondim_output : function, optional
        A function to apply non-dimensionalization to the output data.
    Returns
    -------
    X_data : np.ndarray
        The input data as a numpy array.
    Y_data : np.ndarray
        The output data as a numpy array.
    """
    x_data = data["Points:0"].values.astype(np.float32)
    y_data = data["Points:1"].values.astype(np.float32)
    t_data = data["Time"].values.astype(np.float32)
    u_data = data["U:0"].values.astype(np.float32)
    v_data = data["U:1"].values.astype(np.float32)
    p_data = data["p"].values.astype(np.float32)

    # Reshape to (n_samples, 1)
    x_data = x_data.reshape(-1, 1)
    y_data = y_data.reshape(-1, 1)
    t_data = t_data.reshape(-1, 1)
    u_data = u_data.reshape(-1, 1)
    v_data = v_data.reshape(-1, 1)
    p_data = p_data.reshape(-1, 1)

    # Combine inputs (t, x, y) and outputs (u, v, p)
    X_data = np.hstack([t_data, x_data, y_data])  # Shape: (n_samples, 3)
    Y_data = np.hstack([u_data, v_data, p_data])  # Shape: (n_samples, 3)
    if nondim_input is not None:
        X_data = nondim_input(X_data)
        Y_data = nondim_output(Y_data)
    return X_data, Y_data

def import_data(file_path, nondim_input = None, nondim_output = None):
    """
    Import data from a CSV file and convert it to numpy arrays.
    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    nondim_input : function, optional
        A function to apply non-dimensionalization to the input data.
    nondim_output : function, optional
        A function to apply non-dimensionalization to the output data.
    Returns
    ----------
    X : np.ndarray
        The input data as a numpy array.
    Y : np.ndarray
        The output data as a numpy array.
    """
    df = pd.read_csv(file_path)
    ## Convert to numpy array
    X, Y = convert_to_numpy(df, nondim_input, nondim_output)
    return X, Y 
