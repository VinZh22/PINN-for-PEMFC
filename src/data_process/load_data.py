import pandas as pd
import numpy as np
import pdb

import time
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper


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
        The input data as a numpy array. (t,x,y)
    Y_data : np.ndarray
        The output data as a numpy array.
    """

    columns = data.columns.tolist()
    ## get the numbers after Points:
    points_columns = [col.split(":")[1] for col in columns if col.startswith("Points:")]

    position_data = [data["Points:" + point].values.astype(np.float32) for point in points_columns]
    t_data = data["Time"].values.astype(np.float32)
    speed_data = [data["U:" + point].values.astype(np.float32) for point in points_columns]
    p_data = data["p"].values.astype(np.float32)

    # Reshape to (n_samples, 1)
    position_data = [pos.reshape(-1, 1) for pos in position_data]
    t_data = t_data.reshape(-1, 1)
    speed_data = [speed.reshape(-1, 1) for speed in speed_data]
    p_data = p_data.reshape(-1, 1)

    # Combine inputs (t, x, y) and outputs (u, v, p)
    X_data = np.hstack([t_data] + position_data)  # Shape: (n_samples, 3)
    Y_data = np.hstack(speed_data + [p_data])  # Shape: (n_samples, 3)
    if nondim_input is not None:
        X_data = nondim_input(X_data)
        Y_data = nondim_output(Y_data)
    return X_data, Y_data

# @timer_decorator
def import_data(file_path:str, df:pd.DataFrame = None, nondim_input = None, nondim_output = None):
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
    if df is None:
        print(f"Have not imported data frame yet, importing now from {file_path}")
        ## Maybe format it at the same time???
        df = pd.read_csv(file_path)
        df = format_df(df)
    ## Convert to numpy array
    X, Y = convert_to_numpy(df, nondim_input, nondim_output)
    return X, Y 

def format_df(df : pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(file_path)
    if "Time" not in df.columns:
        print("Time column not found in the DataFrame, adding a default time column.")
        df["Time"] = 100  # Add a default time column if
    # Remove the first frame because not relevant and sometime not feasible
    df = df[df["Time"] > 1]
    df = df[df["Time"] <= 3000]  # Remove the last frame because not relevant and sometime not feasible
    time_points = sorted(df["Time"].unique())
    if len(time_points) > 3:
        time_gcd = np.gcd(int(time_points[2]), int(time_points[1]))  # Calculate the GCD of the first two time points
        if time_gcd != 1:
            df.loc[:, "Time"] = df["Time"] / time_gcd  # Normalize time to the greatest common divisor
            ## Normalize the speed now
            columns = df.columns.tolist()
            points_columns = [col.split(":")[1] for col in columns if col.startswith("Points:")]
            for point in points_columns:
                df.loc[:,"U:" + point] = df["U:" + point] / time_gcd
            print(f"Normalized time to the greatest common divisor: {time_gcd}")
    return df