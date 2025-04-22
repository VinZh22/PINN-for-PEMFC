import numpy as np
import src.models.model as model
import src.models.train as train
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import os
import src.data_process.load_data as load_data
import src.tools.plot_tools as plot_tools
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f'./results/{timestamp}'
os.makedirs(save_dir, exist_ok=True)

# create non-dimensional transformation
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

non_dim = True
if not non_dim:
    forward_transform_input = None
    forward_transform_output = None
    time_max = 150
    time_min = 0
else:
    time_max = 1.720554272517321
    time_min = -1.7436489607390302


# PINN = model.PINN_time_windows([3] + [256] * 5 + [3], time_max, time_min, RFF = False, num_windows=5, hard_constraint=None, activation=nn.Tanh)
PINN = model.PINN([3] + [256] * 5 + [3], hard_constraint=None, activation=nn.Tanh)

# final_model, loss_history_train, loss_history_test = train.train_pinn(
#     model=PINN,
#     file_path="./data/cylinder.csv",
#     train_prop=0.1,
#     device=device,
#     nu=0.01,
#     epochs=5000,
#     nondim_input=forward_transform_input,
#     nondim_output=forward_transform_output,
#     adapting_weight=True
# )

train_obj = train.Train_Loop_data(
    model=PINN,
    device=device
)

final_model = train_obj.train_pinn(
    config="./data/cylinder.csv",
    train_prop=0.01,
    nu=0.01,
    epochs=5000,
    nondim_input=forward_transform_input,
    nondim_output=forward_transform_output,
    adapting_weight=True
)

loss_history_train, loss_history_test = train_obj.get_loss_history()

# Define time and space grids
end_T_sim = 200

x = np.linspace(-20, 30, 100)  # Spatial grid (x)
y = np.linspace(-20, 20, 100)  # Spatial grid (y)
t = np.linspace(0, end_T_sim, 100)   # Time grid

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y)

# Plot loss history
plot_tools.plot_loss_history({"train_loss": loss_history_train, "val_loss": loss_history_test}, save_dir=save_dir)
# Plot speed maps
plot_tools.plot_speed_map(
    model=final_model,
    X=X,
    Y=Y,
    t=t,
    save_dir=save_dir,
    device=device,
    non_dim=non_dim,
    forward_transform_input=forward_transform_input
)
