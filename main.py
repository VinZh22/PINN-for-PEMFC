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
import src.tools.util_func as util_func
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f'./results/{timestamp}'
os.makedirs(save_dir, exist_ok=True)

non_dim = False
if not non_dim:
    forward_transform_input = None
    forward_transform_output = None
    time_max = 150
    time_min = 0
else:
    forward_transform_input, forward_transform_output = util_func.get_non_dim_transform()
    time_max = 1.720554272517321
    time_min = -1.7436489607390302


# PINN = model.PINN_time_windows([3] + [256] * 5 + [3], time_max, time_min, RFF = False, num_windows=5, hard_constraint=None, activation=nn.Tanh)
PINN = model.PINN([3] + [256] * 5 + [3], hard_constraint=None, activation=nn.Tanh)

train_obj = train.Train_Loop_nodata(
    model=PINN,
    device=device
)

final_model = train_obj.train_pinn(
    config=[-20, 30, -20, 20, 0, 1, 200, False, 50],
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
