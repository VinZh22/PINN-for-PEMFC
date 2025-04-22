import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PINN(nn.Module):
    def __init__(self, layers_num, hard_constraint = None, activation=nn.Tanh, ):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN, self).__init__()
        self.layers_num = layers_num
        
        layers = []
        for i in range(len(layers_num) - 1):
            layers.append(nn.Linear(layers_num[i], layers_num[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(activation())
        
        self.hard_constraint = hard_constraint
        
        self.model = nn.Sequential(*layers[:-1])  # Exclude the last activation function
        

    def forward(self, x):
        y = self.model(x)
        # Apply the hard constraint if provided
        if self.hard_constraint is not None:
            y = self.hard_constraint(y)
        return y

class PINN_time_windows(nn.Module):
    def __init__(self, layers_num, T_max, T_min = 0, num_windows = 10, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh, ):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_time_windows, self).__init__()
        self.layers_num = layers_num # suppose that the first layer is even
        self.T_max = T_max
        self.T_min = T_min
        self.num_windows = num_windows
        self.window_start = np.linspace(T_min, T_max, num_windows, endpoint=False)
        
        layers = []
        for i in range(len(layers_num) - 1):
            layers.append(nn.Linear(layers_num[i], layers_num[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(activation())
        
        self.hard_constraint = hard_constraint
        
        self.RFF = RFF
        self.register_buffer('kernel_rff', torch.normal(mean = 0., std = RFF_sigma, size = (layers_num[0]//2, 3)))

        self.models = nn.ModuleList([nn.Sequential(*layers[:-1]) for _ in range(num_windows)])  # Exclude the last activation function

    def forward(self, x):
        # Extract time from input
        t = x[:, 0]
        # Initialize output tensor
        y = torch.zeros_like(x)
        # Determine which window the input belongs to
        window_indices = np.digitize(t.cpu().detach().numpy(), self.window_start) - 1
        if self.RFF:
            # Apply RFF transformation
            x = torch.matmul(self.kernel_rff, x.T)
            x_cos = torch.cos(x).T
            x_sin = torch.sin(x).T

            x = torch.cat((x_cos, x_sin), dim = 1)

        for i in range(self.num_windows):
            # Get the indices for the current window
            indices = (window_indices == i)
            if indices.any():
                # Pass the inputs through the corresponding model
                y[indices] = self.models[i](x[indices])
        
        # Apply the hard constraint if provided
        if self.hard_constraint is not None:
            y = self.hard_constraint(y)
        
        return y