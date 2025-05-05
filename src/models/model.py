import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PINN(nn.Module):
    def __init__(self, input_length, output_length, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh, ):
        """
        Abstract class for PINN, NEED TO DECLARE a self.model in the child class
        activation: activation function to be used in the hidden layers
        """
        assert input_length==3 or RFF # if there is no RFF, then the input is u,v,p
        super(PINN, self).__init__()
        self.input_length = input_length
        self.output_length = output_length

        self.hard_constraint = hard_constraint
        self.RFF = RFF
        self.register_buffer('kernel_rff', torch.normal(mean = 0., std = RFF_sigma, size = (input_length//2, 3)))
        
    def forward(self, x):
        if self.RFF:
            # Apply RFF transformation
            x = torch.matmul(self.kernel_rff, x.T)
            x_cos = torch.cos(x).T
            x_sin = torch.sin(x).T

            x = torch.cat((x_cos, x_sin), dim = 1)
        
        y = self.model(x)
        # Apply the hard constraint if provided
        if self.hard_constraint is not None:
            y = self.hard_constraint(y)
        return y

class PINN_linear(PINN):
    def __init__(self, layers_num, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh, ):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_linear, self).__init__(layers_num[0], layers_num[-1], RFF, RFF_sigma, hard_constraint, activation)
        self.layers_num = layers_num

        layers = []
        for i in range(len(layers_num) - 1):
            layers.append(nn.Linear(layers_num[i], layers_num[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(activation())
        
        self.model = nn.Sequential(*layers[:-1])  # Exclude the last activation function

class PINN_import(PINN):
    def __init__(self, model_path, input_len, output_len, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh, ):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """

        super(PINN_import, self).__init__(input_len, output_len, RFF, RFF_sigma, hard_constraint, activation)
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        """
        self.model = torch.load(model_path)

class PINN_LORA(PINN):
    def __init__(self, rank, grid_size, output_len = 3, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh):
        """
        We have input_len = 3 as fixed, no RFF
        """
        assert RFF == False, "RFF is not supported in LORA"
        super(PINN_LORA, self).__init__(3, output_len, RFF, RFF_sigma, hard_constraint, activation)
        self.rank = rank
        self.gridSize = grid_size

        self.model = LORA(rank, output_len, grid_size)

class LORA(nn.Module):
    def __init__(self, rank=1, output_length=3, grid_size=100):
        super(LORA, self).__init__()
        self.rank = rank
        self.input_length = 3
        self.output_length = output_length
        self.gridSize = grid_size

        self.vecMode = [self.input_length - 1 - i for i in range(self.input_length)]  # The order of the coordinates

        # Initialize low-rank matrices
        self.line_vec = self.init_one_svd(self.rank, self.gridSize, 0.2)
        self.basis_mat = torch.nn.Linear(self.rank, self.output_length, bias=False)
    
    def init_one_svd(self, n_component, gridSize, scale):
        line_coef = []
        for i in range(self.input_length):
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize, 1))))
        return torch.nn.ParameterList(line_coef)


    def forward(self, xyz_sampled):
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.line_vec[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.line_vec[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.line_vec[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)


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