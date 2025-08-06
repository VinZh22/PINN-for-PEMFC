import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb


class PINN(nn.Module):
    def __init__(self, input_length, output_length, data_input, device, RFF = False, LT = False, RFF_sigma = 1., LT_nb_geometry = 1, hard_constraint = None):
        """
        Abstract class for PINN, NEED TO DECLARE a self.model in the child class
        activation: activation function to be used in the hidden layers
        """
        assert input_length==data_input or (RFF and input_length%2==0) # if there is no RFF, then the input is u,v,p
        super(PINN, self).__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.data_input = data_input
        self.device = device

        self.hard_constraint = hard_constraint
        self.RFF = RFF
        self.register_buffer('kernel_rff', torch.normal(mean = 0., std = RFF_sigma, size = (input_length//2, data_input)))
        self.LT = LT  # Flag for using Lagrangian Topology
        if self.LT:
            self.initialize_LT(nb_geometries=LT_nb_geometry)
    
    def initialize_LT(self, nb_geometries):
        self.deltas = []
        print(f"Creating {nb_geometries} distance functions for Lagrangian Topology")
        for i in range(nb_geometries):
            delta = Distance_function(seed=i)
            delta.to(self.device)
            self.deltas.append(delta)
        
        self.deltas = nn.ModuleList(self.deltas)
        
    def forward(self, x, nn_only = False):
        if self.RFF:
            # Apply RFF transformation
            x_bar = torch.matmul(self.kernel_rff, x.T)
            x_cos = torch.cos(x_bar).T
            x_sin = torch.sin(x_bar).T

            x_star = torch.cat((x_cos, x_sin), dim = 1)
        else:
            x_star = x
        y = self.model(x_star)

        if self.LT and not nn_only:
            # Apply the distance functions for each geometry
            # pdb.set_trace()
            prod = self.deltas[0](x)  # Initialize with the first delta
            for delta in self.deltas[1:]:
                prod *= delta(x)
            y = prod.view(-1, 1) * y

        # Apply the hard constraint if provided
        if self.hard_constraint is not None:
            y = self.hard_constraint(y)
        return y

class PINN_linear(PINN):
    def __init__(self, layers_num, data_input, device, RFF = False, LT = False, RFF_sigma = 1., LT_nb_geometry = 1, hard_constraint = None, activation=nn.Tanh):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_linear, self).__init__(layers_num[0], layers_num[-1], data_input, device, RFF, LT, RFF_sigma, LT_nb_geometry, hard_constraint)
        self.layers_num = layers_num

        layers = []
        for i in range(len(layers_num) - 1):
            layers.append(nn.Linear(layers_num[i], layers_num[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(activation())
        
        self.model = nn.Sequential(*layers[:-1])  # Exclude the last activation function

class PINN_mod_MLP(PINN):
    def __init__(self, layers_num, data_input, device, RFF = False, LT = False, RFF_sigma = 1., LT_nb_geometry = 1, hard_constraint = None, activation=nn.Tanh, ):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_mod_MLP, self).__init__(layers_num[0], layers_num[-1], data_input, device, RFF, LT, RFF_sigma, LT_nb_geometry, hard_constraint)
        self.layers_num = layers_num

        self.model = ModifiedMLP(layers_num=layers_num, device = device, activation=activation)
class ModifiedMLP(nn.Module):
    def __init__(self, layers_num, device, activation = nn.Tanh):
        super().__init__()
        self.device = device

        self.U = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), activation()).to(device)
        self.V = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), activation()).to(device)
        nn.init.xavier_uniform_(self.U[0].weight)
        nn.init.xavier_uniform_(self.V[0].weight)

        self.layers = []
        for i in range(1, len(layers_num) - 1):
            self.layers.append(nn.Sequential(nn.Linear(layers_num[i-1], layers_num[i]), activation()).to(device))
            nn.init.xavier_uniform_(self.layers[-1][0].weight)
        self.layers.append(nn.Linear(layers_num[-2], layers_num[-1]).to(device))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        U = self.U(x)
        V = self.V(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = (1-x)*U + x*V
        
        x = self.layers[-1](x)
        return x

class DM_PINN(PINN):
    def __init__(self, layers_num, data_input, device, RFF = False, LT = False, RFF_sigma = 1., LT_nb_geometry = 1, hard_constraint = None, activation=nn.Tanh):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(DM_PINN, self).__init__(layers_num[0], layers_num[-1], data_input, device, RFF, LT, RFF_sigma, LT_nb_geometry, hard_constraint)
        self.layers_num = layers_num

        self.model = DM_MLP(layers_num=layers_num, device=device, activation=activation)

class DM_MLP(nn.Module):
    def __init__(self, layers_num, device, activation = nn.Tanh):
        super().__init__()
        self.device = device

        self.layers = []
        for i in range(1, len(layers_num) - 1):
            self.layers.append(nn.Sequential(nn.Linear(layers_num[i-1], layers_num[i]), activation()).to(device))
            nn.init.xavier_uniform_(self.layers[-1][0].weight)
        self.layers.append(nn.Linear(layers_num[-2], layers_num[-1]).to(device))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        x = self.layers[0](x)
        prod = torch.ones_like(x)
        for layer in self.layers[1:-1]:
            x = layer(x) * prod
            prod = x
        
        x = self.layers[-1](x)
        return x

class PINN_PirateNet(PINN): ## We will modify a bit compared to the paper, with less internal computation in a layer while retaining the core idea
    def __init__(self, layers_num, data_input, device, RFF = False, LT = False, RFF_sigma = 1., LT_nb_geometry = 1, hard_constraint = None, activation=nn.Tanh, ):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_PirateNet, self).__init__(layers_num[0], layers_num[-1], data_input, device, RFF, LT, RFF_sigma, LT_nb_geometry, hard_constraint)
        self.layers_num = layers_num

        self.model = PirateNet(layers_num=layers_num, device = device, activation=activation)
class PirateNet(nn.Module):
    def __init__(self, layers_num, device, activation = nn.Tanh):
        super().__init__()
        self.device = device
        self.layers_num = layers_num

        self.U = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), activation()).to(device)
        self.V = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), activation()).to(device)
        nn.init.xavier_uniform_(self.U[0].weight)
        nn.init.xavier_uniform_(self.V[0].weight)

        self.layers1 = []
        self.layers2 = []
        for i in range(1,len(layers_num) - 1):
            self.layers1.append(nn.Sequential(nn.Linear(layers_num[i-1], layers_num[i]), activation()).to(device))
            nn.init.xavier_uniform_(self.layers1[-1][0].weight)
            self.layers2.append(nn.Sequential(nn.Linear(layers_num[i-1], layers_num[i]), activation()).to(device))
            nn.init.xavier_uniform_(self.layers2[-1][0].weight)

        self.layers1 = nn.ModuleList(self.layers1)
        self.layers2 = nn.ModuleList(self.layers2)

        self.alphas = nn.Parameter(torch.zeros(len(layers_num)-1, device=device))  # learnable parameters for the alphas
        self.last_layer = nn.Linear(layers_num[-2], layers_num[-1]).to(device)
        nn.init.xavier_uniform_(self.last_layer.weight)

    def forward(self, x):
        # pdb.set_trace()
        U = self.U(x)
        V = self.V(x)
        n = len(self.layers1)
        for layer_num in range(n):
            passed = self.layers1[layer_num](x)
            passed = (1-passed)*U + passed*V
            passed = self.layers2[layer_num](passed)
            x = (1-self.alphas[layer_num]) * x + self.alphas[layer_num] * passed
        
        x = self.last_layer(x)
        return x

class PINN_import(PINN):
    def __init__(self, model_path, input_len, output_len, data_input, device,RFF = False, LT = False, RFF_sigma = 1., LT_nb_geometry = 1, hard_constraint = None,):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """

        super(PINN_import, self).__init__(input_len, output_len, data_input, device, RFF, LT, RFF_sigma, LT_nb_geometry, hard_constraint)
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        """
        self.model = torch.load(model_path,  weights_only=False)

class PINN_import_lora(PINN):
    def __init__(self, model_path, input_len, output_len, data_input, device, RFF = False, LT = False, RFF_sigma = 1., LT_nb_geometry = 1, hard_constraint = None, r=4):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_import_lora, self).__init__(input_len, output_len, data_input, device, RFF, LT, RFF_sigma, LT_nb_geometry, hard_constraint)
        self.load_model(model_path, input_len, output_len, r)

    def load_model(self, model_path, input_len, output_len, r):
        """
        Load the model from the specified path.
        """
        self.model = Import_LORA(model_path=model_path, inp=input_len, out=output_len, r=r)

class Import_LORA(nn.Module):
    def __init__(self, model_path, inp, out, r=4):
        super().__init__()
        self.r = r
        self.lora_A = nn.Parameter(torch.randn(inp, r))
        self.lora_B = nn.Parameter(torch.randn(r, out))

        self.loaded_model = torch.load(model_path, weights_only=False)
        self.loaded_model.eval()  # Set the loaded model to evaluation mode

    def forward(self, x):
        return self.loaded_model(x) + (x @ self.lora_A) @ self.lora_B

def softmax(a : torch.Tensor,b : torch.Tensor) -> torch.Tensor:
    """
    Softmax function for two inputs a and b.
    """
    k = torch.scalar_tensor(5., dtype=torch.float32) # controls the sharpness (HYPERPARAMETER)
    epsilon = torch.scalar_tensor(1e-10, dtype=torch.float32)  # to avoid zero in log
    exp_a = torch.exp(k * a)
    exp_b = torch.exp(k * b)
    # assert torch.min(exp_a)>epsilon*100 and torch.min(exp_a)>epsilon*100, "Softmax inputs are too small, might lead to numerical instability"
    return torch.log(exp_a + exp_b + epsilon)/k

def softAbsolute(x : torch.Tensor) -> torch.Tensor:
    """
    Soft absolute function to ensure smoothness.
    """
    epsilon = torch.scalar_tensor(1e-20, dtype=torch.float32)  # to avoid zero in log
    return torch.sqrt(torch.square(x) + epsilon)

class SDF_box(nn.Module):
    def __init__(self, seed):
        """

        """
        torch.manual_seed(seed)  # Set the seed for reproducibility
        super(SDF_box, self).__init__()
        # random initialization
        self.center = nn.Parameter(torch.randn(2, dtype=torch.float32))  # Center of the box (x, y)
        self.dimensions = nn.Parameter(torch.rand(2, dtype=torch.float32))  # sqrt of Width and height of the box (gonna square later to ensure positiveness)

    def forward(self, X):
        ## Formula of sdf according to https://iquilezles.org/articles/distfunctions2d/
        x,y = X[:,1], X[:,2] ## shape (batch_size,) each
        ## Using squared values instead of absolute values for the SDF
        p = torch.vstack([ torch.abs(x-self.center[0]), torch.abs(y-self.center[1]) ]) ## shape (2, batch_size)
        d = p - torch.relu(self.dimensions).view(2, -1)

        max_d_0 = torch.max(d, torch.zeros_like(d))  # max(d, 0.)
        max_d = torch.max(d[0], d[1])  # max(d_x, d_y) 
        min_d_0 = -torch.max(-max_d, torch.zeros_like(max_d))  # min( max(d.x, d.y), 0.)
        length = torch.linalg.vector_norm(max_d_0, dim=0)  # Length(max(d, 0.))

        return length + min_d_0  # SDF = Length(max(d, 0.)) + min( max(d.x, d.y), 0.)
        

class Distance_function(nn.Module):
    def __init__(self, seed):
        """
        Info : we use the sdf of a 2D box, because we suppose that the data is already filtered to be in the bipolar plate, and then the geometry is invariant in the z direction.
        """
        super().__init__()
        self.sdf = SDF_box(seed)

    def forward(self, X):
        beta = 100 # controls the sharpness (HYPERPARAMETER)
        return 1 / (1 + torch.exp(- beta * self.sdf(X)))  # Sigmoid function to ensure the output is between 0 and 1
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