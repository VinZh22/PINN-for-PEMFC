import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb


class PINN(nn.Module):
    def __init__(self, input_length, output_length, data_input, device, RFF = False, RFF_sigma = 1., hard_constraint = None):
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

        # Apply the hard constraint if provided
        if self.hard_constraint is not None:
            y = self.hard_constraint(y)
        return y

class PINN_linear(PINN):
    def __init__(self, layers_num, data_input, device, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_linear, self).__init__(layers_num[0], layers_num[-1], data_input, device, RFF, RFF_sigma, hard_constraint)
        self.layers_num = layers_num

        layers = []
        for i in range(len(layers_num) - 1):
            layers.append(nn.Linear(layers_num[i], layers_num[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(activation())
        
        self.model = nn.Sequential(*layers[:-1])  # Exclude the last activation function

class PINN_mod_MLP(PINN):
    def __init__(self, layers_num, data_input, device, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh, ):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_mod_MLP, self).__init__(layers_num[0], layers_num[-1], data_input, device, RFF, RFF_sigma, hard_constraint)
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
    def __init__(self, layers_num, data_input, device, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(DM_PINN, self).__init__(layers_num[0], layers_num[-1], data_input, device, RFF, RFF_sigma, hard_constraint)
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
    def __init__(self, layers_num, data_input, device, RFF = False, RFF_sigma = 1., hard_constraint = None, activation=nn.Tanh, ):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_PirateNet, self).__init__(layers_num[0], layers_num[-1], data_input, device, RFF, RFF_sigma, hard_constraint)
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
    def __init__(self, model_path, input_len, output_len, data_input, device,RFF = False, RFF_sigma = 1., hard_constraint = None,):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """

        super(PINN_import, self).__init__(input_len, output_len, data_input, device, RFF, RFF_sigma, hard_constraint)
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        """
        self.model = torch.load(model_path,  weights_only=False)

class PINN_import_lora(PINN):
    def __init__(self, model_path, input_len, output_len, data_input, device, RFF = False, RFF_sigma = 1., hard_constraint = None, r=4):
        """
        layers: list of integers representing the number of neurons in each layer
        activation: activation function to be used in the hidden layers
        """
        super(PINN_import_lora, self).__init__(input_len, output_len, data_input, device, RFF, RFF_sigma, hard_constraint)
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