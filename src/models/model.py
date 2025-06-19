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
        self.layers3 = []
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

class PINN_LORA(PINN):
    def __init__(self, rank, features, pos_enc, mlp = 'modified_mlp', output_len = 3, RFF = False, RFF_sigma = 1., hard_constraint = None, ):
        """
        We have input_len = 3 as fixed, no RFF
        We don't exploit the collocation points trick, nor the forward AD, it's very slow.
        """
        assert RFF == False, "RFF is not supported in LORA"
        super(PINN_LORA, self).__init__(3, output_len, RFF, RFF_sigma, hard_constraint)

        self.model = LORA(features, rank, output_len, pos_enc, mlp)
class LORA(nn.Module):
    def __init__(self, features, r, out_dim, pos_enc, mlp):
        super().__init__()
        self.features = features
        self.r = r
        self.out_dim = out_dim
        self.pos_enc = pos_enc
        self.mlp_type = mlp
        
        if mlp == 'mlp':
            self.t_mlp = self._make_mlp()
            self.x_mlp = self._make_mlp()
            self.y_mlp = self._make_mlp()
        elif mlp == 'modified_mlp':
            self.t_mlp = self._make_modified_mlp()
            self.x_mlp = self._make_modified_mlp()
            self.y_mlp = self._make_modified_mlp()
        
    def _make_mlp(self):
        layers = []
        for fs in self.features[:-1]:
            layers.append(nn.LazyLinear(fs))
            layers.append(nn.Tanh())
        layers.append(nn.LazyLinear(self.r * self.out_dim))
        return nn.Sequential(*layers)
    
    def _make_modified_mlp(self):
        class ModifiedMLPBlock(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.U = nn.Sequential(nn.LazyLinear(out_features), nn.Tanh())
                self.V = nn.Sequential(nn.LazyLinear(out_features), nn.Tanh())
                self.H = nn.Sequential(nn.LazyLinear(out_features), nn.Tanh())
                self.Z = nn.Sequential(nn.Linear(out_features, out_features), nn.Tanh())
                
            def forward(self, x):
                U = self.U(x)
                V = self.V(x)
                H = self.H(x)
                Z = self.Z(H)
                return (1 - Z) * U + Z * V
        
        layers = []
        for fs in self.features[:-1]:
            layers.append(ModifiedMLPBlock(None, fs))
        layers.append(nn.LazyLinear(self.r * self.out_dim))
        return nn.Sequential(*layers)
    
    def forward(self, input):
        t,x,y = input[:, 0], input[:, 1], input[:, 2]
        if self.pos_enc != 0:
            # positional encoding for spatial coordinates (x and y)
            freq = torch.arange(1, self.pos_enc + 1, 1, device=x.device).unsqueeze(0)
            x = torch.cat([torch.ones(x.shape[0], 1), 
                          torch.sin(x @ freq), 
                          torch.cos(x @ freq)], dim=1)
            y = torch.cat([torch.ones(y.shape[0], 1), 
                          torch.sin(y @ freq), 
                          torch.cos(y @ freq)], dim=1)
            
        # Process each input
        t_out = self.t_mlp(t.unsqueeze(1)).T
        x_out = self.x_mlp(x.unsqueeze(1)).T
        y_out = self.y_mlp(y.unsqueeze(1)).T

            
        outputs = [t_out, x_out, y_out]
        pred = []
        
        for i in range(self.out_dim):
            # ft, fx -> ftx
            tx_i = torch.einsum('ft,fx->ftx', 
                              outputs[0][self.r*i:self.r*(i+1)], 
                              outputs[1][self.r*i:self.r*(i+1)])
            # ftx, fy -> txy
            pred_i = torch.einsum('ftx,fy->txy', 
                                tx_i, 
                                outputs[2][self.r*i:self.r*(i+1)])
            ## Shape (batch_size, batch_size, batch_size), indice (t,x,y)
            # Let's ignore the benefits of SPINN collocation points and take the initial points
            n = pred_i.shape[0]
            idx = torch.arange(n, device = pred_i.device)
            pred_i = pred_i[idx, idx, idx]
            pred.append(pred_i)

        # pdb.set_trace()
        pred = torch.stack(pred, dim = 0).T
        return pred[0] if len(pred) == 1 else pred

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