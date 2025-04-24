from src.data_process.load_data import import_data

import src.models.loss as Loss_module
import src.models.model as model
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import pdb


class Train_Loop(ABC):
    """
    Abstract class for training loops.
    """

    def __init__(self, model, device, need_loss, lambda_list, nu, alpha, epochs, refresh_bar = 200):
        self.model = model
        self.device = device
        self.need_loss = need_loss
        self.nu = nu
        self.alpha = alpha
        self.epochs = epochs
        self.refresh_bar = refresh_bar

        self.lambda_data = lambda_list[0]
        self.lambda_pde = lambda_list[1]
        self.lambda_boundary = lambda_list[2]
        self.lambda_initial = lambda_list[3]

        self.loss_obj = Loss_module.Loss(model = model, device = device, nu=nu, alpha=alpha, need_loss=need_loss, lambda_list=lambda_list)
        self.loss_history_train = []
        self.loss_history_test = []
    

    def train_pinn(self, config, train_prop = 0.01, nu=0.01, epochs=10000, alpha = 0.9, adapting_weight = True, nondim_input = None, nondim_output = None):
        """
        Train the PINN model using the Navier-Stokes equations.

        Parameters:
        - model: The PINN model to be trained.
        - file_path: Path to the data file.
        """

        device = self.device
        self.model.to(device)

        # Load data
        X,Y = self.import_data(config, nondim_input, nondim_output)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=1. - train_prop) ## test_size as a hyperparameter
        del X, Y
        txy_col = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
        output_data = torch.tensor(Y_train, dtype=torch.float32, requires_grad=False).to(device)
        xyt_col_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)
        output_data_test = torch.tensor(Y_test, dtype=torch.float32, requires_grad=False).to(device)
        del X_train, X_test, Y_train, Y_test

        # Shuffle data for batching
        batch_size = 4048  # Define batch size
        num_batches = txy_col.size(0) // batch_size + (txy_col.size(0) % batch_size != 0)
        print("num_batches: ", num_batches, "batch_size: ", batch_size)

        # pdb.set_trace()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
        for epoch in progress_bar:
            optimizer.zero_grad()
            self.loss_obj.new_epoch(self.lambda_data, self.lambda_pde, 0., 0.)
            for inputs, targets in batch_data(txy_col, output_data, batch_size, shuffle=False):
                self.loss_obj.new_batch()
                outputs = self.model(inputs)
                # Compute the loss
                if torch.isnan(self.loss_obj.loss_pde):
                    pdb.set_trace()
                self.compute_loss(inputs, outputs, targets)

                optimizer.zero_grad()
                if epoch % self.refresh_bar == 0 or epoch == epochs - 1:
                    # Adapting the weights for different losses
                    # Doing it every batch of the epoch
                    if adapting_weight:
                        self.lambda_data, self.lambda_pde, self.lambda_boundary, self.lambda_initial = self.loss_obj.update_lambda()
                self.loss_obj.backward(retain = False)
                optimizer.step()

            self.loss_history_train.append(self.loss_obj.get_loss_epoch().item() / num_batches)
            if epoch % self.refresh_bar == 0 or epoch == epochs - 1:
                loss_test = self.evaluate(xyt_col_test, output_data_test)  # Test loss computation using known data
                self.update_progress_bar(progress_bar, loss_test, num_batches)
                self.loss_history_test.append(loss_test.item())
        return self.model

    def get_loss_history(self):
        return self.loss_history_train, self.loss_history_test

    @abstractmethod
    def update_progress_bar(self, progress_bar, loss_test, num_batches):
        pass

    @abstractmethod
    def compute_loss(self, inputs, outputs, targets = None):
        pass
    
    @abstractmethod
    def evaluate(self, input_test, output_test = None):
        pass

    @abstractmethod
    def import_data(self, config, nondim_input = None, nondim_output = None):
        # config can be any type of variable, depending on the training desired
        pass

class Train_Loop_data(Train_Loop):
    """
    Class for training loops with data.
    """

    def __init__(self, model, device):
        super().__init__(model, device, [True, True, False, False], [1., 1., 0., 0.], nu=0.01, alpha=0.9, epochs=10000)

    def compute_loss(self, inputs, outputs, targets):
        """
        Compute the loss for the given inputs and targets.

        Parameters:
        - inputs: The input data.
        - outputs: The model predictions.
        - targets: The target data.

        Returns:
        - loss: The computed loss.
        """
        self.loss_obj.data_loss(outputs, targets)
        self.loss_obj.pde_loss(inputs, outputs)
        self.loss_obj.get_total_loss()

    def evaluate(self, input_test, output_test = None):
        with torch.no_grad():
            pred_test = self.model(input_test)
        loss_test = nn.MSELoss()(pred_test, output_test)  # Test loss computation using known data
        return loss_test
    
    def import_data(self, config, nondim_input = None, nondim_output = None):
        """
        Import data from a CSV file and apply non-dimensionalization if needed.
        Parameters:
        - config: Path to the CSV file.
        - nondim_input: Function to apply non-dimensionalization to input data.
        - nondim_output: Function to apply non-dimensionalization to output data.
        Returns:
        - X: Non-dimensionalized input data. (In Numpy (t,x,y))
        - Y: Non-dimensionalized output data.
        """
        file_path = config
        return import_data(file_path, nondim_input, nondim_output)

class Train_Loop_nodata(Train_Loop):
    def __init__(self, model, device):
        super().__init__(model, device, [False, False, True, False], [0., 0., 1., 0.], nu=0.01, alpha=0.9, epochs=10000)
        

        ## TO DO along with compute_loss for BC
        x = 0.5*torch.cos(torch.linspace(0, 2*3.14, 50)).reshape(-1,1)
        y = 0.5*torch.sin(torch.linspace(0, 2*3.14, 50)).reshape(-1,1)
        t = torch.arange(0, 150.).reshape(-1,1)
        X, Y, T = np.meshgrid(x, y, t)
        self.BC_geom = np.vstack((T.flatten(), X.flatten(), Y.flatten())).T
        self.BC_geom = torch.tensor(self.BC_geom, dtype=torch.float32, requires_grad=True).to(device)
    
    def compute_loss(self, inputs, outputs, targets = None):
        # assert targets is None, "Targets should be None for no data training"
        self.loss_obj.pde_loss(inputs, outputs)

        ## TO DO : make it cleaner and integrated in the framework
        ## FOR NOW just use predefined boundary conditions
        self.loss_obj.boundary_loss(self.BC_geom, outputs)

        self.loss_obj.get_total_loss()

    
    def evaluate(self, input_test, output_test=None):
        """
        Don't give anything to output_test, it's just to comply with the interface
        """
        # assert output_test is None, "Output test should be None for no data training"
        # pred_test = self.model(input_test) # no torch.no_grad() because we need to compute the loss
        # loss_test = self.loss_obj.compute_pde_loss(input_test, pred_test)  # Test loss computation using known data
        ## will need to batchify
        return torch.Tensor([0.]).to(self.device)  # Dummy loss, as we don't have any test data
    
    def import_data(self, config, nondim_input=None, nondim_output=None):
        """
        Because we won't use data, it is going to be created
        Return (t,x,y) or (t,x,y,z) depending on the need
        """
        x_min, x_max, y_min, y_max, z_min, z_max, t_max, need_3D, nb_div = config
        # Create a grid of points in the domain
        x = np.linspace(x_min, x_max, nb_div)
        y = np.linspace(y_min, y_max, nb_div)
        t = np.arange(0, t_max)
        if need_3D:
            z = np.linspace(z_min, z_max, nb_div)
            X, Y, Z, T = np.meshgrid(x, y, z, t)

            return np.vstack((T.flatten(), X.flatten(), Y.flatten(), Z.flatten())).T, np.zeros((nb_div**3 *t_max, 3))
        else:
            X, Y, T = np.meshgrid(x, y, t)
            return np.vstack((T.flatten(), X.flatten(), Y.flatten())).T, np.zeros((nb_div**2 *t_max, 3))

    def update_progress_bar(self, progress_bar, loss_test, num_batches):
        progress_bar.set_postfix(
            loss_pde=float(self.loss_obj.get_loss_pde()) / num_batches,
            loss_bc=float(self.loss_obj.get_loss_boundary()) / num_batches,
            lambda_pde=float(self.lambda_pde),
            lambda_bc=float(self.lambda_boundary),
        )

def batch_data(X,Y, batch_size, shuffle=True):
    n_samples = X.shape[0]
    indices = torch.arange(n_samples, device=X.device)

    if shuffle:
        indices = indices[torch.randperm(n_samples, device=X.device)]
    res = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        res.append([X[batch_idx], Y[batch_idx]])    
    return res
