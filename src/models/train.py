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

    def __init__(self, model, device, need_loss, lambda_list, nu, alpha, epochs):
        self.model = model
        self.device = device
        self.need_loss = need_loss
        self.nu = nu
        self.alpha = alpha
        self.epochs = epochs

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
            for inputs, targets in batch_data(txy_col, output_data, batch_size, shuffle=True):
                self.loss_obj.new_batch()
                outputs = self.model(inputs)
                # Compute the loss
                self.compute_loss(inputs, outputs, targets)

                optimizer.zero_grad()
                if epoch % 1000 == 0 or epoch == epochs - 1:
                    # Adapting the weights for different losses
                    # Doing it every batch of the epoch
                    if adapting_weight:
                        self.lambda_data, self.lambda_pde, _, _ = self.loss_obj.update_lambda()
                self.loss_obj.backward(retain = False)
                optimizer.step()

            self.loss_history_train.append(self.loss_obj.get_loss_epoch().item() / num_batches)
            if epoch % 1000 == 0 or epoch == epochs - 1:
                loss_test = self.evaluate(xyt_col_test, output_data_test)  # Test loss computation using known data
                self.update_progress_bar(progress_bar, loss_test, num_batches)
                self.loss_history_test.append(loss_test.item())
        return self.model

    def get_loss_history(self):
        return self.loss_history_train, self.loss_history_test

    def update_progress_bar(self, progress_bar, loss_test, num_batches):
        progress_bar.set_postfix(
            loss_train=float(self.loss_obj.get_loss_data()) / num_batches,
            loss_test=loss_test.item(),
            epoch_loss_pde=float(self.loss_obj.get_loss_pde()) / num_batches,
            lambda_data=float(self.lambda_data),
            lambda_pde=float(self.lambda_pde)
        )

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
        - X: Non-dimensionalized input data. (In Numpy)
        - Y: Non-dimensionalized output data.
        """
        file_path = config
        return import_data(file_path, nondim_input, nondim_output)

class Train_Loop_nodata(Train_Loop):
    def __init__(self, model, device):
        super().__init__(model, device, [False, True, True, True], [0., 1., 1., 1.], nu=0.01, alpha=0.9, epochs=10000)
    
    def compute_loss(self, inputs, outputs, targets = None):


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

def train_pinn(model, file_path, device, train_prop = 0.01, nu=0.01, epochs=10000, alpha = 0.9, adapting_weight = True, nondim_input = None, nondim_output = None):
    """
    Train the PINN model using the Navier-Stokes equations.

    Parameters:
    - model: The PINN model to be trained.
    - file_path: Path to the data file.
    """

    loss_history_train = []
    loss_history_test = []
    model.to(device)

    # Load data
    X,Y = import_data(file_path, nondim_input, nondim_output)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=1. - train_prop) ## test_size as a hyperparameter
    del X, Y
    txy_col = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
    output_data = torch.tensor(Y_train, dtype=torch.float32, requires_grad=False).to(device)
    xyt_col_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)
    output_data_test = torch.tensor(Y_test, dtype=torch.float32, requires_grad=False).to(device)
    del X_train, X_test, Y_train, Y_test

    lambda_data = 1.
    lambda_pde = 1.

    # Shuffle data for batching
    batch_size = 4048  # Define batch size
    num_batches = txy_col.size(0) // batch_size + (txy_col.size(0) % batch_size != 0)
    print("num_batches: ", num_batches, "batch_size: ", batch_size)

    # pdb.set_trace()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_obj = Loss_module.Loss(model = model, device = device, nu=nu, alpha=alpha, need_loss=[True, True, False, False], lambda_list=[lambda_data, lambda_pde, 0., 0.])
    progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    for epoch in progress_bar:
        optimizer.zero_grad()
        loss_obj.new_epoch(lambda_data, lambda_pde, 0., 0.)
        for inputs, targets in batch_data(txy_col, output_data, batch_size, shuffle=True):
            loss_obj.new_batch()
            outputs = model(inputs)
            # Compute the loss
            loss_obj.data_loss(outputs, targets)
            loss_obj.pde_loss(inputs, outputs)
            loss_obj.get_total_loss()

            optimizer.zero_grad()
            if epoch % 1000 == 0 or epoch == epochs - 1:
                # Adapting the weights for different losses
                # Doing it every batch of the epoch
                if adapting_weight and False:
                    lambda_data, lambda_pde, _, _ = loss_obj.update_lambda()
            loss_obj.backward(retain = False)
            optimizer.step()

        loss_history_train.append(loss_obj.get_loss_epoch().item() / num_batches)
        if epoch % 1000 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                pred_test = model(xyt_col_test)
            loss_test = nn.MSELoss()(pred_test, output_data_test)  # Test loss computation using known data

            progress_bar.set_postfix(
                loss_train=float(loss_obj.get_loss_data()) / num_batches,
                loss_test=loss_test.item(),
                epoch_loss_pde=float(loss_obj.get_loss_pde()) / num_batches,
                lambda_data=float(lambda_data),
                lambda_pde=float(lambda_pde)
            )
            loss_history_test.append(loss_test.item())
    return model, loss_history_train, loss_history_test
