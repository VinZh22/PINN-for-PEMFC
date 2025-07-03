from src.data_process.load_data import import_data

import src.models.loss as Loss_module
import src.models.model as model
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from abc import ABC, abstractmethod
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.tools.plot_tools import evaluate_model
from src.tools.util_func import save_functions
from torch.optim.lr_scheduler import ExponentialLR
from src.tools.soap import SOAP

import pdb


class Train_Loop(ABC):
    """
    Abstract class for training loops.
    """

    def __init__(self, model:model.PINN, device, need_loss, lambda_list, 
                 nu, Re, alpha, batch_size, data:pd.DataFrame,
                 nondim_input = None, nondim_output = None, 
                 refresh_bar = 500, log_interval = 50, sample_interval = 1000):


        self.model = model
        self.device = device
        self.need_loss = need_loss
        self.nu = nu
        self.Re = Re
        self.alpha = alpha
        self.refresh_bar = refresh_bar
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.nondim_input = nondim_input
        self.nondim_output = nondim_output
        self.batch_size = batch_size 
        self.data = data

        self.lambda_data = lambda_list[0]
        self.lambda_pde = lambda_list[1]
        self.lambda_boundary = lambda_list[2]
        self.lambda_initial = lambda_list[3]

        self.loss_obj = Loss_module.Loss(model = model, device = device, nu=nu, Re=Re, alpha=alpha, need_loss=need_loss, lambda_list=lambda_list)
        self.loss_history_PDE = []
        self.loss_history_data = []
        self.loss_history_boundary = []
        self.loss_history_initial = []
        self.loss_history_test = []
        
        self.time_to_complete = f"Total time: {0.:.2f}s | Avg speed: {0.:.2f} it/s"
    
    def set_data(self, config, train_prop):
        """
        Set the data for the training loop.

        Parameters:
        - config: Path to the data file.
        - nondim_input: Function to apply non-dimensionalization to input data.
        - nondim_output: Function to apply non-dimensionalization to output data.
        """
        device = self.device
        X, Y = self.import_data(config, self.nondim_input, self.nondim_output)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, train_size=train_prop) ## test_size as a hyperparameter
        del X, Y
        txy_col = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
        output_data = torch.tensor(Y_train, dtype=torch.float32, requires_grad=False).to(device)
        xyt_col_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)
        output_data_test = torch.tensor(Y_test, dtype=torch.float32, requires_grad=False).to(device)
        del X_train, X_test, Y_train, Y_test

        return txy_col, output_data, xyt_col_test, output_data_test

    def train_pinn(self, config, save_path, 
                   train_prop = 0.01, nu=0.01, epochs=10000, adapting_weight = True, 
                   train_data_shuffle = False, data_shuffle_size = 4, sample_mode = "random", optimizer_name = "Adam",
                   additional_name = ""):
        """
        Train the PINN model using the Navier-Stokes equations.

        Parameters:
        - model: The PINN model to be trained.
        - file_path: Path to the data file.
        """
        device = self.device
        batch_size = self.batch_size
        self.model.to(device)
        # Load data
        txy_col, output_data, xyt_col_test, output_data_test = self.set_data(config, train_prop)

        # If data shuffle, prepare the parameters
        id_time_window = 0 ## only used if sample_mode is "time_windowed"
        time_max = txy_col[:, 0].max().item()
        time_min = txy_col[:, 0].min().item()
        time_ampl = time_max - time_min
        divide_size = epochs // self.sample_interval
        divide_size = max(divide_size, 1)  # Ensure divide_size is at least 1 to avoid division by zero
        time_step_size = time_ampl / divide_size
        if train_data_shuffle:
            # Shuffle the data
            size_data_epoch = data_shuffle_size*self.batch_size

            print(f"Shuffling data {size_data_epoch} through {txy_col.size(0)}")

            total_data_input = txy_col.clone().to(device)
            total_data_output = output_data.clone().to(device)

            if sample_mode == "random":
                idx = torch.randperm(total_data_input.size(0), device=device)[:size_data_epoch]
            elif sample_mode == "time_windowed":
                print(f'Each time window will be of size {time_step_size} and we will have {divide_size} time windows')
                idx = (total_data_input[:, 0] < time_min + (id_time_window+1) * time_step_size)
                idx = torch.where(idx)[0]
                sample_idx = torch.randperm(idx.size(0), device=device)[:size_data_epoch]
                idx = idx[sample_idx]
            else:
                raise ValueError(f"Unknown sample mode: {sample_mode}")
            txy_col = total_data_input[idx]
            output_data = total_data_output[idx]
            txy_col = txy_col.detach().requires_grad_(True)
            output_data = output_data.detach()

        else:
            size_data_epoch = txy_col.size(0)

        # Prepare for batching
        num_batches = size_data_epoch // batch_size + (size_data_epoch % batch_size != 0)
        print("num_batches: ", num_batches, "batch_size: ", batch_size)
        
        ## cte for optim and scheduler (not hyper parameters for now)
        lr = 1e-3
        decay = 0.1
        decay_steps = 2000

        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "SOAP":
            optimizer = SOAP(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        scheduler = ExponentialLR(optimizer, gamma=(1 - decay/decay_steps))
        progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

        loss_log: list[dict[str, float]] = []

        for epoch in progress_bar:
            # pdb.set_trace()
            optimizer.zero_grad()
            if epoch!=0 and (epoch % self.log_interval == 0 or epoch == epochs - 1):
                self.add_log_entry(loss_log, epoch, num_batches)
            self.loss_obj.new_epoch(self.lambda_data, self.lambda_pde, self.lambda_boundary, self.lambda_initial)
            tmp_loader = batch_data(txy_col, output_data, batch_size, shuffle=False)
            for inputs, targets in tmp_loader:
                self.loss_obj.new_batch()
                outputs = self.model(inputs)
                # Compute the loss
                self.compute_loss(inputs, outputs, targets)

                optimizer.zero_grad()
                if epoch % self.refresh_bar == 0 or epoch == epochs - 1:
                    # Adapting the weights for different losses
                    # Doing it every batch of the epoch, to allow a backward without retaining the graph
                    if adapting_weight:
                        self.lambda_data, self.lambda_pde, self.lambda_boundary, self.lambda_initial = self.loss_obj.update_lambda()
                self.loss_obj.backward(retain = False)
                optimizer.step()
                scheduler.step()
            self.loss_history_PDE.append(self.loss_obj.get_loss_pde() / num_batches)
            if self.lambda_data > 0.:
                self.loss_history_data.append(self.loss_obj.get_loss_data() / num_batches)
            if self.lambda_boundary > 0.:
                self.loss_history_boundary.append(self.loss_obj.get_loss_boundary() / num_batches)
            if self.lambda_initial > 0.:
                self.loss_history_initial.append(self.loss_obj.get_loss_initial() / num_batches)

            if epoch % self.refresh_bar == 0 or epoch == epochs - 1:
                test_loader = batch_data(xyt_col_test, output_data_test, batch_size, shuffle=False)
                loss_test = self.evaluate(test_loader)  # Test loss computation using known data
                self.update_progress_bar(progress_bar, loss_test, num_batches)
                self.loss_history_test.append((epoch,float(loss_test)))
                self.save_log(save_path, loss_log, additional_name)

            if epoch % self.sample_interval == 0:
                ## Let's shuffle another time the data
                if train_data_shuffle:
                    if sample_mode == "random":
                        idx = torch.randperm(total_data_input.size(0), device=device)[:size_data_epoch]
                    elif sample_mode == "time_windowed":
                        id_time_window +=1
                        print(f"Current id_time_window: {id_time_window} out of {divide_size}")
                        idx = (total_data_input[:, 0] < time_min + (id_time_window+1) * time_step_size)
                        idx = torch.where(idx)[0]
                        sample_idx = torch.randperm(idx.size(0), device=device)[:size_data_epoch]
                        idx = idx[sample_idx]
                    else:
                        raise ValueError(f"Unknown sample mode: {sample_mode}")
                    txy_col = total_data_input[idx]
                    output_data = total_data_output[idx]
                    txy_col = txy_col.detach().requires_grad_(True)
                    output_data = output_data.detach()
        

        self.loss_history_test = np.array(self.loss_history_test)
        self.loss_history_PDE = np.array(self.loss_history_PDE)
        self.loss_history_data = np.array(self.loss_history_data)
        self.loss_history_boundary = np.array(self.loss_history_boundary)
        self.loss_history_initial = np.array(self.loss_history_initial)

        self.time_to_complete = f"Total time: {progress_bar.format_dict["elapsed"]:.2f}s | Avg speed: {epochs/progress_bar.format_dict["elapsed"]:.2f} it/s"
        self.save_log(save_path, loss_log, additional_name)
        self.save_completed_stat(save_path, additional_name)

        pdb.set_trace()

        return self.model

    def get_loss_history(self):
        losses = {"PDE Loss (on training points)": self.loss_history_PDE, "Test loss": self.loss_history_test}
        if self.loss_history_data.shape[0] > 0:
            losses["Data Loss (on training points)"] = self.loss_history_data
        if self.loss_history_boundary.shape[0] > 0:
            losses["Boundary Loss (on training points)"] = self.loss_history_boundary
        if self.loss_history_initial.shape[0] > 0:
            losses["Initial Condition Loss (on training points)"] = self.loss_history_initial
        return losses
    
    def save_model(self, path, additional_name = ""):
        """
        Save the model to a file.
        """
        torch.save(self.model.model, os.path.join(path,'model' + additional_name + '.pth'))
    
    def save_model_weights(self, path):
        """
        Save the model to a file.
        """
        torch.save(self.model.state_dict(), os.path.join(path,'model_weights.pth'))

    def add_log_entry(self, log, epoch, num_batches):
        """
        Add an entry to the log.
        """
        log.append({
            'epoch': epoch,
            'loss_pde': float(self.loss_obj.get_loss_pde())/num_batches,
            'loss_data': float(self.loss_obj.get_loss_data())/num_batches,
            'loss_boundary': float(self.loss_obj.get_loss_boundary())/num_batches,
            'loss_initial': float(self.loss_obj.get_loss_initial())/num_batches,
            'lambda_pde': float(self.lambda_pde),
            'lambda_data': float(self.lambda_data),
            'lambda_boundary': float(self.lambda_boundary),
            'lambda_initial': float(self.lambda_initial),
        })

    def save_log(self, path, log, additional_name = ""):
        """
        Save the losses to a file.
        """
        os.makedirs(path, exist_ok=True)
        pd.DataFrame(log).to_csv(os.path.join(path, "log" + additional_name + ".csv"), index=False)

    def save_completed_stat(self, path, additional_name = ""):
        """
        Save the time to complete the training.
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'completed_stat'+ additional_name+'.txt'), 'w') as f:
            f.write(self.time_to_complete)
            f.write("\n")
            f.write(f"loss_pde: {self.loss_obj.get_loss_pde()}\n")
            f.write(f"loss_data: {self.loss_obj.get_loss_data()}\n")
            f.write(f"loss_boundary: {self.loss_obj.get_loss_boundary()}\n")
            f.write(f"loss_initial: {self.loss_obj.get_loss_initial()}\n")
            f.write(f"loss_test: {self.loss_history_test[-1]}\n")
    
    def set_condition_data(self):
        radius = 0.99
        x = radius*torch.cos(torch.linspace(0, 2*3.14, 50)).reshape(-1,1)
        y = radius*torch.sin(torch.linspace(0, 2*3.14, 50)).reshape(-1,1)
        t = torch.arange(0, 150.).reshape(-1,1)
        X, Y, T = np.meshgrid(x, y, t)
        self.BC_geom = np.vstack((T.flatten(), X.flatten(), Y.flatten())).T
        if self.nondim_input is not None:
            self.BC_geom = self.nondim_input(self.BC_geom)
        self.BC_geom = torch.tensor(self.BC_geom, dtype=torch.float32, requires_grad=True).to(self.device)

        x = -20. * torch.ones((50,1))
        y = torch.linspace(-20, 20, 50).reshape(-1,1)
        t = torch.arange(0, 150.).reshape(-1,1)
        X, Y, T = np.meshgrid(x, y, t)
        self.BC_geom2 = np.vstack((T.flatten(), X.flatten(), Y.flatten())).T
        if self.nondim_input is not None:
            self.BC_geom2 = self.nondim_input(self.BC_geom2)
        self.BC_geom2 = torch.tensor(self.BC_geom2, dtype=torch.float32, requires_grad=True).to(self.device)

    @abstractmethod
    def update_progress_bar(self, progress_bar, loss_test, num_batches):
        pass

    @abstractmethod
    def compute_loss(self, inputs, outputs, targets = None):
        pass
    
    @abstractmethod
    def evaluate(self, test_loader):
        pass

    @abstractmethod
    def import_data(self, config, nondim_input = None, nondim_output = None):
        # config can be any type of variable, depending on the training desired
        pass

class Train_Loop_data(Train_Loop):
    """
    Class for training loops with data.
    """

    def __init__(self, model:model.PINN, device, batch_size, data, nondim_input = None, nondim_output = None, nu=0.01, Re = 100., alpha=0.9, refresh_bar=500, log_interval=50, sample_interval=1000):
        super().__init__(model, device,
                         [True, True, False, False], [1., 1., 0., 0.], refresh_bar=refresh_bar, log_interval=log_interval, sample_interval=sample_interval,
                         nu=nu, Re = Re, alpha=alpha, batch_size=batch_size, data=data, nondim_input=nondim_input, nondim_output=nondim_output,)

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
        self.loss_obj.pde_loss(inputs, outputs, enhanced_gradient=False) ### THERE decide or not to use enhanced gPINN
        return self.loss_obj.get_total_loss()

    def evaluate(self, test_loader):
        return evaluate_model(self.model, test_loader, self.device, self.batch_size)
    
    def import_data(self, config:tuple, nondim_input = None, nondim_output = None):
        """
        Import data from a CSV file and apply non-dimensionalization if needed.
        Parameters:
        - config: Path to the CSV file and a DataFrame containing the data.
        - nondim_input: Function to apply non-dimensionalization to input data.
        - nondim_output: Function to apply non-dimensionalization to output data.
        Returns:
        - X: Non-dimensionalized input data. (In Numpy (t,x,y))
        - Y: Non-dimensionalized output data.
        """
        file_path = config
        return import_data(file_path, self.data, nondim_input, nondim_output)
    
    def update_progress_bar(self, progress_bar, loss_test, num_batches):
        progress_bar.set_postfix(
            loss_pde=float(self.loss_obj.get_loss_pde()) / num_batches,
            loss_data=float(self.loss_obj.get_loss_data()) / num_batches,
            loss_test = float(loss_test),
            lambda_pde=float(self.lambda_pde),
            lambda_data=float(self.lambda_data),
        )

class Train_Loop_nodata(Train_Loop):
    def __init__(self, model, device, init_path, data, batch_size, nondim_input = None, nondim_output = None, nu=0.01, Re = 100., alpha=0.9):
        super().__init__(model, device, 
                         [False, True, True, True], [0., 0., 1., 0.], data=data,
                         nu=nu, Re = Re, alpha=alpha, batch_size=batch_size, nondim_input=nondim_input, nondim_output=nondim_output,)
        
        ## TO DO along with compute_loss for BC
        self.set_condition_data()
        X,Y = import_data(init_path, data) ## we don't put non-dim function because we don't need to apply it to the whole data
        if nondim_input is not None:
            indices = X[:,0] == nondim_input([1,0,0])[0] # get the indices of the points where t = 1
        else:
            indices = X[:,0] <= 3 # get the indices of the points where t = 1
        self.init_data = X[indices]
        self.init_data = torch.tensor(self.init_data, dtype=torch.float32, requires_grad=True).to(device)
        self.init_target = Y[indices]
        self.init_target = torch.tensor(self.init_target, dtype=torch.float32, requires_grad=False).to(device)
    
    def compute_loss(self, inputs, outputs, targets = None):
        # assert targets is None, "Targets should be None for no data training"
        self.loss_obj.pde_loss(inputs, outputs, enhanced_gradient=False)

        ## TO DO : make it cleaner and integrated in the framework
        ## FOR NOW just use predefined boundary conditions
        outputs_circle = self.model(self.BC_geom)
        circle_bc = torch.zeros_like(outputs_circle).to(self.device)
        outputs_line = self.model(self.BC_geom2)
        line_bc = torch.ones_like(outputs_line).to(self.device)
        outputs = torch.cat((outputs_circle, outputs_line), dim=0)
        boundary = torch.cat((circle_bc, line_bc), dim=0)
        self.loss_obj.boundary_loss(outputs, boundary)
        outputs = self.model(self.init_data)
        self.loss_obj.initial_loss(self.init_data, outputs, self.init_target)

        self.loss_obj.get_total_loss()

    
    def evaluate(self, test_loader):
        """
        Don't give anything to output_test, it's just to comply with the interface
        """
        loss_test = 0.
        size_test = 0
        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            pred_test = self.model(inputs) # no torch.no_grad() because we need to compute the loss
            loss_test += self.loss_obj.compute_pde_loss(inputs, pred_test, eval=True, enhanced_gradient=False) * inputs.shape[0]
            size_test += inputs.shape[0]
        loss_test /= size_test
        return loss_test
    
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
            loss_init=float(self.loss_obj.get_loss_initial()) / num_batches,
            lambda_pde=float(self.lambda_pde),
            lambda_bc=float(self.lambda_boundary),
            lambda_ic=float(self.lambda_initial),
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
