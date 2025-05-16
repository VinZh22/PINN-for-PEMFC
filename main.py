import argparse
import numpy as np
import src.models.model as model
import src.models.train as train
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os
import pdb

from matplotlib.animation import FuncAnimation

import src.data_process.load_data as load_data
import src.tools.plot_tools as plot_tools
import src.tools.util_func as util_func
from datetime import datetime

def train_and_save_data(model, device, config, save_dir, epochs, non_dim,
                        log_interval, update_interval,
                        forward_transform_input, forward_transform_output):
    train_data = train.Train_Loop_data(
        model=model,
        device=device,
        nondim_input=forward_transform_input,
        nondim_output=forward_transform_output,
        refresh_bar=update_interval,
        log_interval=log_interval,
    )

    intermediate_model = train_data.train_pinn(
        config=config,
        save_path=save_dir,
        train_prop=0.01,
        nu=0.01,
        epochs=epochs,
        adapting_weight=True,
        additional_name="_intermediate_model",
    )

    os.makedirs(save_dir, exist_ok=True)

    # Define time and space grids
    end_T_sim = 150

    x = np.linspace(-20, 30, 100)  # Spatial grid (x)
    y = np.linspace(-20, 20, 100)  # Spatial grid (y)
    t = np.linspace(0, end_T_sim, end_T_sim)   # Time grid

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    plot_tools.plot_speed_map(
        model=intermediate_model,
        X=X,
        Y=Y,
        t=t,
        save_dir=save_dir,
        device=device,
        non_dim=non_dim,
        forward_transform_input=forward_transform_input,
        additional_name="_intermediate_model"
    )

    loss_history_train, loss_history_test = train_data.get_loss_history()
    plot_tools.plot_loss_history({"train_loss": loss_history_train, "val_loss": loss_history_test}, save_dir=save_dir, additional_name="_intermediate_model")

    train_data.save_model(
        path=save_dir,
        additional_name="_intermediate_model",
    )

    return intermediate_model

def train_and_save_nodata(model, device, config, save_dir, epochs, non_dim, forward_transform_input, forward_transform_output):
    train_obj = train.Train_Loop_nodata(
        model=model,
        device=device,
        init_path="./data/cylinder.csv",
        nondim_input=forward_transform_input,
        nondim_output=forward_transform_output,
    )

    final_model = train_obj.train_pinn(
        config=config,
        save_path=save_dir,
        train_prop=0.4,
        nu=0.01,
        epochs=epochs,
        adapting_weight=True
    )

    loss_history_train, loss_history_test = train_obj.get_loss_history()

    # Define time and space grids
    end_T_sim = 150

    x = np.linspace(-20, 30, 100)  # Spatial grid (x)
    y = np.linspace(-20, 20, 100)  # Spatial grid (y)
    t = np.linspace(0, end_T_sim, end_T_sim)   # Time grid

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    os.makedirs(save_dir, exist_ok=True)
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

    return final_model

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'./results/{timestamp}'

    util_func.save_args(args, save_dir=save_dir)

    non_dim = args.non_dim
    if not non_dim:
        forward_transform_input = None
        forward_transform_output = None
    else:
        forward_transform_input, forward_transform_output = util_func.get_non_dim_transform()


    layer = [args.in_dim] + [args.features] * args.n_layers + [3]
    if args.model == 'mlp':
        PINN = model.PINN_linear(layer, RFF = args.RFF, hard_constraint=None, activation=nn.Tanh, device = device)
    elif args.model == 'modified_mlp':
        PINN = model.PINN_mod_MLP(layer, RFF = args.RFF, hard_constraint=None, activation=nn.Tanh, device = device)

    intermediate_model = train_and_save_data(
        model=PINN,
        device=device,
        config=data_dir,
        save_dir=save_dir,
        epochs=args.epochs_data,
        non_dim=non_dim,
        forward_transform_input=forward_transform_input,
        forward_transform_output=forward_transform_output,
        update_interval=args.update_iter,
        log_interval=args.log_iter
    )

    if args.pde_refine:
        print("Finished training the intermediate model. Gonna start refining using pde and equations")

        final_model = train_and_save_nodata(
            model=intermediate_model,
            device=device,
            config=[-20, 30, -20, 20, 0, 1, 150, False, 50],
            save_dir=save_dir,
            epochs=args.epochs_nodata,
            non_dim=non_dim,
            forward_transform_input=forward_transform_input,
            forward_transform_output=forward_transform_output
        )
    
    else:
        final_model = intermediate_model

    print("Finished training the final model.")

    plot_tools.plot_difference_reference(final_model, device, 
                                        data_path=data_dir, 
                                        save_dir=save_dir, 
                                        forward_transform_input=forward_transform_input, 
                                        forward_transform_output=forward_transform_output)


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # data directory
    parser.add_argument('--data_dir', type=str, default="./data/cylinder.csv", help='a directory to reference solution')

    # training settings
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--non_dim', type=bool, default=False, help='whether to use non-dimensionalization')
    # parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--pde_refine', type=bool, default=False, help='whether to use no data to refine the model')
    parser.add_argument('--epochs_data', type=int, default=5000, help='training epochs for data')
    parser.add_argument('--epochs_nodata', type=int, default=2000, help='training epochs for no data')

    # model settings
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'modified_mlp', 'saved'], help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=4, help='the number of layer')
    parser.add_argument('--features', type=int, default=256, help='feature size of each layer')
    parser.add_argument('--RFF', type=bool, default=False, help='whether to use Random Fourier Features')
    parser.add_argument('--in_dim', type=int, default=3, help='size of model input, might not be 3 if using RFF')
    parser.add_argument('--saved_model', type=str, default=None, help='path to saved model')

    # log settings
    parser.add_argument('--log_iter', type=int, default=50, help='print log every...')
    parser.add_argument('--update_iter', type=int, default=500, help='update progress bar every...')

    args = parser.parse_args()

    assert args.model != 'saved' or args.saved_model is not None, "Please provide a saved model path if using the saved model option."

    main(args)