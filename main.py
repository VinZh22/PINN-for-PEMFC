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
from torchinfo import summary

from matplotlib.animation import FuncAnimation

import src.data_process.load_data as load_data
import src.tools.plot_tools as plot_tools
import src.tools.util_func as util_func
from datetime import datetime

def train_and_save(train_obj:train.Train_Loop, device:str, config, save_dir, epochs, non_dim,
                   forward_transform_input, forward_transform_output, inverse_transform_input, inverse_transform_output,
                   train_prop, data_MC, data_shuffle_size, 
                   plot_horiz_axis = 1, plot_vert_axis = 2, plot_depth_axis = 3,
                   sample_mode = "random", optimizer_name = "Adam", additional_name = ""):
    trained_model = train_obj.train_pinn(
        config=config,
        save_path=save_dir,
        train_prop=train_prop,
        nu=0.01,
        epochs=epochs,
        adapting_weight=True,
        additional_name=additional_name,
        train_data_shuffle=data_MC,
        data_shuffle_size= data_shuffle_size,
        sample_mode=sample_mode,
        optimizer_name=optimizer_name,
    )

    os.makedirs(save_dir, exist_ok=True)

    plot_tools.plot_loss_history(train_obj.get_loss_history(), save_dir=save_dir, additional_name="_intermediate_model")

    # Define time and space grids
    inp, _ = load_data.import_data(
        file_path=config,
        df=train_obj.data)
    start_T_sim = int(np.min(inp[:, 0]))  # Start time of simulation
    end_T_sim = int(np.max(inp[:, 0]))  # End time of simulation
    x_min, x_max = float(np.min(inp[:, plot_horiz_axis])), float(np.max(inp[:, plot_horiz_axis]))  # Spatial bounds for x
    y_min, y_max = float(np.min(inp[:, plot_vert_axis])), float(np.max(inp[:, plot_vert_axis]))  # Spatial bounds for y
    x = np.linspace(x_min, x_max, 100)  # Spatial grid (x)
    y = np.linspace(y_min, y_max, 100)  # Spatial grid (y)
    t = np.arange(start_T_sim, end_T_sim+1)   # Time grid

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    sampled_z = None
    if inp.shape[1] == 4:
        sampled_z = np.unique(inp[:, plot_depth_axis])
        sampled_z = np.random.choice(sampled_z)

    plot_tools.plot_speed_map(
        model=trained_model,
        X=X,
        Y=Y,
        t=t,
        save_dir=save_dir,
        device=device,
        non_dim=non_dim,
        forward_transform_input=forward_transform_input,
        inverse_transform_output=inverse_transform_output,
        additional_name="_intermediate_model",
        sample_z=sampled_z,
        axis_order=[plot_horiz_axis, plot_vert_axis, plot_depth_axis]
    )

    train_obj.save_model(
        path=save_dir,
        additional_name="_intermediate_model",
    )

    return trained_model

def force_2D(df):
    z_values = df['Points:2'].values
    z_sampled = z_values[0]
    df = df[df['Points:2'] == z_sampled]
    df = df.drop(columns=['Points:2', 'U:2'])

    return df

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data_dir = os.path.join(args.data_path_file, args.data_name_file)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data file {data_dir} does not exist. Please check the path.")
    if args.bc_geom_file is not None:
        args.bc_geom_file = os.path.join(args.data_path_file, args.bc_geom_file)
        if not os.path.exists(args.bc_geom_file):
            raise FileNotFoundError(f"Boundary condition geometry file {args.bc_geom_file} does not exist. Please check the path.")    

    print(f"Loading data from {data_dir}")
    df = pd.read_csv(data_dir)
    df, time_dependant = load_data.format_df(df)    

    if args.force2D:
        print("Forcing the data to be 2D by removing the z dimension.")
        df = force_2D(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'./results/{timestamp}'

    util_func.save_args(args, save_dir=save_dir)

    non_dim = args.non_dim
    if not non_dim:
        forward_transform_input = None
        forward_transform_output = None
        inverse_transform_input = None
        inverse_transform_output = None
        Re = 1.
        constants = None
    else:
        forward_transform_input, forward_transform_output, inverse_transform_input, inverse_transform_output, Re, constants = util_func.get_ND_non_dim(data_dir, df, args.nu)

    if args.force2D:
        data_input = 3
    else:
        data_input = 4

    geometry_nb = 15 ## HYPERPARAMETER, number of geometry points to use for Lagrangian Topology

    layer = [args.in_dim] + [args.features] * args.n_layers + [args.out_dim]
    if args.model == 'mlp':
        PINN = model.PINN_linear(layer, data_input, RFF = args.RFF, hard_constraint=None, activation=nn.Tanh, device = device, LT = args.LT, LT_nb_geometry=geometry_nb)
    elif args.model == 'modified_mlp':
        PINN = model.PINN_mod_MLP(layer, data_input, RFF = args.RFF, hard_constraint=None, activation=nn.Tanh, device = device, LT = args.LT, LT_nb_geometry=geometry_nb)
    elif args.model == 'saved':
        PINN = model.PINN_import(args.saved_model, input_len=args.in_dim, output_len=args.out_dim, data_input=data_input, RFF = args.RFF, device = device, LT = args.LT, LT_nb_geometry=geometry_nb)
    elif args.model == 'dm_mlp':
        PINN = model.DM_PINN(layer, data_input, RFF = args.RFF, hard_constraint=None, activation=nn.Tanh, device = device, LT = args.LT, LT_nb_geometry=geometry_nb)
    elif args.model == 'pirate_mlp':
        PINN = model.PINN_PirateNet(layer, data_input, RFF = args.RFF, hard_constraint=None, activation=nn.Tanh, device = device, LT = args.LT, LT_nb_geometry=geometry_nb)
    elif args.model == 'saved_lora':
        r = args.rank_lora
        print(f"Using LoRA with rank {r} for the saved model.")
        PINN = model.PINN_import_lora(args.saved_model, r=r, input_len=args.in_dim, output_len=args.out_dim, data_input=data_input, RFF = args.RFF, device = device, LT = args.LT, LT_nb_geometry=geometry_nb)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    summary(PINN, input_size=(args.batch_size, data_input), device=device)

    train_data = train.Train_Loop_data(
        model=PINN,
        device=device,
        batch_size = args.batch_size,
        data=df,
        nondim_input=forward_transform_input,
        nondim_output=forward_transform_output,
        nu = args.nu,
        Re = Re,
        refresh_bar=args.update_iter,
        log_interval=args.log_iter,
        sample_interval=args.shuffle_iter,
        use_bc = args.use_bc,
        file_bc_path=args.bc_geom_file,
        time_dependant = time_dependant
    )

    intermediate_model = train_and_save(
        train_obj=train_data,
        device=device,
        config=data_dir,
        save_dir=save_dir,
        epochs=args.epochs_data,
        non_dim=non_dim,
        forward_transform_input=forward_transform_input,
        forward_transform_output=forward_transform_output,
        inverse_transform_input=inverse_transform_input,
        inverse_transform_output=inverse_transform_output,
        additional_name="_intermediate_model",
        train_prop=args.train_prop,
        data_MC=args.data_MC,
        data_shuffle_size=args.data_MC_n,
        sample_mode=args.data_MC_sample_mode,
        optimizer_name=args.optimizer,
        plot_horiz_axis=args.plot_horiz_axis,
        plot_vert_axis=args.plot_vert_axis,
        plot_depth_axis=args.plot_depth_axis,
    )

    if args.pde_refine:
        print("Finished training the intermediate model. Gonna start refining using pde and equations")

        train_nodata = train.Train_Loop_nodata(
        model=intermediate_model,
        device=device,
        init_path=data_dir,
        data = df,
        batch_size = args.batch_size,
        nu = args.nu,
        Re = Re,
        nondim_input=forward_transform_input,
        nondim_output=forward_transform_output,
        )

        final_model = train_and_save(
            train_obj=train_nodata,
            device=device,
            config=[-20, 30, -20, 20, 0, 1, 150, False, 50],
            save_dir=save_dir,
            epochs=args.epochs_nodata,
            non_dim=non_dim,
            forward_transform_input=forward_transform_input,
            forward_transform_output=forward_transform_output,
            inverse_transform_input=inverse_transform_input,
            inverse_transform_output=inverse_transform_output,
            train_prop = 0.6,
            data_MC=args.data_MC,
            data_shuffle_size=args.data_MC_n,
            sample_mode=args.data_MC_sample_mode,
            optimizer_name=args.optimizer,
        )
    
    else:
        final_model = intermediate_model

    print("Finished training the final model.")

    util_func.save_functions(
        save_dir=save_dir,
        forward_transform_input=forward_transform_input,
        forward_transform_output=forward_transform_output,
        inverse_transform_input=inverse_transform_input,
        inverse_transform_output=inverse_transform_output,
    )
    if not args.skip_difference:
        plot_tools.plot_difference_reference(final_model, device, 
                                            data_path=data_dir, 
                                            save_dir=save_dir, 
                                            df = df,
                                            non_dim=non_dim,
                                            forward_transform_input=forward_transform_input, 
                                            forward_transform_output=forward_transform_output,
                                            inverse_transform_input=inverse_transform_input,
                                            axis_order=[args.plot_horiz_axis, args.plot_vert_axis, args.plot_depth_axis],)
    
    if args.LT:
        plot_tools.plot_Lagrangian_topology(
            model=final_model,
            save_dir=save_dir,
            device=device,
            file_bc_points=args.bc_geom_file,
            inverse_transform_input=inverse_transform_input,
            constants=constants,
        )


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # data directory
    parser.add_argument('--data_path_file', type=str, default="./data/", help='path to the data file')
    parser.add_argument('--data_name_file', type=str, default="cylinder.csv", help='name of the data file')
    parser.add_argument('--force2D', type=bool, default=False, help='whether to force the data to be 2D, i.e., only use x and y coordinates')
    parser.add_argument("--skip_difference", type=bool, default=False, help="whether to skip the difference plot at the end of training")

    # training settings
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SOAP'], help='optimizer to use')
    parser.add_argument('--data_MC', type=bool, default=False, help='whether to use Monte Carlo sampling for data, and cap the sample size to n*batch_size')
    parser.add_argument('--data_MC_n', type=int, default=4, help='number of multiplier per batch_size to sample for MC sampling, only used if data_MC is True')
    parser.add_argument('--data_MC_sample_mode', type=str, default='random', choices=['random', 'time_windowed'], help='sampling mode for MC sampling, only used if data_MC is True')
    parser.add_argument('--non_dim', type=bool, default=False, help='whether to use non-dimensionalization')
    parser.add_argument('--nu', type=float, default=0.01, help='kinematic viscosity')
    parser.add_argument('--batch_size', type=int, default=8192, help='batch size')
    parser.add_argument('--pde_refine', type=bool, default=False, help='whether to use no data to refine the model')
    parser.add_argument('--epochs_data', type=int, default=5000, help='training epochs for data')
    parser.add_argument('--epochs_nodata', type=int, default=2000, help='training epochs for no data')
    parser.add_argument('--use_bc', type=bool, default=False, help='whether to use boundary conditions in the training')
    parser.add_argument('--bc_geom_file', type=str, default=None, help='path to the boundary condition geometry file, only used if use_bc is True')

    # model settings
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'modified_mlp', 'saved', 'saved_lora', 'dm_mlp', 'pirate_mlp'], help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=4, help='the number of layer')
    parser.add_argument('--features', type=int, default=256, help='feature size of each layer')
    parser.add_argument('--RFF', type=bool, default=False, help='whether to use Random Fourier Features')
    parser.add_argument('--LT', type=bool, default=False, help='whether to use Lagrangian Topology')
    parser.add_argument('--in_dim', type=int, default=3, help='size of model input, might not be 3 if using RFF')
    parser.add_argument('--out_dim', type=int, default=3, help='size of model output, might not be 3 if using RFF')
    parser.add_argument('--saved_model', type=str, default=None, help='path to saved model')
    parser.add_argument('--train_prop', type=float, default=0.01, help='proportion of training data')
    parser.add_argument('--rank_lora', type=int, default=25, help='rank for LoRA model, only used if model is saved_lora')

    # log settings
    parser.add_argument('--log_iter', type=int, default=50, help='print log every...')
    parser.add_argument('--update_iter', type=int, default=500, help='update progress bar every...')
    parser.add_argument('--shuffle_iter', type=int, default=4000, help='sample data every... in the case of MC sampling')

    # plotting settings
    parser.add_argument('--plot_horiz_axis', type=int, default=1, help='index of the horizontal axis for plotting (1 for x, 2 for y, 3 for z bcs there is time at 0)')
    parser.add_argument('--plot_vert_axis', type=int, default=2, help='index of the vertical axis for plotting (1 for x, 2 for y, 3 for z)')
    parser.add_argument('--plot_depth_axis', type=int, default=3, help='index of the depth axis for plotting (1 for x, 2 for y, 3 for z)')

    args = parser.parse_args()

    assert args.model != 'saved' or args.saved_model is not None, "Please provide a saved model path if using the saved model option."
    assert args.model != 'saved_lora' or args.saved_model is not None, "Please provide a saved model path if using the saved LoRA model option."
    assert args.use_bc == False or args.bc_geom_file is not None, "Please provide a boundary condition geometry file if using boundary conditions."

    main(args)