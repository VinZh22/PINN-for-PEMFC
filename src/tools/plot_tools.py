import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib.animation import FuncAnimation
from src.data_process.load_data import import_data
from tqdm import tqdm

def evaluate_model(model, test_loader, device, max_batch_size=1024):
    """
    Evaluate the model on test data with optimized batch processing.
    
    Parameters:
    -----------
    model : nn.Module
        The trained model
    test_loader : DataLoader
        DataLoader for test data
    device : str
        Device to use ('cuda' or 'cpu')
    max_batch_size : int
        Maximum number of samples to process in a single forward pass
        
    Returns:
    --------
    float
        Test MSE loss
    """
    model.eval()
    criterion = nn.MSELoss()
    
    # Determine if we should use optimized batch processing
    use_optimized = model.input_length > 1000 or model.output_length > 1000
    
    test_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        if use_optimized:
            # Optimized evaluation for long sequences
            # Collect all batches
            test_batches = []
            for x, y in test_loader:
                test_batches.append((x, y))
            
            # Process in larger combined batches
            for i in range(0, len(test_batches), 4):  # Process 4 batches at a time
                end_idx = min(i + 4, len(test_batches))
                batch_slice = test_batches[i:end_idx]
                
                # Concatenate batches
                x_combined = torch.cat([b[0] for b in batch_slice], dim=0).to(device)
                y_combined = torch.cat([b[1] for b in batch_slice], dim=0).to(device)
                
                # Process in chunks if needed
                chunk_size = min(max_batch_size, x_combined.size(0))
                num_chunks = (x_combined.size(0) + chunk_size - 1) // chunk_size
                
                batch_loss = 0.0
                for j in range(num_chunks):
                    start_idx = j * chunk_size
                    end_idx = min((j + 1) * chunk_size, x_combined.size(0))
                    
                    x_chunk = x_combined[start_idx:end_idx]
                    y_chunk = y_combined[start_idx:end_idx]
                    
                    # Forward pass
                    y_pred = model(x_chunk)
                    
                    # Calculate loss (only on the predicted part)
                    loss = criterion(y_pred, y_chunk[:, model.output_length:])
                    
                    batch_loss += loss.item() * (end_idx - start_idx)
                
                test_loss += batch_loss
                total_samples += x_combined.size(0)
            
            # Calculate average loss
            if total_samples > 0:
                test_loss /= total_samples
        else:
            # Standard evaluation for normal sequences
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                y_pred = model(x)
                
                # Calculate loss (only on the predicted part)
                loss = criterion(y_pred, y)
                
                test_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
            
            # Calculate average loss
            if total_samples > 0:
                test_loss /= total_samples
    
    return test_loss


def plot_loss_history(history, save_dir):
    """
    Plot training and validation loss history.
    
    Parameters:
    -----------
    history : dict
        Training history with keys 'train_loss' and 'val_loss'.
    save_dir : str
        Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Save figure
    plt.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_speed_map(model, X, Y, t, save_dir, device, non_dim=True, forward_transform_input=None, additional_name = ""):
    """
    Plot speed maps over time using the trained model.
    Parameters:
    -----------
    model : nn.Module
        The trained model
    X : np.ndarray
        Meshgrid X coordinates
    Y : np.ndarray
        Meshgrid Y coordinates
    t : np.ndarray
        Time points
    save_dir : str
        Directory to save the plot
    device : str
        Device to use ('cuda' or 'cpu')
    non_dim : bool
        Whether to apply non-dimensionalization
    forward_transform_input : function, optional
        Function to apply non-dimensionalization to input data
    """
    # Initialize lists to store speed (magnitude) at each time step
    speed_maps = []

    for ti in t:
        # Create input tensor: (t, x, y) for all spatial points at time ti
        xy = np.vstack((X.flatten(), Y.flatten())).T
        txy = np.hstack((np.full((len(xy)), ti).reshape(-1, 1), xy))
        if non_dim:
            txy = forward_transform_input(txy)
        txy = torch.tensor(txy, dtype=torch.float32, requires_grad=True).to(device)

        # Predict velocity (u, v) and pressure (p)
        with torch.no_grad():
            uvp_pred = model(txy)
        u_pred = uvp_pred[:, 0].reshape(X.shape)
        v_pred = uvp_pred[:, 1].reshape(X.shape)
        
        u_pred = u_pred.cpu().numpy()
        v_pred = v_pred.cpu().numpy()
        # Compute speed magnitude: sqrt(u^2 + v^2)
        # speed = np.sqrt(u_pred**2 + v_pred**2)
        speed = u_pred
        speed_maps.append(speed)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.contourf(X, Y, speed_maps[0], levels=20, cmap='jet')
    circle = plt.Circle((0, 0), .5, color='white', fill=True, linewidth=2) ## ONE HYPERPARAMETER TO ALLOW CHANGE LATER
    plt.colorbar(cax, label='Speed (m/s)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Speed Map at t = {t[0]:.2f}s')

    def update(frame):
        ax.clear()
        cax = ax.contourf(X, Y, speed_maps[frame], levels=20, cmap='jet')
        ax.add_artist(circle)
        ax.set_title(f'Speed Map at t = {t[frame]:.2f}s')
        return cax

    # Generate animation
    ani = FuncAnimation(fig, update, frames=len(t), interval=100, blit=False)

    # Save as GIF
    ani.save(os.path.join(save_dir,'speed_over_time'+additional_name+'.gif'), writer='pillow', fps=15, dpi=100)
    plt.close()

def plot_difference_reference(model, device, data_path, save_dir, non_dim=True, forward_transform_input=None, forward_transform_output=None):
    """
    Plot for the model the MSE of the space with time, plot at each point in space the average MSE over time and 
    """
    
    X,Y = import_data(data_path, nondim_input=forward_transform_input, nondim_output=forward_transform_output)

    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, requires_grad=True).to(device)
    with torch.no_grad():
        uvp_pred = model(X_tensor)

    # --- First the MSE over the space at each time step
    mse_over_time = []
    time_steps = np.unique(X[:,0]) # it also sorts the time steps
    for ti in time_steps:
        indices = X[:,0] == ti
        pred = uvp_pred[indices]
        target = Y_tensor[indices]
        mse = nn.MSELoss()(pred, target)
        mse_over_time.append(mse.item())
    mse_over_time = np.array(mse_over_time)
    # Plot the MSE over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, mse_over_time, label='MSE over time')
    plt.title('MSE over time')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'mse_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # --- Now the MSE over time at each point in space
    mse_over_space = []
    space = []
    points = np.unique(X[:,1:], axis=0)
    for point in tqdm(points):
        indices = X[:,1:] == point
        indices = indices[:,0] & indices[:,1]
        pred = uvp_pred[indices]
        target = Y_tensor[indices]
        mse = nn.MSELoss()(pred, target)
        mse_over_space.append(mse.item())
        space.append(point)
    mse_over_space = np.array(mse_over_space)
    space = np.array(space)
    # Plot the MSE over space
    plt.figure(figsize=(10, 6))
    plt.scatter(space[:,0], space[:,1], c=mse_over_space, cmap='jet')
    plt.colorbar(label='MSE')
    plt.title('MSE over space')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'mse_over_space.png'), dpi=300, bbox_inches='tight')
    plt.close()




