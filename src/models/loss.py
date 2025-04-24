import torch
import torch.nn as nn

import time
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper


def navier_stokes(txy_col, output, nu = 0.01, non_dim = False):
    """Time-dependent Navier-Stokes PDE residual."""
    Re =  (3.204348e-01) * (1.304893e-01) / 0.01 # U * L / nu

    # PDE Residual Loss
    txy_col.requires_grad_(True)
    u, v, p = output[:, 0], output[:, 1], output[:, 2]

    # First derivatives
    du = torch.autograd.grad(u, txy_col, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dt, du_dx, du_dy = du[:, 0], du[:, 1], du[:, 2]

    dv = torch.autograd.grad(v, txy_col, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dv_dt, dv_dx, dv_dy = dv[:, 0], dv[:, 1], dv[:, 2]

    dp = torch.autograd.grad(p, txy_col, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dx, dp_dy = dp[:, 1], dp[:, 2]

    # Second derivatives
    d2u_dx2 = torch.autograd.grad(du_dx, txy_col, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 1]
    d2u_dy2 = torch.autograd.grad(du_dy, txy_col, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0][:, 2]

    d2v_dx2 = torch.autograd.grad(dv_dx, txy_col, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0][:, 1]
    d2v_dy2 = torch.autograd.grad(dv_dy, txy_col, grad_outputs=torch.ones_like(dv_dy), create_graph=True)[0][:, 2]

    # Continuity equation (∇·u = 0)
    continuity = du_dx + dv_dy
    
    # Momentum equations
    if non_dim:
        momentum_x = du_dt + u * du_dx + v * du_dy + dp_dx - (1/Re) * (d2u_dx2 + d2u_dy2)
        momentum_y = dv_dt + u * dv_dx + v * dv_dy + dp_dy - (1/Re) * (d2v_dx2 + d2v_dy2)
        u_out = du_dx /Re - p
        v_out = dv_dx
    else:
        momentum_x = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
        momentum_y = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)
        u_out = du_dx*nu - p
        v_out = dv_dx

    return continuity, momentum_x, momentum_y, u_out, v_out


class Loss:
    ### An object that we're gonna reset at each epoch
    def __init__(self, model, device, nu = 0.01, alpha = 0.9, non_dim = True, need_loss = [True]*4, lambda_list = [1.]*4):
        self.nu = nu
        self.non_dim = non_dim
        self.model = model
        self.device = device
        self.alpha = alpha

        self.lambda_data = lambda_list[0]
        self.lambda_pde = lambda_list[1]
        self.lambda_boundary = lambda_list[2]
        self.lambda_initial = lambda_list[3]

        self.need_data = need_loss[0]
        self.need_pde = need_loss[1]
        self.need_boundary = need_loss[2]
        self.need_initial = need_loss[3]

        ### Losses for the epoch (so no reset in between batches)
        self.loss_pde = torch.Tensor([0.]).to(self.device)
        self.loss_data = torch.Tensor([0.]).to(self.device)
        self.loss_boundary = torch.Tensor([0.]).to(self.device)
        self.loss_initial = torch.Tensor([0.]).to(self.device)
        self.loss_epoch = torch.Tensor([0.]).to(self.device)

        # for ONE batch
        self.loss_pde_tmp = torch.Tensor([0.]).to(self.device)
        self.loss_data_tmp = torch.Tensor([0.]).to(self.device)
        self.loss_boundary_tmp = torch.Tensor([0.]).to(self.device)
        self.loss_initial_tmp = torch.Tensor([0.]).to(self.device)
        self.loss_epoch_tmp = torch.Tensor([0.]).to(self.device)
        self.loss_total = torch.Tensor([0.]).to(self.device) 

    def new_batch(self):
        self.loss_pde_tmp.zero_()
        self.loss_data_tmp.zero_()
        self.loss_boundary_tmp.zero_()
        self.loss_initial_tmp.zero_()
        self.loss_epoch_tmp.zero_()
        self.loss_total.zero_() 

    def compute_pde_loss(self, txy_col, output, outflow_eq = True):
        continuity, momentum_x, momentum_y, u_out, v_out = navier_stokes(txy_col, output, self.nu, self.non_dim)
        tmp = torch.mean(continuity**2) + torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + outflow_eq * torch.mean(u_out**2) + outflow_eq * torch.mean(v_out**2)
        return tmp.to(self.device)


    def pde_loss(self, txy_col, output, outflow_eq = True):
        assert self.loss_pde_tmp==0 ## otherwise we need to call new_batch before it 
        tmp = self.compute_pde_loss(txy_col, output, outflow_eq)
        self.loss_pde += tmp
        self.loss_pde_tmp = tmp
        return tmp
    
    def compute_data_loss(self, output, target):
        tmp = nn.MSELoss()(output, target).to(self.device)
        return tmp

    def data_loss(self, output, target):
        assert self.loss_data_tmp==0 ## otherwise we need to call new_batch before it

        tmp = self.compute_data_loss(output, target)
        self.loss_data += tmp
        self.loss_data_tmp = tmp
        return tmp
    
    def compute_boundary_loss(self, txy_col, output):
        """
        txy_col: [N, 3] tensor of points in the domain
        output: [N, 3] tensor of outputs at those points
        boundary: [N, 3] tensor of boundary conditions at those points
        """
        boundary = torch.zeros_like(output).to(self.device)
        tmp = nn.MSELoss()(output, boundary).to(self.device)
        return tmp
    
    def boundary_loss(self, txy_col, output):
        assert self.loss_boundary_tmp==0
        tmp = self.compute_boundary_loss(txy_col, output)
        self.loss_boundary += tmp
        self.loss_boundary_tmp = tmp
        return tmp
    
    def compute_initial_loss(self, txy_col, output, target):
        """
        txy_col: [N, 3] tensor of points in the domain
        output: [N, 3] tensor of outputs at those points
        initial: [N, 3] tensor of initial conditions at those points
        """
        tmp = nn.MSELoss()(output, target).to(self.device)
        return tmp

    def initial_loss(self, txy_col, output, target):
        assert self.loss_initial_tmp==0
        tmp = self.compute_initial_loss(txy_col, output, target)
        self.loss_initial += tmp
        self.loss_initial_tmp = tmp
        return tmp

    def get_total_loss(self, pde, data = 0., bc = 0., ic = 0.):
        """
        We should at least have the loss for pde residual (the whole point of PINN)
        """
        assert self.loss_total==0 ## otherwise we need to call new_batch before it

        self.loss_total += (self.lambda_data * data +
                           self.lambda_pde * pde +
                           self.lambda_boundary * bc +
                           self.lambda_initial * ic)
        
        self.loss_epoch += self.loss_total
        return self.loss_total

    def get_total_loss(self):
        assert self.loss_total==0 ## otherwise we need to call new_batch before it
        self.loss_total = (self.lambda_data * self.loss_data_tmp +
                           self.lambda_pde * self.loss_pde_tmp +
                           self.lambda_boundary * self.loss_boundary_tmp +
                           self.lambda_initial * self.loss_initial_tmp)
        self.loss_epoch += self.loss_total
        return self.loss_total
    
    def get_loss_epoch(self):
        return self.loss_epoch

    def get_loss_data(self):
        return self.loss_data
    
    def get_loss_pde(self):
        return self.loss_pde
    
    def get_loss_boundary(self):
        return self.loss_boundary
    
    def get_loss_initial(self):
        return self.loss_initial

    def backward(self, retain = True):
        self.loss_total.backward(retain_graph=retain)
    
    def reset_losses(self):
        self.loss_pde.zero_()
        self.loss_data.zero_()
        self.loss_boundary.zero_()
        self.loss_initial.zero_()
        self.loss_epoch.zero_()
    
    def set_lambda(self, lambda_data = 1., lambda_pde = 1., lambda_boundary = 1., lambda_initial = 1.):
        """
        No input = reset
        """
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_boundary = lambda_boundary
        self.lambda_initial = lambda_initial
    
    def reset_all(self):
        self.reset_loss()
        self.set_lambda()
    
    def new_epoch(self, lambda_data = 1., lambda_pde = 1., lambda_boundary = 1., lambda_initial = 1.):
        self.reset_losses()
        self.set_lambda(lambda_data, lambda_pde, lambda_boundary, lambda_initial)
    
    def set_nu(self, nu):
        self.nu = nu

    def get_grad_loss(self, loss):
        """
        Get the gradient of the loss with respect to the model parameters
        WE CALL ZERO GRAD HERE
        """
        if not loss.grad_fn and not loss.requires_grad:
            return 0.

        loss_grad_data = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
        loss_grad_data = [grad.view(-1) for grad in loss_grad_data if grad is not None]
        loss_grad_data = torch.cat(loss_grad_data)
        return torch.norm(loss_grad_data)

    def update_lambda(self):
        """
        Update the lambda values using the formula in the expert guide paper
        Do it every n (n=1000 typically) epochs
        """
        norm_loss_data = self.get_grad_loss(self.loss_data_tmp)
        norm_loss_pde = self.get_grad_loss(self.loss_pde_tmp)
        norm_loss_boundary = self.get_grad_loss(self.loss_boundary_tmp)
        norm_loss_initial = self.get_grad_loss(self.loss_initial_tmp)

        sum_loss_grad = norm_loss_data + norm_loss_pde + norm_loss_boundary + norm_loss_initial

        if self.need_data:
            lambda_data_hat = sum_loss_grad / norm_loss_data
            self.lambda_data = self.alpha * self.lambda_data + (1 - self.alpha) * lambda_data_hat
        if self.need_pde:
            lambda_pde_hat = sum_loss_grad / norm_loss_pde
            self.lambda_pde = self.alpha * self.lambda_pde + (1 - self.alpha) * lambda_pde_hat
        if self.need_boundary:
            lambda_boundary_hat = sum_loss_grad / norm_loss_boundary
            self.lambda_boundary = self.alpha * self.lambda_boundary + (1 - self.alpha) * lambda_boundary_hat
        if self.need_initial:
            lambda_initial_hat = sum_loss_grad / norm_loss_initial
            self.lambda_initial = self.alpha * self.lambda_initial + (1 - self.alpha) * lambda_initial_hat

        return [self.lambda_data, self.lambda_pde, self.lambda_boundary, self.lambda_initial]