import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.linalg import inv
from src.utils import *

class KalmanFilter(nn.Module):
    """Kalman filter

    x: observation layer
    z: hidden layer
    """
    def __init__(self, A, B, C, Q, R, latent_size) -> None:
        super().__init__()
        self.A = A.clone()
        self.B = B.clone()
        self.C = C.clone()
        # control input, a list/1d array
        self.latent_size = latent_size
        # covariance matrix of noise
        self.Q = Q
        self.R = R
        
    def projection(self):
        z_proj = torch.matmul(self.A, self.z) + torch.matmul(self.B, self.u)
        P_proj = torch.matmul(self.A, torch.matmul(self.P, self.A.t())) + self.Q
        return z_proj, P_proj

    def correction(self, z_proj, P_proj):
        """Correction step in KF

        K: Kalman gain
        """
        K = torch.matmul(torch.matmul(P_proj, self.C.t()), inv(torch.matmul(torch.matmul(self.C, P_proj), self.C.t()) + self.R))
        self.z = z_proj + torch.matmul(K, self.x - torch.matmul(self.C, z_proj))
        self.P = P_proj - torch.matmul(K, torch.matmul(self.C, P_proj))

    def inference(self, inputs, controls):
        zs = []
        pred_xs = []
        exs = []
        seq_len = inputs.shape[1]
        # initialize mean and covariance estimates of the latent state
        self.z = torch.zeros((self.latent_size, 1)).to(inputs.device)
        self.P = torch.eye(self.latent_size).to(inputs.device)
        for l in range(seq_len):
            self.x = inputs[:, l:l+1]
            self.u = controls[:, l:l+1]
            z_proj, P_proj = self.projection()
            self.correction(z_proj, P_proj)
            zs.append(self.z.detach().clone())
            pred_x = torch.matmul(self.C, self.z)
            pred_xs.append(pred_x)
            exs.append(self.x - pred_x)
        # collect predictions on the observaiton level
        self.pred_xs = torch.cat(pred_xs, dim=1)
        self.exs = torch.cat(exs, dim=1)
        return torch.cat(zs, dim=1)
    

class TemporalPC(nn.Module):
    def __init__(self, control_size, hidden_size, output_size, nonlin='tanh'):
        """A more concise and pytorchy way of implementing tPC

        Suitable for image sequences
        """
        super(TemporalPC, self).__init__()
        self.hidden_size = hidden_size
        self.Win = nn.Linear(control_size, hidden_size, bias=False)
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wout = nn.Linear(hidden_size, output_size, bias=False)

        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")
    
    def forward(self, u, prev_z):
        pred_z = self.Win(self.nonlin(u)) + self.Wr(self.nonlin(prev_z))
        return pred_z

    def init_hidden(self, bsz):
        """This function initializes prev_z"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))

    def update_errs(self, x, u, prev_z, z):
        pred_z = self.forward(u, prev_z)
        self.pred_x = self.Wout(self.nonlin(z))
        err_z = z - pred_z
        err_x = x - self.pred_x
        return err_z, err_x

    def inference(self, inf_iters, inf_lr, x, u, prev_z, sparse_penal=0.5, update_x=False):
        """prev_z should be set up outside the inference, from the previous timestep

        Args:
            train: determines whether we are at the training or inference stage
        
        After every time step, we change prev_z to self.z
        """
        with torch.no_grad():
            # first, initialize the current hidden state as a forward pass
            self.z = self.forward(u, prev_z)

            # update the current hidden state
            for i in range(inf_iters):
                err_z, err_x = self.update_errs(x, u, prev_z, self.z)
                delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone()) + sparse_penal * torch.sign(self.z)
                self.z -= inf_lr * delta_z
                if update_x:
                    delta_x = err_x
                    x -= inf_lr * delta_x

    def update_grads(self, x, u, prev_z):
        """x: input at a particular timestep in stimulus
        
        Could add some sparse penalty to weights
        """
        err_z, err_x = self.update_errs(x, u, prev_z, self.z)
        self.hidden_loss = torch.sum(err_z**2)
        self.obs_loss = torch.sum(err_x**2)
        # self.Win.weight.grad = -torch.matmul(err_z.t(), self.nonlin(u))
        # self.Wr.weight.grad = -torch.matmul(err_z.t(), self.nonlin(prev_z))
        # self.Wout.weight.grad = -torch.matmul(err_x.t(), self.nonlin(self.z))
        loss = self.hidden_loss + self.obs_loss
        loss.backward()


class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlin='tanh'):
        """Simple implementation of a recurrent neural network

        It slightly differs from the mainstream RNN implementation in that
        the weights are applied after nonlinearities so we may use tanh for output
        outside of the range (-1,1)
        """
        super(myRNN, self).__init__()
        self.hidden_size = hidden_size
        self.Win = nn.Linear(input_size, hidden_size, bias=False)
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wout = nn.Linear(hidden_size, output_size, bias=False)

        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")
        
    def forward(self, x, h):
        """We now use the notation for RNNs
        
        x: input at the current step
        h: hidden state
        """
        hidden = self.Win(self.nonlin(x)) + self.Wr(self.nonlin(h))
        output = self.Wout(self.nonlin(hidden))

        return hidden, output
    
    def init_hidden(self, bsz):
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))

                

