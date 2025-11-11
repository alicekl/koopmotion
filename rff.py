'''
Defines the lifting function, that lifts our states into a space where the dynamics now evolve linearly
'''

import numpy as np
import torch
from torch import nn
import torch.nn.init as init

class LearnedFF(nn.Module):
    '''
    This model lifts its inputs (X) to a higher dimensional space, 
    of dimension (rff_n + states_n, inputs_n). 
    The lifting function is parameterized by by Fourier features,
    defined by learnable weights and biases.
    
    For each Fourier feature we have a weight or phase (W), 
    which is a vector matching the dimension of the number of states of the system
    and a corresponding bias or phase shift (b), which is just one constant.
    '''
        
    def __init__(self, append_inputs=True, rff_n=100, states_n=2):
        super().__init__()
        
        
        self.append_inputs = append_inputs
        self.rff_n = rff_n 
        self.states_n = states_n 
        
        self.b = nn.Parameter(torch.empty((self.rff_n, 1), requires_grad=True)) 
        self.W = nn.Parameter(torch.empty((self.rff_n, states_n), requires_grad=True))
                
        # Xavier uniform initialization worked well here over others 
        init.xavier_uniform_(self.b)
        init.xavier_uniform_(self.W)

    def forward(self, X):
        states_n, inputs_n = X.shape

        # Note that the same set of weights (W), and bias (b) are applied to every input         
        self.B = (2 * torch.pi * self.b.reshape(-1, 1)).expand(-1, inputs_n)
        norm = 1./ torch.sqrt(torch.as_tensor(self.rff_n))

        
        if self.rff_n == 0:
            return X
        else:
            Z = norm * torch.sqrt(torch.as_tensor(2.)) * torch.cos((2 * torch.pi * self.W) @ X + self.B)   

            # Append the states themselves for reconstruction using the identity observable, in reconstruct.py
            if self.append_inputs:
                Z = torch.concatenate((torch.as_tensor(X), Z), axis=0)
                 
        return Z

