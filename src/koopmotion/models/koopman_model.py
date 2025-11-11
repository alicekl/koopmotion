'''
Koopman Operator Theoretic model class 
'''

from torch import nn
import torch
import numpy as np
import torch.nn.init as init


class KoopmanModel(nn.Module):
    def __init__(self, observables, training_args, D=2):
        super().__init__()
        self.training_args = training_args
        
        # Append the dimension of the state itself for reconstruction with the identity observable 
        self.num_rff = self.training_args['num_rff'] + D 
                
        # Note, we use a lower rank version of the Koopman operator as this seemed to help with regularization         
        rank = self.training_args['operator_rank']
        self.U = nn.Parameter(torch.empty((self.num_rff, rank), requires_grad=True))
        self.V = nn.Parameter(torch.empty((self.num_rff, rank), requires_grad=True))

        init.xavier_uniform_(self.U)
        init.xavier_uniform_(self.V)
        
        self.observables = observables

    def forward(self, inputs):
        lifted_inputs = self.observables(inputs)
        
        # Applying linear Koopman operator in the lifted space where Koopman, K, is defined by self.U @ self.V.T
        K = self.U @ self.V.T
        forward_propagated_lifted_inputs = K @ lifted_inputs

        return lifted_inputs, forward_propagated_lifted_inputs