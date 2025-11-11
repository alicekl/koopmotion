'''
Defines the loss terms to optimize over in KoopMotion 
'''

import torch
import torch.nn as nn


class LossFunction():
    def __init__(self, loss_type='custom_loss'):
        self.loss_type = loss_type

        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'custom_loss':
            self.loss_fn = custom_loss


    def __call__(self, training_args,lifted_inputs_current, lifted_inputs_next,  U, V, lifted_goal_next, kernel, dense_training_points):
        return self.loss_fn(training_args, lifted_inputs_current, lifted_inputs_next,  U, V, lifted_goal_next, kernel, dense_training_points)


def custom_loss(training_args, lifted_inputs_current, lifted_inputs_next, U, V, 
                lifted_goal_next,
                kernel, dense_training_points):
    
    mse_loss = nn.MSELoss()
    loss = 0 

    K = U @ V.T

    # Linearity loss in the lifted space 
    loss_koopman_linearity = mse_loss(lifted_inputs_next, K @ lifted_inputs_current)   

    # Invariance under action of Koopman operator in the lifted space 
    loss_goal_convergence = mse_loss(lifted_goal_next, K @ lifted_goal_next)

    # Divergence Loss 
    divergence = compute_divergence(dense_training_points, kernel, K) 
    loss_div_local = mse_loss(torch.tensor(0.), divergence)
    
    # Apply weights for all our losses, defined in config file 
    loss_div_local_weighted = loss_div_local * training_args['divergence_weight'] 
    loss_koopman_linearity_weighted = loss_koopman_linearity *  training_args['koopman_weight'] 
    loss_goal_convergence_weighted = loss_goal_convergence * training_args['convergence_weight'] 
    
    loss += loss_koopman_linearity_weighted + loss_div_local_weighted + loss_goal_convergence_weighted 
    
    # Returning these for scaling weights in ablation study 
    losses = {
            "koopman": loss_koopman_linearity_weighted,
            "divergence": loss_div_local_weighted,
            "goal": loss_goal_convergence_weighted,
            }   
    return loss, losses 


def compute_divergence(x, kernel, K):
    '''
    Computes the divergence of the region of the vector field whose tails are the demonstration trajectory points 
    '''
    x = x.clone().detach().T.requires_grad_(True) 
   
    f_x = (K @ kernel(x.T))[:2, :].T  
    div_f = torch.zeros(x.shape[0], device=x.device) 

    # Looping over the output dimensions
    for i in range(f_x.shape[1]):  
        grad = torch.autograd.grad(f_x[:, i].sum(), x, create_graph=True)[0]
        
         # Taking the sum of all df_i/dx_i terms 
        div_f += grad[:, i] 
        
    return torch.mean(div_f)


