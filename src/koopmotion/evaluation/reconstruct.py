'''
Reconstructs the learnt vector field 
'''


import numpy as np
import torch 

from koopmotion.models.rff import LearnedFF 
from koopmotion.models.koopman_model import KoopmanModel

import matplotlib.pyplot as plt 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)


class Reconstructor:
    def __init__(self):
        
        pass 
        
    def get_flow_estimate(self, training_args, data_args, reconstruction_args, trained_weights_path, comparison_points=None):
        '''
        Computes a flow estimate, which is defined by the vector_field produced by an initial grid of points
        and it's forward propagation. 
        '''

        model = self.setup_model(training_args, data_args, trained_weights_path)
        
        # Prepare initial conditions to predict flow from
        if comparison_points is None:
            grid_points = self.get_uniform_grid_points(data_args, reconstruction_args)
            print('Generating uniform grid points \n')
        else:
            grid_points = comparison_points
            print('Using inputted inference points')

        
        # Prepare vector_field matrix
        num_points = grid_points.shape[1]
        point_propagation_prediction = np.zeros((data_args['features_n'], 
                                                num_points, 
                                                reconstruction_args['future_state_prediction'])) 
                                
        # Initial conditions or the tails of the vectors 
        point_propagation_prediction[:, :, 0]  = grid_points 

        # Forward propagate via Koopman and 'de-lift'
        with torch.no_grad():  

            # Koopman operator
            K = (model.U @ model.V.T)

            # # Note if we're training on sparse data, point_propagation_prediction will evolve depending on the sparse data time difference
            for i in range(1, point_propagation_prediction.shape[2]):
                lifting_operator = self.compute_lifting(point_propagation_prediction[:, :, i-1], 
                                                        model.observables(point_propagation_prediction[:, :, i-1]).numpy())
                
                # Generate vcetor_field 
                point_propagation_prediction[:, :, i] = self.get_point_propagation_prediction(data_args, 
                                                                                                reconstruction_args,
                                                                                                model.observables, 
                                                                                                point_propagation_prediction[:, :, i - 1], 
                                                                                                lifting_operator, 
                                                                                                K)            
            
        return point_propagation_prediction  
    


    def setup_model(self, training_args, data_args, trained_weights_path):
        '''
        Loads a model based on trained weight path 
        '''

        # Inialize model
        lifting_kernel = LearnedFF(training_args['append_inputs'], training_args['num_rff'], data_args['features_n'])
        initialized_model = KoopmanModel(lifting_kernel, training_args, data_args['features_n'])
        optimizer = torch.optim.Adam(initialized_model.parameters(), 
                                     lr=float(training_args['lr']), 
                                     weight_decay=float(training_args['weight_decay']))
        
        # get model based on weights 
        model, _, _ = self.load_model_weights(training_args, initialized_model, optimizer, trained_weights_path)
        
        return model 

    def get_uniform_grid_points(self, data_args, reconstruction_args):
        '''
        Gets a uniform grid of points to evaluate over '''
        
        num_rows =  reconstruction_args['num_rows'] 
        num_cols =  reconstruction_args['num_cols'] 
        num_height =  reconstruction_args['num_cols'] 
    
 
        if data_args['features_n'] == 2:
            x_values = np.linspace(*data_args['bounds_x'], num_cols)  
            y_values = np.linspace(*data_args['bounds_y'], num_rows)  
            
            x, y = np.meshgrid(x_values, y_values)
            grid_points = np.vstack((x.ravel(), y.ravel()))
            
        elif data_args['features_n'] == 3:
       
            x_values = np.linspace(*data_args['bounds_x'] , num_cols)  
            y_values = np.linspace(*data_args['bounds_y'], num_rows)  
            z_values = np.linspace(*data_args['bounds_z'], num_height)  
            
            x, y, z = np.meshgrid(x_values, y_values, z_values)
            grid_points = np.vstack((x.ravel(), y.ravel(), z.ravel()))
        
        return grid_points # shape:(dim, num_points)

    def get_point_propagation_prediction(self, data_args, reconstruction_args, kernel_psi, X, lifting_operator, K):
        ''' 
        Both versions should give the same output. Listing doing both for sanity checks.
        '''
    
        if reconstruction_args['estimate_type'] == 'using_proxy_delifting':    
            propagated_system = np.linalg.pinv(lifting_operator) @ (K @ torch.as_tensor(kernel_psi(X)).numpy())
        elif reconstruction_args['estimate_type'] == 'using_identity_observable':  
            propagated_system = (K @ torch.as_tensor(kernel_psi(X)).numpy())[:data_args['features_n'], :]
    
        return propagated_system 
        
    def compute_lifting(self, X, PsiX):
        ''' 
        Computes lifting_operator, for which we use a (different) linear delifting method to reconstruct 
        '''
        
        lifting_operator, _, _, _ = np.linalg.lstsq(PsiX.T, X.T, rcond=None) # PsiX.T (num_points x num_rff), (X.T) (num_pointsxN) 
        return lifting_operator 
    

    def load_model_weights(self, training_args, model, optimizer, trained_weights_path):
        '''
        Loads model weights 
        '''
        print('******Loading model weights for reconstruction from:', trained_weights_path)
        trained_weights = torch.load(trained_weights_path, weights_only=False)
        model.load_state_dict(trained_weights['model_state_dict'])
        optimizer.load_state_dict(trained_weights['optimizer_state_dict'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=1, 
                                                       gamma=0.1)
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=float(training_args['lr']), 
                                     weight_decay=float(training_args['weight_decay']))
        
        return model, optimizer, lr_scheduler

  