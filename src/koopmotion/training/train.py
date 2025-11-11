''' 
Training/optimization functions 
'''

import numpy as np
import yaml as yaml 
import os  
import torch 
import matplotlib.pyplot as plt 

from koopmotion.datasets.data_getter import TrainingData  

from time import strftime
import sys

# For checking gradients during training 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter() 


class ModelTrainer():
    def __init__(self, training_args, data_args, plotting_args):
        self.training_args = training_args 
        self.data_args = data_args 
        self.plotting_args = plotting_args
        self.data = TrainingData(self.training_args, self.data_args, self.plotting_args)
    
    def save_model(self, epoch, model, optimizer, lr_scheduler, loss, config_filename, n_update):       

        model_path = os.path.join(os.getcwd(), 'trained_weights', self.data_args['system'])
        path_exists = os.path.exists(model_path)
        if not path_exists:
            os.makedirs(model_path)
            
        time_str = strftime("%Y%m%d-%H%M%S")
        model_filename = time_str + '_' + config_filename +  '_' + 'ep' + str(n_update)
        trained_weights_path = os.path.join(model_path, model_filename)
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'training_args_dictionary': model.training_args,
                    'observables': model.observables,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': loss,
                    }, trained_weights_path)
        print('Saved to:',  trained_weights_path ,'\n ')
        
        return trained_weights_path 
        
        
    def train(self, n_update, model, data, concatenated_data, loss_fn, optimizer, lr_scheduler, config_filename):
        try:
            data_input, data_output = data
            training_n = data_output.shape[-1]
    
            # Get groups of training points - this is more typical for Koopman 
            # (if using this, reconstruction error is lower when using a small batch size defined in config yaml)
            if self.training_args['training_via_trajectories']:
                permutation = torch.arange(training_n)
                
            # Training via sparse flow measurements 
            else:
                permutation = torch.randperm(training_n)


            for batch_i in range(0, training_n, self.training_args['batch_size']): 
                optimizer.zero_grad()
                
                # In one epoch, we'll iterate through all indices grouped by batch_size
                batch_ints = permutation[batch_i: batch_i+self.training_args['batch_size']]
              
                training_points_for_divergence_loss = self.get_points_around_training(concatenated_data.T)

                PsiX, _ = model(data_input[:, batch_ints])
                PsiY, _ = model(data_output[:, batch_ints])
                
                # Assuming the last point of the trajectory is the goal we must converge to 
                goal = data_output[:, -1].reshape(-1, 1)
                lifted_goal_next, _ = model(goal)               

                loss, _  = loss_fn(self.training_args,
                                    PsiX, PsiY,
                                    model.U, model.V, 
                                    lifted_goal_next,
                                    model.observables,
                                    training_points_for_divergence_loss)
                
                loss.backward()
                
                log_every_epochs = 5
                if n_update % log_every_epochs == 0 and batch_i == 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
                    writer.add_scalar("grad_norm", total_norm, n_update)

                optimizer.step()
                writer.add_scalar("total", loss, n_update)  

            print('lr: ', lr_scheduler.get_last_lr()[0], ' | loss:', loss, ' | epoch:', str(n_update), '/', self.training_args['epochs_n'])
        
            
            if n_update == 0 or  n_update == self.training_args['epochs_n'] - 1: 
                trained_weights_path = self.save_model(n_update, model, optimizer, lr_scheduler, loss, config_filename, n_update) 
                lr_scheduler.step()   
                
                return trained_weights_path, lr_scheduler 
        

        except KeyboardInterrupt:
            print('KeyboardInterrupt: Training stopped. Do you want to save? [y/n]')
            user_input = sys.stdin.readline().strip()
            if user_input == 'y':
                print('Saving weights.')            
                trained_weights_path = self.save_model(self.training_args['epochs_n'], model, optimizer, lr_scheduler, loss, config_filename)
                os._exit(0) 
            elif user_input == 'n':
                os._exit(0)
            os._exit(0)

        lr_scheduler.step()    
        return None, lr_scheduler 

    
    def get_points_around_training(self, training_points, num_samples=4):
        '''
        Grabbing points corresponding to spatial points of the training data for divergence loss
        ''' 
        i = torch.randint(0, training_points.shape[0], (num_samples,), device=training_points.device)
        points = training_points[i].T.contiguous()
        return points




        
        

