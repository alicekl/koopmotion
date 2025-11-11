'''
Runs training, vector field reconstruction, and plotting 
'''

import yaml as yaml 
import os 
from time import strftime
import argparse
import torch 

from train import ModelTrainer
from utils import int_constructor, get_training_time
from rff import LearnedFF 
from koopman_model import KoopmanModel
from loss import LossFunction

from evaluation import reconstruct_learnt_vector_field, plot_learnt_vector_field, save_vector_field
from checking_trajectories import run_checking_trajectories




def train_loop(training_args, data_args, plotting_args, config_filename):    
    '''
    Setups the whole training procedure 
    '''
    # Get model   
    lifting_kernel = LearnedFF(training_args['append_inputs'], training_args['num_rff'], data_args['features_n'])
    model = KoopmanModel(lifting_kernel, training_args, data_args['features_n'])  
    
    if training_args['training_on_gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
    
    # Get training data 
    trainer = ModelTrainer(training_args, data_args, plotting_args)
    training_data_torch = get_training_data(trainer, training_args)
    training_inputs, training_outputs  = training_data_torch
    concatenated_training_data_torch = [training_inputs, training_outputs]
    concatenated_training_data_torch = torch.concatenate(concatenated_training_data_torch, axis=1)

    # Train model
    time_str_start = strftime("%Y%m%d-%H%M%S")
    print('Date and time before training:', time_str_start)
    
    # Saving all the weights paths in case earlier ones are needed later
    # trained_weights_paths = [] 
    for n_update in range(training_args['epochs_n']):    
       
        optimizer = torch.optim.AdamW(model.parameters(), 
                                     lr=float(training_args['lr']), 
                                     weight_decay=float(training_args['weight_decay']))
        loss_fnc = LossFunction(training_args['loss_type'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        training_args['lr_step_size'],
                                                        gamma=training_args['lr_gamma'])

        trained_weights_path, lr_scheduler = trainer.train(n_update, model, 
                                                           training_data_torch, concatenated_training_data_torch, 
                                                           loss_fnc, 
                                                           optimizer, lr_scheduler, 
                                                           config_filename)
        # trained_weights_paths.append(trained_weights_path)   
        
    time_str_end = strftime("%Y%m%d-%H%M%S")
    print('Date and time after training:', time_str_end)
    time_difference = get_training_time(time_str_start, time_str_end)  
    print('Total training time in seconds', time_difference)   

    return trainer, trained_weights_path # trained_weights_paths[-1] 
    
def get_training_data(trainer, training_args):
    '''
    Sets up training data for pytorch 
    '''
    print("Shape of Training data inputs:", trainer.data.training_inputs.shape ,'\n')  
    inputs = torch.tensor(trainer.data.training_inputs.copy())
    outputs = torch.tensor(trainer.data.training_outputs.copy())

    if training_args['training_on_gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs.to(device)
        outputs.to(device)
    
    training_data_torch = [inputs, outputs]
    
    return training_data_torch

def get_config_file_params(config_folder_path, config_filename):
    ''' 
    Loads configuration file
    '''
    config_file_path =  os.path.join(config_folder_path, config_filename)

    with open(config_file_path + '.yaml' , 'r') as file:
        config_args = yaml.load(file, Loader=yaml.SafeLoader)
    
        # # check for poorly chosen inputs
        # assert config_args['data_params']['num_traj'] > config_args['training_params']['batch_size'], \
        # 'Must choose num_traj greater than batch_size.'

    return config_args   


if __name__ == '__main__':
    yaml.add_constructor('tag:yaml.org,2002:int', int_constructor, Loader=yaml.SafeLoader)   
    parser = argparse.ArgumentParser(description="Learn dynamical systems based on config files in folder defined")
    parser.add_argument("folder", type=str, help="Name of the folder with config files")

    args = parser.parse_args()
    config_folder = args.folder

    # Inside this folder, you can learn a dynamical system for different shapes
    config_names = os.listdir(config_folder)
    config_names.sort()

    print(f'Learning dynamical systems for {config_names}')

    for config_name in config_names:

        config_filename = 'model_config'
        config_folder_path = os.path.join(os.getcwd(), 
                                          config_folder,
                                          config_name) 
                                           
        subfolders = os.listdir(config_folder_path)
        subfolders.sort()    
    

        config_args = get_config_file_params(config_folder_path, config_filename)    
        print(config_args)

        
        
        trainer, trained_weights_path = train_loop(config_args['training_args'], 
                                                   config_args['data_args'], 
                                                   config_args['plotting_args'],
                                                   config_filename)
        
        vector_field = reconstruct_learnt_vector_field(config_args['training_args'], 
                                                config_args['data_args'],
                                                config_args['reconstruction_args'],
                                                trained_weights_path)
        
         # Save your vector field for later use
        if config_args['reconstruction_args']['save_vector_field']:
            save_vector_field(vector_field, config_args['data_args'])

        if config_args['plotting_args']['plot_figure']:
            plot_learnt_vector_field(config_args['training_args'], 
                                     config_args['data_args'], 
                                     config_args['plotting_args'],
                                     config_args['reconstruction_args'],
                                     vector_field)

        #  Forward propagates trajectories for evaluation 
        run_checking_trajectories(vector_field,
                                  config_args['data_args'],
                                  config_args['reconstruction_args'],
                                  config_args['plotting_args'])