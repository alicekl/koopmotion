from koopmotion.evaluation.plotter import Plotter
from koopmotion.evaluation.reconstruct import Reconstructor
import argparse
import yaml as yaml 
from koopmotion.utils.utils import int_constructor

from koopmotion.evaluation.checking_trajectories import run_checking_trajectories
import os 

import numpy as np 




def reconstruct_learnt_vector_field(training_args, data_args, reconstruction_args, trained_weights_path):
    '''
    Reconstructs the learnt vector field 
    '''

    my_reconstructor = Reconstructor()  
    vector_field = my_reconstructor.get_flow_estimate(training_args, 
                                               data_args,
                                               reconstruction_args,
                                               trained_weights_path)        

    return vector_field

    
def plot_learnt_vector_field(training_args, data_args, plotting_args, reconstruction_args, vector_field):
    assert vector_field.shape[0] == 2, 'Plotting is not configured for system with 2 states'
        
    my_plotter = Plotter()    
    my_plotter.plot_flow(training_args, vector_field)
    my_plotter.plot_streamlines(training_args, data_args, plotting_args, reconstruction_args,vector_field)


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

def save_vector_field(vector_field, data_args):
    folder_path = os.path.join('results', data_args['system']) 
    path_exists = os.path.exists(folder_path)
    if not path_exists:
        os.makedirs(folder_path)
    
    np.save(folder_path + '/vector_field.npy', vector_field)

if __name__ == '__main__':
    yaml.add_constructor('tag:yaml.org,2002:int', int_constructor, Loader=yaml.SafeLoader)   
    parser = argparse.ArgumentParser(description="Reconstructs vector field from model weights")
    parser.add_argument("config_path", type=str, help="Path with the config file. e.g., /home/user/Github/koopmotion/configuration_files/lasa_angle")
    parser.add_argument("trained_weights_path", type=str, help="Path with the trained weights. e.g., /home/user/Github/koopmotion/trained_weights/lasa_angle/20250101-000000_model_config_ep224")

    args = parser.parse_args()

    config_folder_path = args.config_path 
    trained_weights_path = args.trained_weights_path 

    config_filename = 'model_config'
    config_args = get_config_file_params(config_folder_path, config_filename)  

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