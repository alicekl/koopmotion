'''
Grabs training data from data folder under data/data_name_selected_in_config/train.npy
'''

import os 
import numpy as np
import matplotlib.pyplot as plt 


class TrainingData():
    def __init__(self, training_args, data_args, plotting_args):
        self.training_args = training_args
        self.data_args = data_args
        self.plotting_args = plotting_args
        self.path =  os.path.join(os.getcwd(), 
                                  'data', 
                                  self.data_args['system'], 
                                  'train.npy' )
        
        self.training_inputs, self.training_outputs = self.get_training_data()

        
    def get_training_data(self):
        ''' 
        Prepares trajectory data for training 

        '''
        all_data = np.load(self.path)
        print('All data shape', all_data.shape)
    
        training_inputs, training_outputs= self.get_one_snapshot_data(all_data)
        
        if self.plotting_args['plot_figure']:
            self.plot_training_data(training_inputs, training_outputs)
        
        # Ensure data are float32  for training 
        training_inputs = training_inputs.astype(np.float32)
        training_outputs = training_outputs.astype(np.float32)
               
        return training_inputs, training_outputs 
    
    
    def plot_training_data(self, training_inputs, training_outputs):
        '''
        Plotter for planar trajectory data 
        '''

        assert training_inputs.shape[0] == 2, 'As of now, plotting is not configured for system with 2 states! Set config file param plot_figure : False'
        
        plt.figure(figsize=(6,6), dpi=300)
        
        plt.scatter(*training_inputs, alpha=0.5)
        plt.scatter(*training_outputs, alpha=0.5)
        plt.title('Training Data')
        plt.xlabel(self.plotting_args['x_label'])
        plt.ylabel(self.plotting_args['y_label'])
        
        difference = training_outputs - training_inputs
        plt.quiver(*training_inputs, difference[0], difference[1])
        plt.xlim(*self.data_args['bounds_x'])
        plt.ylim(*self.data_args['bounds_y'])

        # TODO: Can make saving path modular 
        folder_path = os.path.join('figures', self.data_args['system'])
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(folder_path + '/training_data.png', bbox_inches='tight')

    

    def get_one_snapshot_data(self, all_data):
        ''' 
        To generate one snap shot data by separating into training inputs and outputs
        and adding noise with noise variance defined in config  
            
        '''
    
        training_sampled_inputs = all_data[:, :, 0] 
        training_sampled_outputs = all_data[:, :, 1]
        
        # add noise to data - only to outputs for noisy observations, but also to inputs if noisy estimates of robot
        # training_sampled_inputs  += np.random.normal(0, self.data_args['training_noise_sigma'], training_sampled_inputs.shape)
        
        average_vector = np.mean(training_sampled_inputs - training_sampled_outputs, axis=1)
        average_vector_norm = np.linalg.norm(average_vector)
        
        # Not adding noise for the last sample, to make sure we reach the goal/ target
        training_sampled_outputs[:, :-1]  += np.random.normal(0, average_vector_norm * self.data_args['training_noise_sigma'], training_sampled_outputs[:, :-1].shape)
        
        return training_sampled_inputs, training_sampled_outputs
    
    def normalize_data(self, data):
        x, y = data

        data[0, :] = (x - min(self.data_args['bounds_x'])) / (max(self.data_args['bounds_x']) - min(self.data_args['bounds_x'])) 
        data[1, :] = (y - min(self.data_args['bounds_y'])) / (max(self.data_args['bounds_y']) - min(self.data_args['bounds_y'])) 

        return data
        
    