'''
Some basic plotting code 
'''

import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata
import os 

class Plotter:
    def __init__(self):
        pass 
    
    def plot_flow(self, training_args, vector_field):
        '''
        Plot learnt vector field 
        '''
        skip_n = 5 
        diff_estimate = vector_field[:, ::skip_n, 1] - vector_field[:, ::skip_n, 0]  

        plt.close('all')

        plt.figure(figsize=(6,6), dpi=300)
        plt.title('Estimated Vector Field with M = ' + str(training_args['num_rff']))
        plt.quiver(vector_field[0, ::skip_n, 0], vector_field[1, ::skip_n, 0], 
                   diff_estimate[0, :], diff_estimate[1, :], 
                   color='blue', label='prediction', alpha=0.75)
        
    
    def plot_streamlines(self, training_args, data_args, plotting_args, reconstruction_args, vector_field):
        '''
        Plot streamfunction 
        '''
        
        plt.close('all')

        plt.figure(figsize=(6,6), dpi=300)
        u = vector_field[0, :, 1] - vector_field[0, :, 0]
        v = vector_field[1, :, 1] - vector_field[1, :, 0]
        
        # Create a regular grid to interpolate onto
        grid_x, grid_y = np.meshgrid(np.linspace(*data_args['bounds_x'], reconstruction_args['num_cols']),
                                     np.linspace(*data_args['bounds_y'], reconstruction_args['num_rows']))
        
        # Interpolate u and v components onto the grid
        grid_u = griddata(vector_field[:, :, 0].T, u, (grid_x, grid_y), method='linear')
        grid_v = griddata(vector_field[:, :, 0].T, v, (grid_x, grid_y), method='linear')

        plt.streamplot(grid_x, grid_y, grid_u, grid_v, density=3)
        plt.title('Streamplot '  + str(training_args['num_rff']))
        plt.xlim(*data_args['bounds_x'])
        plt.ylim(*data_args['bounds_y'])
        
        plt.xlabel(plotting_args['x_label'])
        plt.ylabel(plotting_args['y_label'])
        
        # plt.savefig('streamplot_' + str(training_args['num_rff']).zfill(4) + '.png', dpi=300, bbox_inches='tight')
        folder_path = os.path.join('figures', data_args['system']) 
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(folder_path + '/streamfunction_with_labels.png', bbox_inches='tight')
    
    