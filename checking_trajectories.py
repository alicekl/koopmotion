import numpy as np
import matplotlib.pyplot as plt
import os 

from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

from functools import partial


# Define ODE function
def get_vector_field(t, state, interp_vx, interp_vy):
    
    x, y = state
    vx = interp_vx(x, y)
    vy = interp_vy(x, y)
    
    # Handle out of bounds cases where interpolation fails
    if np.isnan(vx) or np.isnan(vy):
        return [0, 0]
    
    return [vx, vy]

def plot_streamfunction(vector_field, reconstruction_args, plotting_args):

    u = vector_field[0, :, 1] - vector_field[0, :, 0]
    v = vector_field[1, :, 1] - vector_field[1, :, 0]
    
    x_min = np.min(vector_field[0, :, 0]) 
    x_max = np.max(vector_field[0, :, 0]) 
    y_min = np.min(vector_field[1, :, 0])
    y_max = np.max(vector_field[1, :, 0])
    
    # Create a regular grid to interpolate onto
    reconstruction_args['num_rows'], reconstruction_args['num_cols']

    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, reconstruction_args['num_cols']),
                                 np.linspace(y_min, y_max, reconstruction_args['num_rows']))
    
    # Interpolate u and v components onto the grid
    grid_u = griddata(vector_field[:, :, 0].T, u, (grid_x, grid_y), method='linear')
    grid_v = griddata(vector_field[:, :, 0].T, v, (grid_x, grid_y), method='linear')
    
    plt.figure(figsize=(6,6), dpi=300)

    plt.streamplot(grid_x, grid_y, grid_u, grid_v, density=2, color='gray')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(plotting_args['x_label'])
    plt.ylabel(plotting_args['y_label'])

def run_checking_trajectories(vector_field, data_args, reconstruction_args, plotting_args):
    
    # Plot the stream functions 
    if vector_field.shape[0] == 2:    
        plot_streamfunction(vector_field, reconstruction_args, plotting_args) 
        plt.axis('off')
        
    folder_path = os.path.join('figures', data_args['system']) 
    plt.savefig(folder_path + '/streamfunction_without_labels.png', bbox_inches='tight')
    
    if vector_field.shape[0] == 2: 
        plot_streamfunction(vector_field, reconstruction_args, plotting_args) 

    

    # Get the original non-subsampled training data (will be plotted as red trajectories)
    training_data = np.load(os.path.join(os.getcwd(), 
                                         'data', 
                                         data_args['system'],
                                         'train_res=1.npy')) # This is the filename we have used for the original, non-sampled LASA data 
    
    
    base_positions = vector_field[:, :, 0].T 
    head_positions = vector_field[:, :, 1].T  
    
    print('Ground truth demonstration data shape', base_positions.shape)

    # Compute vector field directions
    vectors = head_positions - base_positions 
    
    # Interpolate vector field - this is simply to reorganize vectors for inference 
    interp_vx = LinearNDInterpolator(base_positions, vectors[:, 0]) 
    interp_vy = LinearNDInterpolator(base_positions, vectors[:, 1]) 
    

    num_demonstrations = data_args['num_demonstrations']
    demonstration_length = training_data.shape[1]/num_demonstrations
    print('Demonstration length', demonstration_length)
    
    # Get time parameters to define how we forward propagate particles in the domain 
    res_training =  data_args['resolution_training']
    time_steps_actual = demonstration_length 
    time_steps_n = time_steps_actual + 50       # Add some extra time to see how particles from other part of the domain converge 
    end_t = time_steps_n / res_training 
    t_span = (0, end_t)      

    # Initialize some additional points other than the training data initial conditions 
    if data_args['multiple_training_trajectories']:
        length_of_training_trajectory = int(training_data.shape[1]/num_demonstrations)
    else:
        length_of_training_trajectory = training_data.shape[1]

    x_min = np.min(vector_field[0, :, 0]) 
    x_max = np.max(vector_field[0, :, 0]) 
    y_min = np.min(vector_field[1, :, 0])
    y_max = np.max(vector_field[1, :, 0])

    boundary_delta =  2
    nx, ny = 10, 10    
    x_vals = np.linspace(x_min + boundary_delta, x_max - boundary_delta, nx)
    y_vals = np.linspace(y_min + boundary_delta, y_max - boundary_delta, ny)
    initial_condition_list = np.array([[x, y] for x in x_vals for y in y_vals])

    # initialize empty lists for the trajectories we will plot 
    x_trajs = []
    y_trajs = []

    # Get our vector field 
    vector_field_function = partial(get_vector_field, interp_vx=interp_vx, interp_vy=interp_vy)

    for i in range(len(initial_condition_list) + num_demonstrations):  

        # We want to use the same initial conditions as the demonstrations, to compare propagation in time       
        if i < num_demonstrations and length_of_training_trajectory * i < training_data.shape[1] :
            y0 = training_data[:, length_of_training_trajectory*i, 0] 
        else:
            y0 = initial_condition_list[i - num_demonstrations]
        
        sol = solve_ivp(vector_field_function, t_span, y0, t_eval=np.linspace(0, end_t, int(time_steps_n)))
       
        # Extract solution
        x_traj, y_traj = sol.y
        x_trajs.append(x_traj)
        y_trajs.append(y_traj)

        # Then plot these forward propagated trajectories (matching the initial conditions in black, and otherwise in blue)     
        if i < num_demonstrations and length_of_training_trajectory * i < training_data.shape[1] :
            plt.plot(x_traj, y_traj, 'k-', linewidth=3, label='Matching I.C. Predicted Trajectory', zorder=2)
        else:    
            plt.plot(x_traj, y_traj, 'b-', label='Non I.C. Predicted Trajectories', alpha=0.5, zorder=1)            
            
        # Plottting the initial conditions 
        plt.scatter(y0[0], y0[1], color='g', marker='o', label='Initial Conditions (I.C')

        # Plotting the final end point 
        plt.scatter(x_traj[-1], y_traj[-1], color='y', marker='o', alpha=0.7, s= 500, label='End', zorder=3)
        
        # Plot every training trajectory (as their own segment)
        for demo_i in range(num_demonstrations):
            plt.plot(*training_data[:, length_of_training_trajectory*demo_i:length_of_training_trajectory*demo_i+length_of_training_trajectory, 0],'r-', zorder=1)
        
    plt.xlabel(plotting_args['x_label'])
    plt.ylabel(plotting_args['y_label'])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    folder_path = os.path.join('figures', data_args['system']) 
    plt.title("Vector Field and Trajectory")
    plt.savefig(folder_path + '/trajectories.png', bbox_inches='tight')
   