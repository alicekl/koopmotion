'''
This code was modified from https://github.com/justagist/pyLasaDataset

Each Data object has attributes dt and demos (For documentation, 
refer original dataset repo: 
https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt)

'''

import pyLasaDataset as lasa
import matplotlib.pyplot as plt 
import numpy as np
import os 

# DataSet object has all the LASA handwriting data files 
# as attributes, eg:
# angle_data = lasa.DataSet.Angle
# sine_data = lasa.DataSet.Sine


def save_data(demos, data, resolution):
    
    demos_data = []
    for i in range(len(demos)):
        demo_0 = demos[i]
        pos = demo_0.pos    # np.ndarray, shape: (2,2000)

        # We do not use velocity or acceleration in our training data 
        vel = demo_0.vel    # np.ndarray, shape: (2,2000) 
        acc = demo_0.acc    # np.ndarray, shape: (2,2000)
        t = demo_0.t        # np.ndarray, shape: (1,2000)

        pos = pos[:, ::resolution]
        x0 = pos[0, :-1]
        x1 = pos[0, 1:]
        y0 = pos[1, :-1]
        y1 = pos[1, 1:]
        
        # Plot the data if desired 
        # u = x1 - x0
        # v = y1 - y0
        # plt.scatter(x0, y0, alpha=0.5, s=5 * np.arange(len(x0)))
        # plt.scatter(x1, y1, alpha=0.5, s=5 * np.arange(len(x0)))
        # plt.quiver(x0, y0, u, v)
        
        data = np.zeros((2, x0.shape[0], 2))
        data[0, :, 0] = x0 
        data[1, :, 0] = y0
        data[0, :, 1] = x1
        data[1, :, 1] = y1
        
        demos_data.append(data)
        
    demos_data_all = np.concatenate(demos_data, axis=1)
    border_dev = 5

    # These parameters are used to define the bounds in the configuration file (can automate this)
    print('Parameters for configuration file data_args bounds_x and bounds_y')
    print('min x', int(np.min(demos_data_all[0, :, :]) - border_dev), 'max_x', int(np.max(demos_data_all[0, :, :]) + border_dev))
    print('min y', int(np.min(demos_data_all[1, :, :]) - border_dev), 'max_y', int(np.max(demos_data_all[1, :, :]) + border_dev))
    print(demos_data_all.shape)

    folder_path = os.path.join(os.getcwd(), 'data',  folder_name)
    os.makedirs(folder_path, exist_ok=True)

    if resolution == 1:
        np.save(folder_path + '/train_res=' + str(resolution) + '.npy', demos_data_all) 
    else:
        np.save(folder_path + '/train.npy', demos_data_all) 


if __name__ == '__main__':
    # Names used in the original dataset
    _names_lasa = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape',
            'heee','JShape','JShape_2','Khamesh','Leaf_1',
            'Leaf_2','Line','LShape','NShape','PShape',
            'RShape','Saeghe','Sharpc','Sine','Snake',
            'Spoon','Sshape','Trapezoid','Worm','WShape','Zshape',
            'Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4']

    # To use, match the original data set name defined above in names_lasa (Angle) with names_koopmotion (angle)
    # There might be a better way to do this, but this is the best method I found for using the original pyLasaDataset code
    _lasa_data = [lasa.DataSet.Angle, lasa.DataSet.BendedLine, lasa.DataSet.CShape, lasa.DataSet.DoubleBendedLine, lasa.DataSet.GShape,
            lasa.DataSet.heee, lasa.DataSet.JShape, lasa.DataSet.JShape_2, lasa.DataSet.Khamesh,lasa.DataSet.Leaf_1,
            lasa.DataSet.Leaf_2, lasa.DataSet.Line, lasa.DataSet.LShape, lasa.DataSet.NShape, lasa.DataSet.PShape,
            lasa.DataSet.RShape, lasa.DataSet.Saeghe, lasa.DataSet.Sharpc, lasa.DataSet.Sine, lasa.DataSet.Snake,
            lasa.DataSet.Spoon, lasa.DataSet.Sshape, lasa.DataSet.Trapezoid, lasa.DataSet.Worm, lasa.DataSet.WShape, lasa.DataSet.Zshape,
            lasa.DataSet.Multi_Models_1, lasa.DataSet.Multi_Models_2, lasa.DataSet.Multi_Models_3, lasa.DataSet.Multi_Models_4]

    # Names used in the koopmotion (this) repository
    _names_koopmotion = ['angle', 'bended_line', 'c', 'double_bended_line', 'g',
            'heee', 'j', 'j2', 'khamesh', 'leaf1', 
            'leaf2', 'line', 'l', 'n', 'p',
            'r', 'saeghe', 'sharpc', 'sine', 'snake',
            'spoon', 's', 'trapezoid', 'worm', 'w',
            'z', 'mm1', 'mm2', 'mm3', 'mm4']
    
    _lasa_data = [lasa.DataSet.Angle, lasa.DataSet.BendedLine, lasa.DataSet.CShape]
    _names_koopmotion = ['angle', 'bended_line', 'c']

    # Making a dictionary 
    lasa_dict = dict(zip(_names_koopmotion, _lasa_data))
    

    for name in lasa_dict.keys():
        data = lasa_dict[name] 
        folder_name = 'lasa_' + name

        # Each Data object has attributes dt and demos (For documentation, 
        # refer original dataset repo: 
        # https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt)
        dt = data.dt
        demos = data.demos # list of 7 Demo objects, each corresponding to a 
                                # repetition of the pattern

        # Each Demo object in demos list will have attributes pos, t, vel, acc 
        # corresponding to the original .mat format described in 
        # https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt

        # To visualise the data (2D position and velocity) use the plot_model utility
        lasa.utilities.plot_model(data) # give any of the available 
                                        # pattern data as argument


        # Resolution means for training data, for KoopMotion we used every 40th position
        # Remember to change this in the configuration file if this is altered 
        resolution = 40
        save_data(demos, data, resolution)

        # But to compare our training data with the actual trajectories, we also want to save all
        resolution = 1
        save_data(demos, data, resolution)