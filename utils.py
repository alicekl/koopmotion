import yaml as yaml 
import ruamel.yaml 
import numpy as np
from datetime import datetime


def seq(*l):
    s = ruamel.yaml.comments.CommentedSeq(l)
    s.fa.set_flow_style()
    return s

def int_constructor(loader, node):
    return int(loader.construct_scalar(node))   

def load_config_args(config_file_path):
    yaml.add_constructor('tag:yaml.org,2002:int', int_constructor, Loader=yaml.SafeLoader)   
    print('config_file_path', config_file_path)
    with open(config_file_path, 'r') as file:
        config_args = yaml.load(file, Loader=yaml.SafeLoader)
        
    return config_args

def get_training_time(time_str_start, time_str_end):
    start_time = datetime.strptime(time_str_start, "%Y%m%d-%H%M%S")
    end_time = datetime.strptime(time_str_end, "%Y%m%d-%H%M%S")

    # Calculate the time difference in minutes
    time_difference = (end_time - start_time).total_seconds()
    return time_difference 