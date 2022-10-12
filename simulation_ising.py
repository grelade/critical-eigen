#!/usr/bin/env python3

import sys
import os
from shutil import copyfile
import time
import configparser
import getpass
import timeit
# import networkx as nx
import numpy as np
# import numpy.linalg as LA
# from scipy.io import loadmat
import networkx as nx
from typing import Union

# find the model.py package:
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
from model import IsingModel, calc_Tc

from func import determine_unique_postfix

def simulation(n_steps: int,
               n_therm: int,
               n_sweep: int,
               T: float,
               J: float,
               connectome: Union[nx.Graph,np.ndarray],
               init_type: str,
               seed: int) -> np.ndarray:
    
    # set random seed:
    np.random.seed(seed)

    model = IsingModel(n_steps = n_steps,
                       n_transient = n_therm,
                       n_sweep = n_sweep,
                       T = T,
                       J = J,
                       network = connectome,
                       init_type = init_type)

    activation_matrix = model.simulate()
    # activation_matrix[activation_matrix == -1] = 0
    
    return activation_matrix


if __name__ == '__main__':
    # Print initial message:
    initial_time = time.asctime()
    hostname = os.uname()[1].split(".")[0]
    print("Python script started on: {}".format(initial_time))
    print("{:>24}: {}".format('from', hostname))
    print("Name of python script: {}".format(os.path.abspath(__file__)))
    print("Script run by: {}\n".format(getpass.getuser()))

    # Get the file with parameters and read them:
    config_file = sys.argv[1]
    print("Configuration file: {}\n".format(config_file))
    parser = configparser.ConfigParser()
    parser.read(config_file)

    parameters = parser["Parameters"]
    t_max = parameters.getint("t_max", 2000)
    t_th = parameters.getint("t_th", 200)
    t_sweep = parameters.getint("t_sweep", None)
    J = parameters.getfloat("J", 1.0)
    init_type = parameters.get("init_type", 'uniform')
    
    Tc = calc_Tc(J)
    
    dT_init = parameters.getfloat("dT_init", 0)
    dT_final = parameters.getfloat("dT_final", 2.0)
    T_n = parameters.getint("T_n", 1)
    
    seed = parameters.getint("seed", 124)
    connectome_file = parameters.get("connectome_file", 'sample.npz')
    run_name = parameters.get("run_name", 'test_run')

    flags = parser['Flags']
    skip_calculated = flags.getboolean("skip_calculated", True)
    
    # create unique directory
    postfix = determine_unique_postfix(run_name)
    if postfix != '':
        if not skip_calculated:
            run_name += postfix
            print("Run name changed: {}".format(run_name))
        else:
            print("Found a run, skipping: {}".format(run_name))

    # create run directory
    os.makedirs(run_name, exist_ok=False)
    connectome_name = os.path.split(connectome_file)[-1]    

    # write the config file
    with open(os.path.join(run_name, 'sim_config.ini'), 'w') as config:
        parser.write(config)    
        
    start_time = timeit.default_timer()
    # load connectome:
    try:
        cf = np.load(connectome_file,allow_pickle=True)
        if 'adj' in cf.keys():
            print(f'from connectome file {connectome_file} reading: adjacency matrix (SLOW)')
            connectome = cf['adj']
        elif 'graph' in cf.keys():
            connectome = cf['graph']
            print(f'connectome file {connectome_file} reading: nx graph file')
        else:
            print('no adj/graph in connectome file!')
            # connectome = nx.read_gpickle(connectome_file)
            
    except:
        sys.exit("Connectome file {} not found".format(connectome_file))
        
    final_time = timeit.default_timer()
    print(f'Loading connectome took {final_time-start_time:.2f} seconds')

    # save the connectome
    connectome_name = 'connection_matrix.npz'
    copyfile(connectome_file, os.path.join(run_name, connectome_name))        
    
    os.chdir(run_name)
    
    # main simulation goes here:
    start_time = timeit.default_timer()
    
    am_tab = None
    Ts = Tc*np.linspace(dT_init,dT_final,T_n)
    
    for T in Ts:
        print(f"T = {T:.2f}")
        activation_matrix = simulation(n_steps=t_max,
                                       n_therm=t_th,
                                       n_sweep=t_sweep,
                                       T=T,
                                       J=J,
                                       connectome=connectome,
                                       init_type=init_type,
                                       seed=seed)
        activation_matrix = np.expand_dims(activation_matrix,axis=0)
            
        am_tab = activation_matrix if am_tab is None else np.concatenate((am_tab,activation_matrix),axis=0)

    # truncate the extension of the connectome filename
    # connectome_name_wo_ext = os.path.splitext(connectome_name)[0]
    # output_filename = 'activation_matrix_{}'.format(connectome_name_wo_ext)
    
        
    # save the activation matrix
    #np.savetxt(output_filename, activation_matrix, delimiter=",")
    #np.save(output_filename, activation_matrix)
    output_filename = 'output.npz'
    np.savez_compressed(output_filename, activation_matrix = am_tab, Ts = Ts)
    
    
    end_time = time.asctime()
    final_time = timeit.default_timer()
    print()
    print()
    print("Python script ended on: {}".format(end_time))
    print("Total time: {:.2f} seconds".format(final_time - start_time))