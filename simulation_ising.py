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

# find the model.py package:
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
from model import IsingModel, calc_Tc

from func import determine_unique_postfix

def simulation(n_steps: int,
               n_therm: int,
               T: float,
               J: float,
               connectome: nx.Graph,
               seed: int) -> np.ndarray:
    
    # set random seed:
    np.random.seed(seed)

    model = IsingModel(n_steps = n_steps,
                       n_transient = n_therm,
                       T = calc_Tc(J),
                       J = J,
                       network = connectome)

    activation_matrix = model.simulate()
    activation_matrix[activation_matrix == -1] = 0
    # set up model parameters
    # model = SERModel(n_steps=n_steps,
    #                  n_transient=n_therm,
    #                  prob_spont_act=ri,
    #                  prob_recovery=rf,
    #                  prop_e=prop_active,
    #                  threshold=T
    #                  )

    # run simulation using given connectome
    # activation_matrix = model.simulate(adj_mat=conn_matrix)
    # activation_matrix[activation_matrix == -1] = 2 # return refractory nodes to 2
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
    J = parameters.getfloat("J", 1.0)
    # rf = parameters.getfloat("rf", 0.2)
    #T = parameters.getfloat("T", 0.05)
    
    Tc = calc_Tc(J)
    
    dT_init = parameters.getfloat("dT_init", 0)
    dT_final = parameters.getfloat("dT_final", 2.0)
    T_n = parameters.getint("T_n", 1)
    
    seed = parameters.getint("seed", 124)
    # frac_init_active_neurons = parameters.getfloat("frac_init_active_neurons", 0.01)
    connectome_file = parameters.get("connectome_file", 'sample.dat')
    run_name = parameters.get("run_name", 'test_run')

    flags = parser['Flags']
    # r_rocha = flags.getboolean("r_rocha", False)
    # connectome_normalization = flags.getboolean("connectome_normalization", False)
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

    # load connectome:
    try:
        connectome = nx.read_gpickle(connectome_file)
    except:
        sys.exit("Connectome file {} not found".format(connectome_file))

    # apply flags:
    # if connectome_normalization:
    #     for row in connectome:
    #         s = np.sum(row)
    #         if s>0:
    #             row /= s

    # define size of the network
    # Nsize = len(connectome)

    # if r_rocha:
    #     ri = 2.0 / Nsize
    #     rf = ri ** 0.2

    # print the parameters:
    # print("Parameters read from the file:")
    # print("Number of time steps: {}".format(t_max))
    # print("Number of discarded time steps (thermalization): {}".format(t_th))
    # print("Number of neurons in the model: {}".format(Nsize))
    # print("Fraction of initially active neurons: {}".format(frac_init_active_neurons))
    # print("Spontaneous activation probability: {}".format(ri))
    # print("Relaxation probability: {}".format(rf))
    # # print("Value of the threshold (temperature): {}".format(T))
    # print(f"Thresholds (temperatures): np.linspace({T_init},{T_final},{T_n})")
    # print("Connectome file: {}".format(connectome_name))
    # print("Normalization of the connectome: {}".format(connectome_normalization))
    # print("Rocha's ri and rf: {}".format(r_rocha))
    # print(f"Skip finished calculation: {skip_calculated}")
    # only now enter run's directory
    os.chdir(run_name)
    
    # main simulation goes here:
    start_time = timeit.default_timer()
    
    am_tab = None
    Ts = Tc*np.linspace(dT_init,dT_final,T_n)
    for T in Ts:
        print(f"T = {T:.2f}")
        activation_matrix = simulation(n_steps=t_max,
                                       n_therm=t_th,
                                       T=T,
                                       J=J,
                                       connectome=connectome,
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
    
    # save the connectome
    # copyfile(connectome_file, os.path.join(run_name, connectome_name))
    connectome_name = 'connection_matrix.dat'
    nx.write_gpickle(connectome,connectome_name)
    # np.savetxt(connectome_name,connectome)
    
    end_time = time.asctime()
    final_time = timeit.default_timer()
    print()
    print()
    print("Python script ended on: {}".format(end_time))
    print("Total time: {:.2f} seconds".format(final_time - start_time))