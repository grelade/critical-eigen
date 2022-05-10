#!/usr/bin/env python3

import networkx as nx
import numpy as np
import sys
import os
from shutil import copyfile
import time
import configparser
import getpass
import timeit

from func import determine_unique_postfix

def standardize(X,axis=-1):
    stds = X.std(axis=axis,keepdims=True)
    means = X.mean(axis=axis,keepdims=True)
    stds[(stds==0)] = 1

    return (X - means)/stds

def calc_cov(X):
    return 1/T * X @ X.T

def cluster_sizes(adj,mask):
    adj0 = adj[mask][::,mask]
    G0 = nx.from_numpy_array(adj0)
    return np.array([len(c) for c in sorted(nx.connected_components(G0), key=len, reverse=True)])

def find_clusters(X,adj,n_clusters=10):
    clusters = None
    for Xslice in X.T:
        mask = Xslice==1
        cl = cluster_sizes(adj,mask)
        offset = n_clusters-len(cl)
        if offset>0:
            cl = np.pad(cl,[(0,offset)])
        else:
            cl = cl[:n_clusters]
        
        clusters = cl if clusters is None else np.vstack((clusters,cl))
        
    return clusters


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
    new_refractory_value = parameters.getint("new_refractory_value", 0)
    n_clusters = parameters.getint("n_clusters", 10)
    # ri = parameters.getfloat("ri", 0.001)
    # rf = parameters.getfloat("rf", 0.2)
    # T = parameters.getfloat("T", 0.05)
    # seed = parameters.getint("seed", 124)
    # frac_init_active_neurons = parameters.getfloat("frac_init_active_neurons", 0.01)
    sim_file = parameters.get("sim_file", 'simulation.npy')
    connectome_file = parameters.get("connectome_file", 'sample.dat')
    run_name = parameters.get("run_name", 'test_analysis')

    flags = parser['Flags']
    standardize_data = flags.getboolean("standardize_data", False)
    covariance = flags.getboolean("covariance", True)
    clusters = flags.getboolean("clusters", True)
    eigenvalues = flags.getboolean("eigenvalues", True)

    # create unique directory
    postfix = determine_unique_postfix(run_name)
    if postfix != '':
        run_name += postfix
        print("Run name changed: {}".format(run_name))

    # create run directory and copy the connectome
    os.makedirs(run_name, exist_ok=False)
    connectome_name = os.path.split(connectome_file)[-1]
    # copyfile(connectome_file, os.path.join(run_name, connectome_name))
    # write the config file
    with open(os.path.join(run_name, 'an_config.ini'), 'w') as config:
        parser.write(config)

    # load connectome_file:
    try:
        connectome = np.loadtxt(connectome_file)
    except:
        sys.exit("Connectome file {} not found".format(connectome_file))
        
    # load sim_file:
    try:
        X = np.load(sim_file)
    except:
        sys.exit("Simulation file {} not found".format(sim_file))

    # define size of the network
    N,T = X.shape
    Nconn = len(connectome)
    assert Nconn == N, f'Connectome and simulation matrices have incompatible sizes {N} != {Nconn}'


    # print the parameters:
    # print("Parameters read from the file:")
    # print("Number of time steps: {}".format(t_max))
    # print("Number of discarded time steps (thermalization): {}".format(t_th))
    # print("Number of neurons in the model: {}".format(Nsize))
    # print("Fraction of initially active neurons: {}".format(frac_init_active_neurons))
    # print("Spontaneous activation probability: {}".format(ri))
    # print("Relaxation probability: {}".format(rf))
    # print("Value of the threshold (temperature): {}".format(T))
    # print("Connectome file: {}".format(connectome_name))
    # print("Normalization of the connectome: {}".format(connectome_normalization))
    # print("Rocha's ri and rf: {}".format(r_rocha))

    # only now enter run's directory
    os.chdir(run_name)

    # main simulation goes here:
    start_time = timeit.default_timer()
    
    # modify refractory cells
    X[X==2] = new_refractory_value
    
    if standardize_data:
        X = standardize(X)

    if covariance:
        print('Calculating covariance...')
        cov = calc_cov(X)
        
        cov_path = 'covariance.npy'
        np.save(cov_path,cov)
        
    if eigenvalues:
        print('Finding eigenvalues...')
        if cov is None:
            cov = calc_cov(X)
        evs,_ = np.linalg.eigh(cov)
        
        evs_path = 'eigenvalues.npy'
        np.save(evs_path,evs)
        
    if clusters:
        print(f'Finding {n_clusters} largest clusters...')
        cs = find_clusters(X,connectome,n_clusters)
        
        cs_path = 'clusters.npy'
        np.save(cs_path,cs)

    # truncate the extension of the connectome filename
    # connectome_name_wo_ext = os.path.splitext(connectome_name)[0]
    # output_filename = 'activation_matrix_{}'.format(connectome_name_wo_ext)

    # save the activation matrix
    #np.savetxt(output_filename, activation_matrix, delimiter=",")
    # np.save(output_filename, activation_matrix)
    
    
    end_time = time.asctime()
    final_time = timeit.default_timer()
    print()
    print()
    print("Python script ended on: {}".format(end_time))
    print("Total time: {:.2f} seconds".format(final_time - start_time))