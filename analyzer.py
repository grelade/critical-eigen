#!/usr/bin/env python3

import networkx as nx
import numpy as np
import pandas as pd
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
    N,T = X.shape
    return 1/T * X @ X.T

def calc_eigenvalues(cov):
    L,U = np.linalg.eigh(cov)
    return L

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

def autocorr_one(X):
    series = pd.Series(X.sum(axis=0))
    # returns array for possibility of concatenation for batch_autocorr_one function
    return np.array([series.autocorr(lag=1)])

def batch_cov(Xs):
    covs = None
    for X in Xs:
        cov = calc_cov(X)
        cov = np.expand_dims(cov,axis=0)
        covs = cov if covs is None else np.concatenate((covs,cov),axis=0)
        
    return covs

def batch_eigenvalues(covs):
    Ls = None
    for cov in covs:
        L = calc_eigenvalues(cov)
        L = np.expand_dims(L,axis=0)
        Ls = L if Ls is None else np.concatenate((Ls,L),axis=0)
        
    return Ls

def batch_clusters(Xs,adj,n_clusters=10):
    clusters = None
    for X in Xs:
        cl = find_clusters(X,adj,n_clusters).mean(axis=0)
        cl = np.expand_dims(cl,axis=0)
        clusters = cl if clusters is None else np.concatenate((clusters,cl),axis=0)
        
    return clusters

def batch_autocorr_one(Xs):
    ac_one_tab = None
    for X in Xs:
        ac_one = autocorr_one(X)
        ac_one_tab = ac_one if ac_one_tab is None else np.concatenate((ac_one_tab, ac_one), axis=0)

    return ac_one_tab

def batch_mean_corr(Xs):
    mean_corr_tab = None
    for X in Xs:
        X = standardize(X)
        mean_corr = calc_cov(X).mean()
        mean_corr = np.array([mean_corr])
        mean_corr_tab = mean_corr if mean_corr_tab is None else np.concatenate((mean_corr_tab, mean_corr), axis=0)

    return mean_corr_tab

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
    sim_dir = parameters.get("sim_dir", 'sims/test_sim')
    an_dir = parameters.get("an_dir", 'analyses/test_an')

    flags = parser['Flags']
    standardize_data = flags.getboolean("standardize_data", False)
    covariance = flags.getboolean("covariance", True)
    clusters = flags.getboolean("clusters", True)
    eigenvalues = flags.getboolean("eigenvalues", True)
    autocorrelation = flags.getboolean("autocorrelation", True)
    mean_correlation = flags.getboolean("mean_correlation", True)
    skip_calculated = flags.getboolean("skip_calculated", True)

    # create unique directory
    postfix = determine_unique_postfix(an_dir)
    if postfix != '':
        if not skip_calculated:
            an_dir += postfix
            print("Run name changed: {}".format(an_dir))
        else:
            print("Found a run, skipping: {}".format(an_dir))
            sys.exit()

    # create run directory and copy the connectome
    os.makedirs(an_dir, exist_ok=False)
    
    connectome_file = os.path.join(sim_dir,'connection_matrix.dat')
    sim_file = os.path.join(sim_dir,'output.npz')
    connectome_name = os.path.split(connectome_file)[-1]    
    
    # copyfile(connectome_file, os.path.join(run_name, connectome_name))
    # write the config file
    with open(os.path.join(an_dir, 'an_config.ini'), 'w') as config:
        parser.write(config)

    # load connectome_file:
    try:
        connectome = np.loadtxt(connectome_file)
    except:
        sys.exit("Connectome file {} not found".format(connectome_file))
        
    # load sim_file:
    try:
        data = np.load(sim_file)
        Xs = data['activation_matrix']
        Ts = data['Ts']
    except:
        sys.exit("Simulation file {} not found".format(sim_file))

    # define size of the network
    _,N,_ = Xs.shape
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
    os.chdir(an_dir)

    # main simulation goes here:
    start_time = timeit.default_timer()
    
    # modify refractory cells
    for X in Xs:
        X[X==2] = new_refractory_value
        if standardize_data:
            X = standardize(X)

    # output_data = dict()
    # output_data['Ts'] = Ts
    
    covs = None
    if covariance:
        print('Calculating covariance...')
        
        covs = batch_cov(Xs)
        covs_data = dict()
        covs_data['Ts'] = Ts
        covs_data['cov'] = covs
        # output_data['cov'] = covs

        np.savez_compressed("covs_data.npz", **covs_data)
        
    if eigenvalues:
        print('Finding eigenvalues...')
        
        if covs is None:
            covs = batch_cov(Xs)
        evs_data = dict()
        evs_data['Ts'] = Ts
        # evs = batch_eigenvalues(covs)
        evs_data['evs'] = batch_eigenvalues(covs)
        # output_data['cov_evals'] = evs

        np.savez_compressed("evs_data.npz", **evs_data)
        
    if clusters:
        print(f'Finding {n_clusters} largest clusters...')
        clusters_data = dict()
        clusters_data['Ts'] = Ts
        clusters_data['clusters'] = batch_clusters(Xs,connectome,n_clusters)
        # cs = batch_clusters(Xs,connectome,n_clusters)
        # output_data['clusters'] = cs
        np.savez_compressed("clusters_data.npz", **clusters_data)

    if autocorrelation:
        print("Calculating autocorrelations (at lag=1)...")
        ac_one_data = dict()
        ac_one_data['Ts'] = Ts
        ac_one_data['ac_one'] = batch_autocorr_one(Xs)
        np.savez_compressed("ac_one_data.npz", **ac_one_data)

    if mean_correlation:
        print("Caluclating average correlation ...")
        mean_corr_data = dict()
        mean_corr_data["Ts"] = Ts
        mean_corr_data["mean_correlation"] = batch_mean_corr(Xs)
        np.savez_compressed("mean_correlation.npz", **mean_corr_data)


    # np.savez_compressed('output.npz',**output_data)
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