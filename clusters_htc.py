import numpy as np
import networkx as nx
from tqdm import tqdm

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

def batch_clusters(Xs,adj,n_clusters=10):
    clusters = None
    for X in tqdm(Xs):
        cl = find_clusters(X,adj,n_clusters).mean(axis=0)
        cl = np.expand_dims(cl,axis=0)
        clusters = cl if clusters is None else np.concatenate((clusters,cl),axis=0)
        
    return clusters