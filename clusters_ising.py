import matplotlib.pyplot as plt
import numpy as np
import skimage
from tqdm import tqdm


def cluster_sizes(array, lattice_size, mask):
    assert mask.shape == (lattice_size, lattice_size), "Lattice size and mask shape do not match!"
    img = array * mask
    
    img_label = skimage.measure.label(img, connectivity=1, background=0)
    
    region_table = skimage.measure.regionprops_table(img_label, properties=('area',))
#     clusters = sorted(region_table['area'], reverse=True)
    clusters = np.sort(region_table['area'])[::-1] # roughly 10 times faster than line above
    
    return clusters


def find_clusters_ising(X, lattice_size, mask=None, n_clusters=10):
    if np.all(mask == None):
        mask = np.ones((lattice_size, lattice_size))
    else:
        assert mask.shape == (lattice_size, lattice_size), "Lattice size and mask shape do not match!"
        
    clusters = None

    for Xslice in X.T:
        
        lattice_state = Xslice.reshape((lattice_size, lattice_size))
        clusters_t = cluster_sizes(lattice_state, lattice_size, mask)
        
        offset = n_clusters - len(clusters_t)
        if offset > 0:
            clusters_t = np.pad(clusters_t, [(0, offset)])
        else:
            clusters_t = clusters_t[:n_clusters]
        
        clusters = clusters_t if clusters is None else np.vstack((clusters, clusters_t))
    
    return clusters
    

def batch_clusters_ising(Xs, mask=None, n_clusters=10):
    clusters=None
    
    size = Xs.shape[1]
    lattice_size = int(np.sqrt(size))
    
    if np.all(mask == None):
        mask = np.ones((lattice_size, lattice_size))
    else:
        assert mask.shape == (lattice_size, lattice_size), "Lattice size and mask shape do not match!"
        
    for X in tqdm(Xs):
        clusters_X = find_clusters_ising(X, lattice_size, mask, n_clusters)
        clusters_X = np.expand_dims(clusters_X, axis=0)
        clusters = clusters_X if clusters is None else np.vstack((clusters, clusters_X))
    
    return clusters

    
# def patch_mask(L, L_patch, kind):
#     d1 = L // 2 - L_patch//2
#     d2 = L // 2 + L_patch//2
    
#     if kind == 'outside':
#         mask = np.ones((L,L))
#         mask[d1:d2,d1:d2] = 0
#         return mask
    
#     if kind == 'inside':
#         mask = np.zeros((L,L))
#         mask[d1:d2,d1:d2] = 1
#         return mask
    
    
# def half_mask(L):
#     mask = np.ones((L,L))
#     mask[:,L//2:] = 0 
#     return mask
