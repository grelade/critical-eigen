# critical-stroke

Repository accompanying the article ["Investigating structural and functional aspects of the brain’s criticality in stroke."](https://www.nature.com/articles/s41598-023-39467-x).
Contents:
- code with simulation of Ising and HTC dynamics defined on arbitrary networks, 
- various network generators, and
- code for performing cluster analysis

## Requirements
* numpy
* scipy
* numba
* networkx
* scikit-image
* jupyter (only for jupyter notebook tutorials)
* matplotlib (only for jupyter notebook tutorials)

Tested using:
* numpy (1.21.0)
* scipy (1.8.0)
* numba (0.55.1)
* scikit-image (0.19.3)
* networkx (2.8)

## Simulation:
Before starting cluster analysis, create files containing simulation
results (of the HTC model and Ising model). Two executable scripts are
used for this purpose: `simulation_htc.py` and `simulation_ising.py`.
The simulation scripts require configuration files. Examples used in
tutorial-notebooks are located in `cfgs/` directory. Then, the script
are run using following command:

> ./simulation_htc.py Path/to/Config/File.ini

or
> ./simulation_ising.py Path/to/Config/File.ini

For the structure of configuration files see `sim_htc_config.ini` and
`sim_ising_config.ini` which contain detailed information.
The simulation scripts saves all data in  `output.npz` file
in a desired directory.

## Preparation of adjacency (connection) matrices and connectomes
The standard [Hagmann et al.'s connectome](https://doi.org/10.1371/journal.pbio.0060159.g001) is located in `hagmann_connectome.npz`
file. It consists of a connection matrix and labels assigning regions-of-interest
(ROIs) to appropriate resting-state-networks (RNSs).

Examples for preparation of Ising grids (for Ising simulations) and modified
connectomes (for HTC model simulations) can be foud in `gen_connectome.ipynb`.


## Cluster analysis
Examples of cluster analysis are prepared in `cluster_analysis.ipynb` notebook.
Clustering routines for both models can be found in `clusters_htc.py` and
`clusters_ising.py`. In the case of the HTC model clusters are found using graph
approach (`networkx`), while in the Ising model, clusters are computed using image-analysis
based methods (`skicit-image`).
