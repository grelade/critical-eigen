import numpy as np
import networkx as nx
from itertools import product

class Grid:
    
    def __init__(self):
        pass
    
    @staticmethod
    def grid_2d(size_x : int,
                size_y : int,
                periodic : bool = False):
        
        g = nx.grid_2d_graph(size_x, size_y, periodic=periodic)
        nx.set_node_attributes(g,
                               values=0,
                               name='subsystem')      
        return g
    
    @staticmethod
    def grid_2d_patch(size_x : int,
                      size_y : int,
                      x0 : int,
                      y0 : int,
                      dx : int,
                      dy : int,
                      remove_frac : float = 1.0):

        g = Grid.grid_2d(size_x,size_y)
        
        # find edges on the patch boundary
        x1 = x0 + dx
        y1 = y0 + dy
        x0_bunch = [((x0-1,i),(x0,i)) for i in range(y0,y1+1)]
        x1_bunch = [((x1,i),(x1+1,i)) for i in range(y0,y1+1)]
        y0_bunch = [((i,y0-1),(i,y0)) for i in range(x0,x1+1)]
        y1_bunch = [((i,y1),(i,y1+1)) for i in range(x0,x1+1)]
        
        ebunch = x0_bunch + y0_bunch + x1_bunch + y1_bunch
        
        # randomize boundary edges
        rng = np.random.default_rng(42)
        ebunch_ixs = list(rng.permutation(range(len(ebunch))))
        ebunch = [ebunch[i] for i in ebunch_ixs]
        ixmax = int(remove_frac*len(ebunch))
        g.remove_edges_from(ebunch[:ixmax])
        print(f'removing {len(ebunch[:ixmax])} out of {len(ebunch)} edges on the boundary')
        # label subsystems

        patch_nodes = product(list(range(x0,x1+1)),
                              list(range(y0,y1+1)))
        nx.set_node_attributes(g,
                               values={n: 1 for n in patch_nodes},
                               name='subsystem')
        return g
    
    @staticmethod    
    def grid_2d_sliced(size_x : int,
                       size_y : int,
                       slice_ix : int,
                       row = False,
                       remove_frac : float = 1.0):
        
        g = Grid.grid_2d(size_x,size_y)
        if not row:
            ebunch = [((slice_ix-1,i),(slice_ix,i)) for i in range(size_y)]
            patch_nodes = product(list(range(0,slice_ix)),
                                  list(range(0,size_y)))
        else:
            ebunch = [((i,slice_ix-1),(i,slice_ix)) for i in range(size_x)]
            patch_nodes = product(list(range(0,size_x)),
                                  list(range(0,slice_ix)))
        #randomize the slice
        rng = np.random.default_rng(42)
        ebunch_ixs = list(rng.permutation(range(len(ebunch))))
        ebunch = [ebunch[i] for i in ebunch_ixs]
        ixmax = int(remove_frac*len(ebunch))
              
        g.remove_edges_from(ebunch[:ixmax])
        print(f'removing {len(ebunch[:ixmax])} out of {len(ebunch)} edges on the boundary')
        # label subsystems
        nx.set_node_attributes(g,
                               values={n: 1 for n in patch_nodes},
                               name='subsystem')
        
        return g

    @staticmethod
    def nx_to_np(graph: nx.Graph):
        
        adj = nx.to_numpy_matrix(graph)
        subsystems = np.array([attr['subsystem'] for n,attr in graph.nodes.items()])
        nodes = np.array([list(n) for n,attr in graph.nodes.items()])
        
        return {'adj': adj,
                'subsystems': subsystems,
                'nodes': nodes}
    
    
class Network:
    
    def __init__(self):
        pass
    
    @staticmethod
    def watts_strogatz(nodes: int,
                       num_neighbors: int,
                       prob: float):

        g = nx.connected_watts_strogatz_graph(n=nodes,k=num_neighbors,p=prob)
        nx.set_node_attributes(g,
                               values=0,
                               name='subsystem')      
        return g
    
    @staticmethod
    def erdos_renyi(nodes: int,
                    prob: float):

        g = nx.erdos_renyi_graph(n=nodes,p=prob)
        nx.set_node_attributes(g,
                               values=0,
                               name='subsystem')      
        return g
    
    @staticmethod
    def barabasi_albert(nodes: int,
                        edge_num: int):

        g = nx.barabasi_albert_graph(n=nodes,m=edge_num)
        nx.set_node_attributes(g,
                               values=0,
                               name='subsystem')      
        return g    
    
    @staticmethod
    def patch(graph: nx.Graph, n_subnodes: int, remove_frac: float):
        
        g = graph.copy()
        rng = np.random.default_rng(42)
        subnodesA = rng.permutation(g.nodes)[:n_subnodes]
        
        subnodesB = g.nodes-subnodesA

        ebunch = [eb for eb in nx.edge_boundary(g,nbunch1=subnodesA,nbunch2=subnodesB)]
        ebunch = rng.permutation(ebunch)
        ixmax = int(remove_frac*len(ebunch))

        subebunch = ebunch[:ixmax]
        g.remove_edges_from(subebunch)
        
        print(f'removing {len(ebunch[:ixmax])} out of {len(ebunch)} edges on the boundary')
        # label subsystems
        nx.set_node_attributes(g,
                               values={n: 1 for n in subnodesA},
                               name='subsystem')
        
        return g

    @staticmethod
    def nx_to_np(graph: nx.Graph):
        
        adj = nx.to_numpy_matrix(graph)
        subsystems = np.array([attr['subsystem'] for n,attr in graph.nodes.items()])
        nodes = np.array([n for n,attr in graph.nodes.items()])
        
        return {'adj': adj,
                'subsystems': subsystems,
                'nodes': nodes}
    