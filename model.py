# import warnings
import numpy as np
import numba
from typing import Optional,Union
import networkx as nx


class SERModel:
    """
    Model simulating neurons connected via connection matrix. The neurons have three possible states:
        Susceptible (0)
        Excited (1)
        Refractory (-1)
    Using (-1, 0, 1) instead of (0, 1, 2) is slightly faster and easier to implement efficiently.

    Parameters:
        n_steps: int
            Number of time steps of the simulation.
        prob_spont_act: float 0-1
            Probability of spontaneous process S -> E
        prob_recovery: float 0-1
            Probability of recovery, process: R -> S
        threshold: float
            (for a weighted network, as we use)
            Minimum weighted sum over active neighbours to activate a node.
        prop_e: float 0-1
            Proportion of nodes in excited state in the initial state of the system. It can be set to zero,
            then the simulation need more time to "initialize" due to very low probability of spontaneous activation.

    """

    def __init__(
            self, *,
            n_steps: int,
            prob_spont_act: float,
            prob_recovery: float,
            threshold: float,
            prop_e: float,
            n_transient: int = 0
    ) -> None:
        self.n_steps = n_steps
        self.prob_spont_act = prob_spont_act
        self.prob_recovery = prob_recovery
        self.threshold = threshold
        self.prop_e = prop_e
        self.n_transient = n_transient

    @staticmethod
    def init_state(*, n_nodes: int, prop_e: float) -> np.ndarray:
        """
        Prepare an initial state of the system with prop_e as a proportion of excited states.
        The routine assumes that in the initial state we have only Susceptible and Excited states.

        Parameters:
        :param n_nodes: int
            Number of nodes
        :param prop_e: float 0-1
            Proportion of excited nodes

        Returns:
        states: 1-D np.ndarray of length n_nodes

        Activity encoded as follows:
            Susceptible: 0
            Excited: 1
            Refractory: -1

        """

        if prop_e is None:
            raise ValueError("prop_e must be defined!")
        if prop_e > 1.0:
            raise ValueError("prop_e must be <=1, now it is given prop_e={}".format(prop_e))

        # Initialize vector (assuming -1 as Refractory):
        states = np.zeros(n_nodes, dtype=np.int)

        # Compute number of excited nodes:
        n_nodes_e = int(round(n_nodes * prop_e, 2))

        # Set states:
        states[: n_nodes_e] = 1
        np.random.shuffle(states)

        # Print a warning if not all possible states (-1, 0, 1) in the initial state
        # In fact we do not care about that...
        # if len((set(states))):
        #     warnings.warn("Warning: not all states are present in the initial state!")

        return states

    def simulate(self, *,
                 adj_mat: np.ndarray,
                 states: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """
        Run the simulation of the brain model using given connection matrix.

        Parameters
        :param adj_mat: np.ndarray
            Connection (adjacency) matrix of the system.
        :param states: 1-D np.ndarray, optional
            Initial state of the system. If None, random initial state is generated using brain_model class parameters.

        Returns:
        act_mat: 2D np.ndarray (n_nodes, n_steps)
            Activity matrix (node vs time) with:
                Susceptible: 0
                Excited: 1
                Refractory: -1
        """

        # Initialize the state if needed:
        if states is None:
            states = SERModel.init_state(n_nodes=len(adj_mat), prop_e=self.prop_e)

        states = states.astype(adj_mat.dtype)  # cast for numba

        # external routine, necessary for numba acceleration
        return _run(
            adj_mat=adj_mat,
            states=states,
            n_steps=self.n_steps,
            prob_spont_act=self.prob_spont_act,
            prob_recovery=self.prob_recovery,
            threshold=self.threshold,
            n_transient=self.n_transient
        )


@numba.njit(fastmath=True)
def _run(
        adj_mat: np.ndarray,
        states: np.ndarray,
        n_steps: int,
        prob_spont_act: float,
        prob_recovery: float,
        threshold: float,
        n_transient: int = 0
        ) -> np.ndarray:
    """
    :param adj_mat: np.ndarray
        Connection matrix.
    :param states: np.ndarray
        Initial state of the system.
    :param n_steps: int
        Length of the time series.
    :param prob_spont_act: float 0-1
        Probability of spontaneous activation/
    :param prob_recovery: float 0-1
        Probability of relaxation from R to S state.
    :param threshold: float
        Activation threshold.
    :param n_transient: int
        Time of initial simulation which is discarded. If n_transient > 0 then the full time series will have
        (n_steps - n_transient) time steps in total.
    :return: np.ndarray
        2-D activity matrix (node vs time).
    """
    _dtype = adj_mat.dtype
    n_nodes = len(adj_mat)

    # Initialize activity matrix
    act_mat = np.zeros((n_nodes, n_steps), dtype=_dtype)
    act_mat[:, 0] = states

    # Evaluate all the stochastic transition probabilities in advance:
    spont_activated = np.random.random(act_mat.shape) < prob_spont_act
    recovered = np.random.random(act_mat.shape) < prob_recovery

    for t in range(n_steps - 1):
        # E -> R
        act_mat[act_mat[:, t] == 1, t + 1] = -1
        # R -> S stochastic:
        refrac = act_mat[:, t] == -1
        act_mat[refrac, t + 1] = act_mat[refrac, t] + recovered[refrac, t]

        # S -> E threshold + stochastic:
        # (act_mat[:, t] == 1).astype(_dtype) is a vector of 0s and 1s where we have active nodes!
        weighed_neigh_input = adj_mat.T @ (act_mat[:, t] == 1).astype(_dtype)
        susce = act_mat[:, t] == 0
        act_mat[susce, t + 1] += np.logical_or(weighed_neigh_input[susce] >= threshold,
                                               spont_activated[susce, t])

    return act_mat[:, n_transient:]


def calc_Tc(J) -> float:
    return 2*J/np.log(np.sqrt(2)+1)

class IsingModel:
    
    def __init__(self,
                 n_steps: int,
                 T: float,
                 J: float,
                 network: Union[nx.Graph,np.ndarray],
                 n_transient: int = 500) -> None:
        self.n_steps = n_steps
        self.T = T
        self.J = J
        self.n_transient = n_transient
        
        if type(network) == np.ndarray:
            self.network = nx.from_numpy_array(network)
        elif type(network) == nx.Graph:
            self.network = network
        else:
            raise Error('unknown network type')
            
    def E(self,s: dict) -> float:
        E0 = 0
        for n in self.network.nodes:
            nn = np.array([s[n2] for _,n2 in self.network.edges(nbunch=n)]).sum()
            E0 -= self.J * s[n] * nn
        return E0

    def init_state(self) -> dict:
        # return {n: 2*np.random.randint(2)-1 for n in self.network.nodes}
        return {n: -1 for n in self.network.nodes}

    def sweep(self,s: dict) -> None:
        for n in self.network.nodes:
            nn = np.array([s[n2] for _,n2 in self.network.edges(nbunch=n)]).sum()

            new_s = -s[n]
            dE =- self.J * (new_s-s[n])*nn

            if dE <= 0.:
                s[n] = new_s
            elif np.exp(-dE/self.T) > np.random.rand():
                s[n] = new_s        

    def simulate(self) -> np.ndarray:
        def s_arr(s: dict): 
            return np.array(list(s.values()))
        # print(f'running sim grid={grid}; J={J}; T={T}')

        s = self.init_state()
        X = s_arr(s).reshape(-1,1)

        # print('thermalization...')
        for i in range(self.n_transient):
            self.sweep(s)    

        # print('simulation')
        # for i in tqdm(range(self.n_steps-1)):
        for i in range(self.n_steps-1):
            self.sweep(s)
            X = np.hstack((X,s_arr(s).reshape(-1,1)))
        return X

# import numpy.linalg as LA
# from scipy.io import loadmat

# @numba.jit()
# def compute_activity(nsteps: int, T: float, connection_mat: np.ndarray, pinit: float = 0.01, seed: int=12345) -> np.array:
#     """
#     Greenberg-Hastings model for connection_mat. Initial state with pinit (=0.01 default)
#     initial active nodes.
#     There are three possible states:
#                     Q - quiescent:   0
#                     E - excited:     1
#                     R - refractory:  2
#     Transition rules:
#         1.1) Q -> E with some small probability r1 = 0.001
#         1.2) Q -> E if the of the connection weightw w_ij with the active neighbors j
#              is higher than the threshold T
#         2) E -> R always
#         3) R -> Q with a small probability r2 = 0.2, what delays the transition from
#            between refractory and quiescent states
#     """
#
#     np.random.seed(seed=int(seed))
#     r1 = 0.001  # spontaneous activation probability
#     r2 = 0.2  # refractory to Quiscent prob.
#
#     N = (connection_mat.shape)[0]
#     Ninit = int(N * 0.01)
#     S = np.zeros((nsteps, N))
#
#     S[0, np.random.randint(0, N, Ninit)] = 1  # initialize with Ninit active neurons
#
#     for it in range(1, nsteps):
#         S[it] = S[it - 1]
#         S[it, S[it - 1] == 1] = 2  # Previously active goes to Refractory period, point 2)
#         R = S[it - 1] == 2  # Old refractory
#         S[it, R * (np.random.random(N) < r2)] = 0  # Refractory nodes go back to quiescent state with prob. r2, reg
#         Q = S[it - 1] == 0
#         S[it, Q * (np.random.random(N) < r1)] = 1  # Spontaneous activation of quiescent nodes with prob. r1
#
#         E = S[it - 1] == 1
#         Mmat1 = np.copy(connection_mat)
#         Mmat1[E] = 0
#         Mmat1[R] = 0
#         Mmat1[:, Q] = 0
#         Mmat1[:, R] = 0
#         W = np.sum(Mmat1, axis=1)
#         S[it, W > T] = 1
#     #
#     return S.astype(float)
#     # return 0
