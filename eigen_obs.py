import os
import sys
import configparser
from copy import copy
import numpy as np
import networkx as nx
import scipy.io
import scipy.optimize

import pandas as pd
import math
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
# from rpy2.robjects import numpy2ri

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class monopoly(BaseEstimator):
    '''
    Wrapper around MonoPoly R package
    '''
    def __init__(self,
                 degree=19,
                 algorithm='Full'):
        
        self.degree = degree # maximum stable degree is 19
        self.algorithm = algorithm
        
    def fit(self,X,y):

        utils = rpackages.importr('utils')
        _ = utils.chooseCRANmirror(ind=1)
        if not rpackages.isinstalled('MonoPoly'):
            _ = utils.install_packages("MonoPoly",quiet=True,verbose=False,clean=True)
        self.mp_ = rpackages.importr('MonoPoly')

        X, y = check_X_y(X, y, accept_sparse=True,ensure_2d=False)
            
        # numpy2ri.activate()
        # robjects.globalenv['xdata'] = X
        # robjects.globalenv['ydata'] = y
        
        xdata = robjects.FloatVector(X)
        ydata = robjects.FloatVector(y)
        robjects.globalenv['xdata'] = xdata
        robjects.globalenv['ydata'] = ydata

        output = self.mp_.monpol("ydata ~ xdata",degree=self.degree,algorithm=self.algorithm)
        self.coef_ = robjects.r.coef(output)
        
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self
    
    def predict(self,X):
        X = check_array(X, accept_sparse=True,ensure_2d=False)
        
        check_is_fitted(self, 'is_fitted_')
        xdata = robjects.FloatVector(X.reshape(-1))
        yhat = self.mp_.evalPol(xdata,self.coef_)
        return np.array(yhat,dtype=np.float64)
        
    def score(self,X,y):
        yhat = self.predict(X)
        return -sklearn.metrics.mean_squared_error(y, yhat)
    
    def get_params(self, deep=True):
        return {"degree": self.degree,
                "algorithm": self.algorithm}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
class RANSAC:
    '''
    RANSAC implementation from wiki; used for outlier-robust curve fit using monopoly
    '''
    def __init__(self, n=10, k0=5, kmax=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.kmax = kmax              # `kmax`: Maximum iterations allowed
        self.k0 = k0            # 'k0': Maximum number of improvements
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf
        self.best_inliers = None
        
    def fit(self, X, y):
        k0 = 0
        for kk in range(self.kmax):
            # print(kk)
            # print(self.best_error)
            ids = np.random.default_rng().permutation(X.shape[0])

            maybe_inliers = ids[: self.n]
            maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

            thresholded = (
                self.loss(y[ids][self.n :], maybe_model.predict(X[ids][self.n :]))
                < self.t
            )

            inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])

                this_error = self.metric(
                    y[inlier_points], better_model.predict(X[inlier_points])
                )

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = maybe_model
                    self.best_inliers = inlier_points
                    k0+=1
                    
            if k0==self.k0:
                # print(kk,k0)
                break

        if self.best_error == np.inf:
            print('RANSAC failed')
            
        return self

    def predict(self, X):
        return self.best_fit.predict(X)
    
def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]

def goe(n):
    '''
    standard GOE matrix 
    '''
    t = np.tril(np.random.randn(n,n),k=-1)
    d = np.diag(np.random.randn(n))
    m = d + t + t.T

    return m

class RMTStatistic:
    '''
    Class implementing both Nearest-Neighbor Spacing Distribution and Number Variance statistics
    '''
    
    FIT_MODE_MONOPOLY_RANSAC = 'monopoly_ransac'
    FIT_MODE_MONOPOLY = 'monopoly'
    FIT_MODE_GAUSSIAN = 'gaussian'
    FIT_MODE_EXACT_GOE = 'exact_goe'
    FIT_MODE_EXACT_POISSON = 'exact_poisson'
    
    MONOPOLY_RANSAC_PARAMS = {'poly_degree': 19,
                              'poly_alg': 'Full',
                              'maxiter': 100,
                              'maximprov': 2,
                              'n_outliers': 60}
    
    MONOPOLY_PARAMS = {'poly_degree': 19,
                       'poly_alg': 'Full'}
    
    GAUSSIAN_PARAMS = {'alpha': 4,
                       'gamma': 0.608}
    
    DEFAULT_PARAMS = {FIT_MODE_MONOPOLY_RANSAC: MONOPOLY_RANSAC_PARAMS,
                      FIT_MODE_MONOPOLY: MONOPOLY_PARAMS,
                      FIT_MODE_GAUSSIAN: GAUSSIAN_PARAMS,
                      FIT_MODE_EXACT_GOE : dict(),
                      FIT_MODE_EXACT_POISSON : dict()}
    
    def __init__(self,
                 evals : np.ndarray,
                 fit_mode : str = FIT_MODE_MONOPOLY, #monopoly_ransac, monopoly, gaussian, exact_goe, exact_poisson
                 fit_params : dict = MONOPOLY_PARAMS):
        
        self.evals = np.sort(evals)
        self.fit_mode = fit_mode
        self.fit_params = self.DEFAULT_PARAMS[fit_mode]
        for k,v in fit_params.items():
            self.fit_params[k] = v
        
        self.ixs = np.arange(len(self.evals))
        self.unfolded_evals = None
        self.model = None
        
    def unfold(self):
        
        if self.fit_mode == self.FIT_MODE_MONOPOLY_RANSAC:
            
            model0 = monopoly(degree = self.fit_params['poly_degree'],
                             algorithm = self.fit_params['poly_alg'])
            
            self.model = RANSAC(model = model0,
                                     n = len(self.evals)-self.fit_params['n_outliers'],
                                     k0 = self.fit_params['maximprov'],
                                     kmax = self.fit_params['maxiter'],
                                     loss = square_error_loss,
                                     metric = mean_square_error)
            
            _ = self.model.fit(self.evals,self.ixs)
            self.unfolded_evals = self.model.predict(self.evals)
        
        elif self.fit_mode == self.FIT_MODE_MONOPOLY:
            
            self.model = monopoly(degree = self.fit_params['poly_degree'],
                                  algorithm = self.fit_params['poly_alg'])
            
            _ = self.model.fit(self.evals,self.ixs)
            self.unfolded_evals = self.model.predict(self.evals)
            
        elif self.fit_mode == self.FIT_MODE_GAUSSIAN:

            alpha = self.fit_params['alpha']
            deltas = []
            ixs = []
            for i in range(alpha,len(self.evals)-alpha):
                deltas += [(self.evals[i+alpha] - self.evals[i-alpha])/(2*alpha)]
                ixs += [i]

            mu = self.evals[alpha:-alpha]
            sigma = self.fit_params['gamma']*alpha*np.array(deltas)
            # sigma = np.array(.02)
            def nav(x):
                a = (x.reshape(-1,1) - mu.reshape(1,-1))/(np.sqrt(2)*sigma.reshape(1,-1))
                return (1/2*(1+scipy.special.erf(a))).sum(axis=-1)
            
            self.unfolded_evals = nav(self.evals)
            
        elif self.fit_mode == self.FIT_MODE_EXACT_GOE:
            
            self.unfolded_evals = self._goe_cdf(self.evals,n = len(self.evals))
            
        elif self.fit_mode == self.FIT_MODE_EXACT_POISSON:
            
            self.unfolded_evals = self._poisson_cdf(self.evals,n = len(self.evals))
            
        self._check_monotonicity()
    
    def _unfolded_evals_cdf(self,
                            x : np.ndarray):
        
        self._unfold_warning()
        # return (self.unfolded_evals.reshape(-1,1) <= x).sum(axis=0)  
        newshape = (-1,)+tuple([1]*len(x.shape))
        uevals = self.unfolded_evals.reshape(*newshape)
        sh2 = (len(self.unfolded_evals),)
        sh2 += x.shape
        uevals = np.broadcast_to(uevals,sh2)
        return (uevals <= x).sum(axis=0)  
    
    def _unfold_warning(self):
        
        if type(self.unfolded_evals) == type(None):
            print('run unfold() first')
            
    def _check_monotonicity(self):
        
        if (np.diff(self.unfolded_evals)<0).any():
            print('warning, unfolding not monotonic!')
        
    def _goe_cdf(self,
                 x : np.ndarray,
                 n : float):
        
        out = np.zeros_like(x)
        R = 2*np.sqrt(n)
        mask_m = x < -R
        mask_p = x > R
        mask_in = (x>= -R) & (x <= R)
        
        out[mask_m] = 0
        out [mask_p] = 1
        x0 = x[mask_in]
        
        out[mask_in] = n*(1/2 + x0*np.sqrt(R**2-x0**2)/(np.pi*R**2) + np.arcsin(x0/R)/np.pi)
        return out
    
    def _poisson_cdf(self,
                     x : np.ndarray,
                     n : float):
        
        return n*(1-np.exp(-x))
    
    
    # Nearest Neighor Spacing
    
    def calc_nnsd(self,
                  trim_outliers : bool = False):
        
        self._unfold_warning()
        spacings = np.diff(self.unfolded_evals)
        
        if trim_outliers:
            mask = spacings<5
            return spacings[mask]
        else:
            return spacings
    
    def _prob_brody(self,
                    s : np.ndarray,
                    q : float):
        
        cq = math.gamma(1/(q+1))**(q+1)/(q+1)
        return cq*s**q*np.exp(-cq/(q+1)*s**(q+1))
    
    def nnsd_goe(self,
                 s: np.ndarray):
        
        return self._prob_brody(s,1)
    
    def nnsd_poisson(self,
                     s : np.ndarray):
        
        return self._prob_brody(s,0)
    
    def nnsd_picketfence(self,
                         s : np.ndarray):
        
        return np.ones_like(s)
    
    # Number Variance
    
    def _n_levels_plus(self,
                      xs : np.ndarray,
                      L : float,
                      nv_symm : float = 0.5):
        
        self._unfold_warning()
        return self._unfolded_evals_cdf(xs + nv_symm*L)

    def _n_levels_minus(self,
                       xs : np.ndarray,
                       L : float,
                       nv_symm : float = 0.5):
        
        self._unfold_warning()
        return self._unfolded_evals_cdf(xs - (1-nv_symm)*L)
    
    def _n_levels(self,
                 xs : np.ndarray,
                 L : float):
        
        return self._n_levels_plus(xs,L) - self._n_levels_minus(xs,L)
    
    def calc_nv(self,
                L : np.ndarray,
                n : int = 3000,
                eps : float = 0.0,
                alpha : float = 0.5,
                nv_symm : float = 0.5):
        
        self._unfold_warning()
        minn = (1+eps)*(self.unfolded_evals[0] + (1-nv_symm)*L)
        maxx = (1-eps)*(self.unfolded_evals[-1] - nv_symm*L)

        xs = minn.reshape(1,-1) + (maxx-minn).reshape(1,-1)*np.random.rand(n,len(L))

        L0 = L.reshape(1,-1)
        np_sq = self._n_levels_plus(xs,L0,nv_symm)**2
        nm_sq = self._n_levels_minus(xs,L0,nv_symm)**2
        np_ = self._n_levels_plus(xs,L0,nv_symm)
        nm_ = self._n_levels_minus(xs,L0,nv_symm)
        npnm = self._n_levels_plus(xs,L0,nv_symm)*self._n_levels_minus(xs,L0,nv_symm)

        np_sq = np_sq.mean(axis=0)
        nm_sq = nm_sq.mean(axis=0)
        np_ = np_.mean(axis=0)
        nm_ = nm_.mean(axis=0)
        npnm = npnm.mean(axis=0)    
        
        return np_sq - np_**2 + nm_sq - nm_**2 - 2*(npnm - np_*nm_)

    def nv_goe(self,
               L : np.ndarray):
        '''
        formula for GOE Number Variance (asymptotic)
        '''
        return 2/np.pi**2*(np.log(2*np.pi*L) + np.euler_gamma + 1- np.pi**2/8)

    def nv_poisson(self,
                   L : np.ndarray):
        '''
        formula for Poisson Number Variance
        '''        
        return L
    
    
