import os
import sys
import configparser
from copy import copy
import numpy as np
import networkx as nx
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt
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
    def __init__(self,degree=19,algorithm='Full'):
        self.degree = degree # maximum stable degree
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
        

class RANSAC:
    '''
    RANSAC implementation from wiki
    '''
    def __init__(self, n=10, k=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf
        self.best_inliers = None
        
    def fit(self, X, y):

        for kk in range(self.k):
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


class Observable:
    
    '''
    Base function
    '''
    def __init__(self,evals , ransac_tries=10 , poly_degree=19 , n_outliers = 60, poly_alg = 'Full', mode = 'monopoly'):
        self.evals = evals
        self.unfolded_evals = None
        self.ixs = np.arange(len(evals))
        self.poly_degree = poly_degree
        self.n_outliers = n_outliers
        self.poly_alg = poly_alg
        self.ransac_tries = ransac_tries
        self.mode = mode
        self.av_density = None
        
    def unfold(self):
        if self.mode == 'monopoly':
            model = monopoly(degree=self.poly_degree,
                             algorithm=self.poly_alg)
            self.av_density = RANSAC(model=model,
                                     n=len(self.evals)-self.n_outliers,
                                     k=self.ransac_tries,
                                     loss=square_error_loss,
                                     metric=mean_square_error)
            _ = self.av_density.fit(self.evals,self.ixs)
            self.unfolded_evals = self.av_density.predict(self.evals)
            
        elif self.mode == 'exact_goe':
            self.unfolded_evals = self._goe_cdf(self.evals,n=len(self.evals))
            
        self._check_monotonicity()
    
    def unfolded_evals_cdf(self,x):
        self._unfold_warning()
        return (self.unfolded_evals.reshape(-1,1) <= x).sum(axis=0)  
    
    def _unfold_warning(self):
        if type(self.unfolded_evals) == type(None):
            print('run unfold() first')
            
    def _check_monotonicity(self):
        if (np.diff(self.unfolded_evals)<0).any():
            print('warning, unfolding not monotonic!')
            
    def _load_unfolding(self,obs):
        self.evals = obs.evals
        self.unfolded_evals = obs.unfolded_evals
        self.ixs = obs.ixs
        self.av_density = obs.av_density
        
    def _goe_cdf(self,x,n):
        R = 2*np.sqrt(n)
        return n*(1/2 + x*np.sqrt(R**2-x**2)/(np.pi*R**2) + np.arcsin(x/R)/np.pi)
    
class NNSpacingDistribution(Observable):
    '''
    Nearest-Neighbor Spacing Distribution
    '''
    def __init__(self,evals,**kwargs):
        super().__init__(evals,**kwargs)
        
    def spacings(self):
        self._unfold_warning()
        return np.diff(self.unfolded_evals)
    
    def _prob_brody(self,s,q):
        cq = math.gamma(1/(q+1))**(q+1)/(q+1)
        return cq*s**q*np.exp(-cq/(q+1)*s**(q+1))
    
    def nnsd_goe(self,s):
        return self._prob_brody(s,1)
    
    def nnsd_poisson(self,s):
        return self._prob_brody(s,0)
    
class NumberVariance(Observable):
    '''
    Number Variance
    '''
    def __init__(self,evals,**kwargs):
        super().__init__(evals,**kwargs)
  
    def n_levels(self,xs,L):
        self._unfold_warning()
        return self.unfolded_evals_cdf(xs+L/2) - self.unfolded_evals_cdf(xs-L/2)

#     def calc_nv(self,L,n=10000):
#         self._unfold_warning()
#         minn = 1*self.unfolded_evals[0]
#         maxx = 1*self.unfolded_evals[-1]
        
#         w = 0
#         for i in range(n):

#             xs = minn + (maxx-minn)*np.random.rand()
#             # w += (self.n_levels(xs,L)-L)**2
#             w += (self.n_levels(xs,L)-L)**2

#         return w/n 
    
    def calc_nv(self,L,n=20000,eps=0):
        
        self._unfold_warning()
        # minn = (1+eps)*(self.unfolded_evals[0] + L/2)
        # maxx = (1-eps)*(self.unfolded_evals[-1] - L/2)
        minn = (1+eps)*(self.ixs[0])
        maxx = (1-eps)*(self.ixs[-1])        
        
        w1 = 0
        w2 = 0
        w3 = 0
        w4 = 0

        for i in range(n):
            xs = minn + (maxx-minn)*np.random.rand()
            w1 += self.n_levels(xs,L)
            
        for i in range(n):
            xs = minn + (maxx-minn)*np.random.rand()
            w2 += self.n_levels(xs,L)**2

        for i in range(n):
            xs = minn + (maxx-minn)*np.random.rand()
            w3 += self.n_levels(xs,L)**3

        for i in range(n):
            xs = minn + (maxx-minn)*np.random.rand()
            w4 += self.n_levels(xs,L)**4 
            
        w1 = w1/n
        w2 = w2/n
        w3 = w3/n
        w4 = w4/n
        
        nv_mean = w2-w1**2
        nv_std = w4 - 4*w1*w3 + 6*w1**2*w2 - 3*w1**4
        return nv_mean,nv_std

    def nv_goe(self,L):
        '''
        formula for GOE Number Variance (asymptotic)
        '''
        return 2/np.pi**2*(np.log(2*np.pi*L) + np.euler_gamma + 1- np.pi**2/8)
