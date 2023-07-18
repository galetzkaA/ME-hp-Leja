#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:52:22 2021

@author: Armin Galetzka
"""

import numpy as np
from math import isclose
from scipy.special import comb


    
def find_data(data,new_knot):
    if type(new_knot) is list:
        new_knot = np.array(new_knot).reshape(1,-1)
    N = data.shape[1]-1
    # check if knot is already calculated
    ## TODO find faster solution
    if data.shape[0] > 0 :
        idx_all = data[:,:-1]*False
        for n in range(N):
            n_knots = data[:,n]
            for j,knot in enumerate(n_knots):
                val = isclose(knot,new_knot[:,n],rel_tol=np.finfo(float).eps, abs_tol=np.finfo(float).eps)
                idx_all[j,n] = val
        idx = np.where(np.alltrue(idx_all,axis=1))[0]
    else: 
        idx = np.array([])
    return idx


def estimate_TD_basis(N,M):
    # N: parameter dimension
    # M: number of samples
    p_max = 0
    while True:
        m = int(comb(N+p_max,N))
        if m == M:
            return p_max
        elif m>M:
            return p_max-1
        p_max+=1
        
    
def get_lebesque(x):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    # Computes the Lebesgue coefficient for a set of nodes on an interval
    #
    # References:
    #
    # (1) Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange 
    #     Interpolation" 
    #     http://web.comlab.ox.ac.uk/oucl/work/nick.trefethen/berrut.ps.gz
    # (2) Walter Gaustschi, "Numerical Analysis, An Introduction" (1997) p. 76+
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Get number of nodes
    n = x.shape[0]
    
    # use this finer mesh for interpolating between nodes
    N=20*n+1
    
    X = np.matlib.repmat(x.flatten(), n, 1).T
    
    # Compute the weights
    w=1./np.prod(X-X.T+np.eye(n),1)
    
    # Fine mesh for interpolating 
    xp = np.linspace(np.min(x),np.max(x),N)
    
    xdiff = np.matlib.repmat(xp,n,1) - np.matlib.repmat(x.flatten(), N, 1).T
    
    # find all the points where the difference is zero
    zerodex = xdiff==0
    
    # See eq. 3.1 in Ref (1)
    lfun = np.prod(xdiff,0)
    
    # kill zeros
    xdiff[zerodex] = np.finfo(float).eps
    
    # Compute lebesgue function
    lebfun=np.sum(np.abs(np.matmul(np.diag(w),np.matlib.repmat(lfun,n,1))/xdiff),axis=0)
    # plt.figure()
    # plt.plot(xp,lebfun)
    
    L=np.max(lebfun)
    
    return L
        
    
    