#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:07:44 2021

@author: Armin Galetzka
"""
import numpy as np
from scipy.optimize import minimize, fminbound


def new_leja_knot(x_old,leja_function,dist,discontinuous=False):
    bounds = dist.get_support()
    # search for all optima between two nodes
    bnds = np.sort(x_old).tolist()
    
    if len(bnds)==0:
        bnds = bounds
    else:
        if not np.isclose(bnds[0],bounds[0],1e-8): # check if lower bound is already in bounds
            bnds.insert(0,bounds[0])
        if not np.isclose(bnds[-1],bounds[1],1e-8): # check if upper bound is already in bounds
            bnds.append(bounds[1])


    xmax, fmax = [],[]
    for i in range(len(bnds)-1):
        bound = [(bnds[i],bnds[i+1])]


        
        # x0 in the middle of two nodes OR close to node of at the +- inf side
        if bnds[i] == -np.inf and bnds[i+1] == np.inf:
            x0 = dist.get_mean()         
        elif bnds[i] == -np.inf:
            x0 = bnds[i+1] - dist.get_variance()
        elif bnds[i+1] == np.inf:
            x0 = bnds[i] + dist.get_variance()  ##TODO what was that for?
        else:
            x0 = (bnds[i] + bnds[i+1])/2
        
        if discontinuous:
            res = fminbound(leja_function,bound[0][0],bound[0][1],args=(x_old,dist),xtol=1e-12)
            xmax.append(res)
            fmax.append(-leja_function(res,x_old,dist))
        else:
            res = minimize(leja_function,[x0],args=(x_old,dist),method='L-BFGS-B',bounds = bound,tol=1e-12)
            xmax.append(res.x[0])
            fmax.append(-res.fun)


    # calculate boundaries if not inf and middle
    if not bnds[0] == -np.inf and not bnds[-1] == np.inf:
        # xmax.append((bnds[-1]+bnds[0])/2.)
        # fmax.append(-leja_function(xmax[-1],x_old,dist))
        xmax.insert(0,(bnds[-1]+bnds[0])/2.)
        fmax.insert(0,-leja_function(xmax[-1],x_old,dist))
    if not discontinuous:
        if not bnds[0] == -np.inf:
            # xmax.insert(0,bnds[0])
            # fmax.insert(0,-leja_function(bnds[0],x_old,dist))
            xmax.append(bnds[0])
            fmax.append(-leja_function(bnds[0],x_old,dist))
        if not bnds[-1] == np.inf:
            xmax.append(bnds[-1])
            fmax.append(-leja_function(bnds[-1],x_old,dist))
    else:
        eps = 1e-12 #np.finfo(float).eps
        if not bnds[0] == -np.inf:
            xmax.insert(0,bnds[0]+eps)
            fmax.insert(0,-leja_function(bnds[0]+eps,x_old,dist))
        if not bnds[-1] == np.inf:
            xmax.append(bnds[-1]-eps)
            fmax.append(-leja_function(bnds[-1]-eps,x_old,dist))    

    
    while True:
        idx_max = np.argmax(fmax)
        if xmax[idx_max] in x_old:
            xmax.pop(idx_max)
            fmax.pop(idx_max)
        else:
            break
    return xmax[idx_max], fmax[idx_max]    
    
    
def leja_function(x,x_old,dist=None):
    if type(x) is np.ndarray: x=x[0]
    if dist:
        return -np.sqrt(dist.pdf(x))*np.prod(np.abs(x-np.asarray(x_old)))
    else:
        return -np.prod(np.abs(x-np.asarray(x_old)))
    

def seq_lj_1d(order,dist,knots=[],discontinuous=False):
    n_init  = len(knots) 
    for n in range(order+1-n_init):       
        new_knots, _ = new_leja_knot(knots,leja_function,dist,discontinuous=discontinuous)
        knots = np.append(knots,new_knots)
    weights = compute_weights(knots,dist)
    return knots,weights


def compute_weights(knots,dist):
    ## unnecessary
    # bounds = dist.get_support()
    # P = len(knots)
    # polys_per_dim = [Hierarchical1d(knots[:p+1]) for p in range(P)]
    
    # weights = []
    # for i in range(P):
    #     f = lambda x: polys_per_dim[i].evaluate(x)[0]*dist.pdf(x)
    #     w, _ = quad(f,bounds[0],bounds[1])
    #     weights.append(w)
    # return np.array(weights)
    return None

def lejaorder(knots):
    N = len(knots)
    z = {}
    
    
    for n in range(N):
        if knots[n].shape[0] > 0:
            z[n] = np.empty(knots[n].shape)*np.nan
            idxz = np.empty(knots[n].shape[0])*np.nan
            idx = np.argmax(knots[n])
            z[n][0] = knots[n][idx]
            if knots[n].shape[0] > 1:
                temp = np.abs(knots[n]-z[n][0])
                idxz[0] = idx
                idx = np.argmax(temp)
                z[n][1] = knots[n][idx]
                idxz[1] = idx
                for k in range(1,knots[n].shape[0]-1):
                    for i in range(knots[n].shape[0]):
                        temp[i] = temp[i]*np.abs(knots[n][i]-z[n][k])
                    idx = np.argmax(temp)
                    z[n][k+1] = knots[n][idx]
                    idxz[k+1] = idx
        else:
            z[n] = knots[n]
    return z
            
            
            
            
            
            
            
            
            
            
        
        
        
    