#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:21:30 2021

@author: Dimitrios Loukrezis

Returns the Total Degree multi-index set given N parameters (N>=1, integer) 
and polynomial level w (w>=0, integer).

The code is based on source code from the Sparse-Grids-Matlab-Kit, 
v-18-10_Esperanza, by Lorenzo Tamelini and Fabio Nobile.
"""


from scipy.special import comb
import numpy as np


def setsize(N, w):
    return int(comb(N+w-1, N-1))    


def td_set_recursive(N, w, rows):
    if N == 1:
        subset = w*np.ones([rows, 1])
    else:
        if w == 0:
            subset = np.zeros([rows, N])
        elif w == 1:
            subset = np.eye(N)
        else:
            # initialize submatrix
            subset = np.empty([rows, N])
            
            # starting row of submatrix
            row_start = 0
            
            # iterate by polynomial order and fill the multiindex submatrices
            for k in range(0, w+1):
                
                # number of rows of the submatrix
                sub_rows = setsize(N-1, w-k)
                
                # update until row r2
                row_end = row_start + sub_rows - 1
                
                # first column
                subset[row_start:row_end+1, 0] = k*np.ones(sub_rows)
                
                # subset update --> recursive call
                subset[row_start:row_end+1, 1:] = td_set_recursive(N-1, w-k, 
                                                              sub_rows)
                                                                     
                # update row indices
                row_start = row_end + 1
    
    return subset


def td_multiindex_set(N, w):
    """Returns the multiindex set for a total-degree PCE with N parameters and 
    maximum polynomial degree w"""
    
    # size of the total degree multiindex set
    td_size = int(comb(N+w, N))
    
    # initialize total degree multiindex set
    midx_set = np.empty([td_size, N])
    
    # starting row
    row_start = 0
    
    # iterate by polynomial order
    for i in range(0, w+1):
        
        # compute number of rows
        rows = setsize(N, i)
        
        # update up to row r2
        row_end = rows + row_start - 1
        
        # recursive call 
        midx_set[row_start:row_end+1, :] = td_set_recursive(N, i, rows)
        
        # update starting row
        row_start = row_end + 1
        
    return midx_set.astype(int)


