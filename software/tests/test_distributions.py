#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:06:55 2022

@author: Armin Galetzka
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from distributions import Uniform, Normal, TruncNormal, Beta
from ME_Leja import Tree as me_leja
from scipy.integrate import quad

me_leja.parameter['href_split_marker'] = 'median' # 'median', 'mean'
me_leja.parameter['marking_strategy']  = 'absolute' # 'absolute', 'maximum_strategy', 'fixed_energy_fraction'
me_leja.parameter_leja['reuse_Leja_nodes_href'] = True
me_leja.parameter_leja['sort_Leja_nodes'] = True



uniform = Uniform([-1,2])

mean  = 0.289
sigma = 0.8
normal = Normal(mean,sigma,[-np.inf,np.inf])
trunc_normal = TruncNormal(mean,sigma,[0.289-2,0.289+2])
trunc_normal.set_normalization_const(1.0)
beta = Beta(2,2,0,1,[0,1])


"""
objective function
"""
def f(x):
    return 2*x**2+0.5*x+2*np.sin(np.pi*x)


ref_mean_uniform      = quad(lambda x : f(x)*uniform.pdf(x),uniform.bounds[0],uniform.bounds[1])[0]
ref_mean_normal       = quad(lambda x : f(x)*normal.pdf(x),normal.bounds[0],normal.bounds[1])[0]
ref_mean_trunc_normal = quad(lambda x : f(x)*trunc_normal.pdf(x),trunc_normal.bounds[0],trunc_normal.bounds[1])[0]
ref_mean_beta         = quad(lambda x : f(x)*beta.pdf(x),beta.bounds[0],beta.bounds[1])[0]

XX=uniform.rvs(int(1e7))
ref_mean_MC_uniform = np.mean(f(XX))
ref_variance_MC_uniform = np.var(f(XX))

XX=normal.rvs(int(1e7))
ref_mean_MC_normal = np.mean(f(XX))
ref_variance_MC_normal = np.var(f(XX))

XX=trunc_normal.rvs(int(1e7))
ref_mean_MC_trunc_normal = np.mean(f(XX))
ref_variance_MC_trunc_normal = np.var(f(XX))

XX=beta.rvs(int(1e7))
ref_mean_MC_beta = np.mean(f(XX))
ref_variance_MC_beta = np.var(f(XX))

"""
Uniform
"""
print('-------')
print('Uniform')
print('-------')
main_node = me_leja(f,dist=[uniform],p_max=2)


main_node.run(theta1=1e-12,theta2=0.5,thetap=0.5,no_refinements=20,verbose=False)

mean_uniform = main_node.get_moments()[0]
variance_uniform = main_node.get_moments()[1]

"""
Normal
"""
print('-------')
print('Normal')
print('-------')
me_leja.original_dist   = None
main_node = me_leja(f,dist=[normal],p_max=2)


main_node.run(theta1=1e-12,theta2=0.5,thetap=0.5,no_refinements=20,verbose=False)

mean_normal = main_node.get_moments()[0]
variance_normal = main_node.get_moments()[1]

"""
Truncated Normal
"""
print('---------------')
print('Truncated Normal')
print('---------------')
me_leja.original_dist   = None
main_node = me_leja(f,dist=[trunc_normal],p_max=2)

main_node.run(theta1=1e-12,theta2=0.5,thetap=0.3,no_refinements=20,verbose=False)

mean_trunc_normal = main_node.get_moments()[0]
variance_trunc_normal = main_node.get_moments()[1]


"""
Beta
"""
print('---------------')
print('Beta')
print('---------------')
me_leja.original_dist   = None
main_node = me_leja(f,dist=[beta],p_max=2)


main_node.run(theta1=1e-12,theta2=0.5,thetap=0.1,no_refinements=20,verbose=False)

mean_beta = main_node.get_moments()[0]
variance_beta = main_node.get_moments()[1]


"""
Output
"""

print('')
print('error mean uniform: ' + str(np.abs(ref_mean_uniform - mean_uniform)))
print('error mean normal: ' + str(np.abs(ref_mean_normal - mean_normal)))
print('error mean truncated normal: ' + str(np.abs(ref_mean_trunc_normal - mean_trunc_normal)))
print('error mean beta: ' + str(np.abs(ref_mean_beta - mean_beta)))

print('')
print('error variance uniform: ' + str(np.abs(ref_variance_MC_uniform - variance_uniform)))
print('error variance normal: ' + str(np.abs(ref_variance_MC_normal - variance_normal)))
print('error variance truncated normal: ' + str(np.abs(ref_variance_MC_trunc_normal - variance_trunc_normal)))
print('error variance beta: ' + str(np.abs(ref_variance_MC_beta - variance_beta)))
