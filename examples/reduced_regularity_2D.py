#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:18:32 2021

@author: Armin Galetzka
"""

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../software')
from distributions import Uniform
from ME_Leja import Tree as me_leja

# general parameters
verbose = True
np.random.seed(1)         # set seed
Ncv = 10**4               # number of validation samples

# get objective function
def f(x):
    if type(x) is list:
        x = np.array(x).reshape(1,-1)
    delta = 0.1

    return 1./(np.abs(0.3 - x[:,0]**2 - x[:,1]**2) + delta)

# get distribution
N = 2
a,b = 0,1                    # bounds for distribution
jpdf = {}
for i in range(N):
    jpdf[i] = Uniform([a,b])     

# generate validation set
cv_in = np.array([jpdf[i].rvs(size=(Ncv)) for i in range(N)]).T
cv_out = f(cv_in)

# run algorithm

p_max = 2      # TD or TP-degree for elements
thetap = 0.4   # controls hp-adaptivity
theta2 = 1.0   # controls adaptivity for h-refinement
      
me_leja.parameter['href_split_marker'] = 'median'   # 'median', 'mean'
me_leja.parameter['marking_strategy']  = 'absolute' # 'absolute', 'maximum_strategy', 'fixed_energy_fraction'
me_leja.parameter_leja['reuse_Leja_nodes_href'] = True
me_leja.parameter_leja['sort_Leja_nodes'] = True
        
main_node = me_leja(f,discontinuous=False,dist=jpdf,p_max=p_max)

cv_error_rms=[]
fcalls = []
elements = []
saving_rel_percent = []
theta1s = [1e-1,1e-2,1e-3,1e-4,1e-5,5e-6,1e-6,5e-7,1e-7,1e-8,5e-9,1e-9,5e-10,1e-10]
for ii,theta1 in enumerate(theta1s):
    
    print('theta_1: ' + str(theta1))
    main_node.run(theta1=theta1,theta2=theta2,thetap=thetap,no_refinements=None,verbose=False)
    
    # save elements
    elements.append(len(main_node.get_leaves()))

    # cv error     
    print('calculate cv error')
    print('')
    cv_sur = main_node.evaluate(cv_in).flatten()
    errs = np.abs(cv_sur-cv_out)
    cv_error_rms.append(np.sqrt(np.sum(errs**2)/Ncv))
    fcalls.append(main_node.fcalls)
    if me_leja.parameter_leja['reuse_Leja_nodes_href']:
        try: 
            saving_rel_percent.append(100 - (main_node.fcalls_lost/main_node.fcalls_possible_saving)*100)
            fcalls_init, fcalls_local, fcalls_total = main_node.get_savings()
        except Exception:
            saving_rel_percent.append(np.nan)
    else:
        saving_rel_percent.append(0)
    print('fcalls: {} | error: {:.3e}'.format(fcalls[-1],cv_error_rms[-1]))
    print('')
    
    if fcalls[-1] >= 1000:
        break


# plot convergence (RMS error over function evaluatins)
plt.figure()
plt.semilogy(fcalls,cv_error_rms,'-s')   
plt.xlabel('function evaluations')
plt.ylabel('RMS error')


# plot surrogate and objective function
leaves = main_node.get_leaves()
data = None
for leaf in leaves:
    if data is None:
        data = leaf.data
    else:
        data = np.vstack([data,leaf.data])


x,y = np.linspace(jpdf[0].a,jpdf[0].b,100),np.linspace(jpdf[1].a,jpdf[1].b,100)
X,Y = np.meshgrid(x,y)
Z = np.nan*X
Zref = np.nan*X
for i in range(x.shape[0]):
    Z[:,i] = main_node.evaluate(np.array([X[:,i],Y[:,i]]).T).flatten()
    Zref[:,i] = f(np.array([X[:,i],Y[:,i]]).T)

fig = plt.figure(figsize=plt.figaspect(0.4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Surrogate')
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,linewidth=0, antialiased=True)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Objective function')
surf = ax.plot_surface(X, Y, Zref, cmap=cm.viridis,linewidth=0, antialiased=True)

# plot mesh and knots
ll = main_node.get_leaves()
data = main_node.get_data()[:,:-1]
main_node.plot_mesh()
plt.plot(data[:,0],data[:,1],'.',color='black',markersize=5)
plt.xlabel('x_1')
plt.ylabel('x_2')

# plot polynomial degree on mesh
main_node.plot_pmax_on_mesh(colormap='viridis')
plt.title('Polynomial degree per element')

# plot validation error on mesh
main_node.plot_error_on_mesh(cv_in, cv_out, error_norm='rms', colormap='cubehelix')
plt.title('RMS error')

plt.show()
