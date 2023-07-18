#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:06:23 2022

@author: Armin Galetzka
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
from distributions import Uniform, Normal, TruncNormal
from ME_Leja import Tree as me_leja
from scipy.integrate import quad

me_leja.parameter['href_split_marker'] = 'median' # 'median', 'mean'
me_leja.parameter['marking_strategy']  = 'absolute' # 'absolute', 'maximum_strategy', 'fixed_energy_fraction'
me_leja.parameter_leja['reuse_Leja_nodes_href'] = True
me_leja.parameter_leja['sort_Leja_nodes'] = True


theta1 = 1e-4

uniform = {}
uniform[0] = Uniform([0,1])                
uniform[1] = Uniform([0,1]) 

mean  = 0.5
sigma = 0.2
normal = {}
normal[0] = Normal(mean,sigma,[-np.inf,np.inf])
normal[1] = Normal(mean,sigma,[-np.inf,np.inf])
trunc_normal = {}
trunc_normal[0] = TruncNormal(mean,sigma,[mean-3*sigma,mean+3*sigma])
trunc_normal[1] = TruncNormal(mean,sigma,[mean-3*sigma,mean+3*sigma])
trunc_normal[0].set_normalization_const(1.0)
trunc_normal[1].set_normalization_const(1.0)

"""
objective function
"""
def f(x):
    # if type(x) == list:
    #     x = np.array(x).reshape(1,-1)
    # return x[:,0]**2 + x[:,1]**3 + np.sin(np.pi*x[:,0])
    if type(x) is list:
        x = np.array(x).reshape(1,-1)
    Z = (-x[:,0]+0.1*np.sin(30*x[:,0])+np.exp(-(50*(x[:,0]-0.65))**2)) + \
        (-x[:,1]+0.1*np.sin(30*x[:,1])+np.exp(-(50*(x[:,1]-0.65))**2))
        
    if x.shape[0] == 1:
        return Z[0]
    else:
        return Z

"""
Uniform
"""
print('-------')
print('Uniform')
print('-------')
main_node = me_leja(f,dist=uniform,p_max=2)


main_node.run(theta1=theta1,theta2=0.5,thetap=0.5,no_refinements=None,verbose=False)

x,y = np.linspace(uniform[0].bounds[0],uniform[0].bounds[1],100),np.linspace(uniform[1].bounds[0],uniform[1].bounds[1],100)
X,Y = np.meshgrid(x,y)
Z = np.nan*X
Zref = np.nan*X
for i in range(x.shape[0]):
    Z[:,i] = main_node.evaluate(np.array([X[:,i],Y[:,i]]).T).flatten()
    Zref[:,i] = f(np.array([X[:,i],Y[:,i]]).T)
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('surrogate - uniform')    
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,linewidth=0, antialiased=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('f(x) - uniform')
surf = ax.plot_surface(X, Y, Zref, cmap=cm.viridis,linewidth=0, antialiased=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('absolute error - uniform')
surf = ax.plot_surface(X, Y, np.abs(Zref-Z), cmap=cm.viridis,linewidth=0, antialiased=True)


main_node.plot_mesh()

"""
Normal
"""
print('-------')
print('Normal')
print('-------')
me_leja.original_dist   = None
main_node = me_leja(f,dist=normal,p_max=2)


main_node.run(theta1=theta1,theta2=0.5,thetap=0.5,no_refinements=None,verbose=False)

x,y = np.linspace(-0.1,1.1,100),np.linspace(-0.1,1.1,100)
X,Y = np.meshgrid(x,y)
Z = np.nan*X
Zref = np.nan*X
for i in range(x.shape[0]):
    Z[:,i] = main_node.evaluate(np.array([X[:,i],Y[:,i]]).T).flatten()
    Zref[:,i] = f(np.array([X[:,i],Y[:,i]]).T)
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('surrogate - normal')    
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,linewidth=0, antialiased=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('f(x) - normal')
surf = ax.plot_surface(X, Y, Zref, cmap=cm.viridis,linewidth=0, antialiased=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('absolute error - normal')
surf = ax.plot_surface(X, Y, np.abs(Zref-Z), cmap=cm.viridis,linewidth=0, antialiased=True)

main_node.plot_mesh()

"""
Truncated Normal
"""
print('---------------')
print('Truncated Normal')
print('---------------')
me_leja.original_dist   = None
main_node = me_leja(f,dist=trunc_normal,p_max=2)

main_node.run(theta1=theta1,theta2=0.5,thetap=0.5,no_refinements=None,verbose=False)

x,y = np.linspace(trunc_normal[0].bounds[0],trunc_normal[0].bounds[1],100),np.linspace(trunc_normal[1].bounds[0],trunc_normal[1].bounds[1],100)
X,Y = np.meshgrid(x,y)
Z = np.nan*X
Zref = np.nan*X
for i in range(x.shape[0]):
    Z[:,i] = main_node.evaluate(np.array([X[:,i],Y[:,i]]).T).flatten()
    Zref[:,i] = f(np.array([X[:,i],Y[:,i]]).T)
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('surrogate - truncated normal')    
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,linewidth=0, antialiased=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('f(x) - truncated normal')
surf = ax.plot_surface(X, Y, Zref, cmap=cm.viridis,linewidth=0, antialiased=True)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-63, elev=47)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('absolute error - truncated normal')
surf = ax.plot_surface(X, Y, np.abs(Zref-Z), cmap=cm.viridis,linewidth=0, antialiased=True)

main_node.plot_mesh()

main_node.plot_pmax_on_mesh()

plt.show()