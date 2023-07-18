#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:16:05 2021

@author: Armin Galetzka
"""

from scipy.stats import uniform, norm, truncnorm
from scipy.stats import beta as beta_dist
from scipy.optimize import fsolve
from scipy.integrate import quad
import numpy as np


class Uniform():    
    def __init__(self, bounds):
        self.cp_pdf = uniform(bounds[0],bounds[1]-bounds[0])
        self.bounds = bounds
        self.type = 'uniform'
        self.a = bounds[0]                                      # original support 
        self.b = bounds[1]                                      # original support 
        self.mean = 0.5*(self.a+self.b)                         # original mean
        self.var  = 1./12.*(self.b-self.a)**2                   # original variance
        self.local_mean = None
        self.local_var  = None
        self.norm_const = None
        
    def pdf(self,x):
        return self.cp_pdf.pdf(x)
    
    def get_mean(self):
        if not self.local_mean:
            self.local_mean = 0.5*(self.bounds[0]+self.bounds[1])            
        return self.local_mean
    
    def get_variance(self):
        if not self.local_var:
            self.local_var = 1./12.*(self.bounds[1]-self.bounds[0])**2
        return self.local_var
    
    def get_support(self):
        return self.bounds
    
    def get_normalization_const(self):
        if not self.norm_const:
            self.norm_const = (self.bounds[1]-self.bounds[0])/(self.b-self.a)
        return self.norm_const

    def rvs(self,size=1):
        # sample from pdf
        return self.cp_pdf.rvs(size=size)
            
    def set_support(self,bounds):
        # current bounds of pdf in the current element
        self.bounds = bounds
        
    
    def get_copy(self):
        copy = Uniform([self.a,self.b])
        # save original support
        copy.a = self.a
        copy.b = self.b
        # save current support
        copy.bounds = self.bounds
        # save original mean and variance
        copy.mean = self.mean
        copy.var  = self.var
        copy.norm_const = self.norm_const
        # save local mean and variance
        copy.local_mean = self.local_mean
        copy.local_var = self.local_var
        
        return copy
    

        
class Normal():    
    def __init__(self, mean, sigma, bounds):
        self.local_mean = None
        self.local_var  = None
        self.local_median = None
        self.norm_const = None
        self.a = bounds[0]             # original bounds 
        self.b = bounds[1]             # original bounds 
        self.cp_pdf = norm(loc=mean,scale=sigma)
        
        # if support is unbounded -> set support to +-8.22*sigma2
        if bounds[0] == -np.inf:
            bounds[0] = mean-8.22*sigma
        if bounds[1] == np.inf:
            bounds[1] = mean+8.22*sigma
            
        self.set_support(bounds)
        self.type = 'normal'
        self.mean = mean               # original mean
        self.var  = sigma              # original variance

        
    def pdf(self,x):
        return self.cp_pdf.pdf(x)
    
    def get_mean(self):
        if self.local_mean is None: 
            def f(x):
                return x*self.pdf(x)/self.get_normalization_const()
            self.local_mean = quad(f,self.bounds[0],self.bounds[1])[0]            
        return self.local_mean
    
    def get_variance(self):
        if self.local_var is None:
            def f(x):
                return (x-self.get_mean())**2*self.pdf(x)/self.get_normalization_const()
            self.local_var = quad(f,self.bounds[0],self.bounds[1])[0]            
        return self.local_var
    
    def get_median(self):
        if self.local_median is None: 
            def f1(x):
                return self.pdf(x)/self.get_normalization_const()
            def f2(y):
                return quad(f1,self.bounds[0],y)[0] - 0.5
            self.local_median = fsolve(f2,(self.bounds[1]+self.bounds[0])/2)[0]            
        return self.local_median
    
    def get_support(self):
        return self.bounds
    
    def rvs(self,size=1):
        # sample from pdf
        return self.cp_pdf.rvs(size=size)
    
    def get_normalization_const(self):
        if not self.norm_const:
            def f(x):
                return self.pdf(x)
            self.norm_const = quad(f,self.bounds[0],self.bounds[1])[0]
        return self.norm_const

        
    def set_support(self,bounds):
        # current bounds of pdf in the current element
        self.bounds = bounds

    def get_copy(self):
        copy = Normal(self.mean,self.var,self.bounds)
        # save original support
        copy.a = self.a
        copy.b = self.b
        # save original mean and variance
        copy.mean = self.mean
        copy.var  = self.var
        
        return copy
    
    
class TruncNormal():    
    def __init__(self, mean, sigma, bounds):
        self.local_mean = None
        self.local_var  = None
        self.local_median = None
        self.norm_const = None
        self.a = bounds[0]             # original bounds 
        self.b = bounds[1]             # original bounds 
        a_cp, b_cp = (bounds[0] - mean) / sigma, (bounds[1] - mean) / sigma
        self.cp_pdf = truncnorm(a_cp,b_cp,loc=mean,scale=sigma)
        
        self.norm_const = 1.0
            
        self.set_support(bounds)
        self.type = 'truncated_normal'
        self.mean = mean               # original mean
        self.var  = sigma              # original variance

        
    def pdf(self,x):
        return self.cp_pdf.pdf(x)
    
    def get_mean(self):
        if self.local_mean is None: 
            self.local_mean = self.cp_pdf.mean()
        return self.local_mean
    
    def get_variance(self):
        if self.local_var is None:
            self.local_var = self.cp_pdf.var()
        return self.local_var
    
    def get_median(self):
        if self.local_median is None: 
            self.local_median = self.cp_pdf.median()
        return self.local_median
    
    def get_support(self):
        return self.bounds
    
    def rvs(self,size=1):
        # sample from pdf
        return self.cp_pdf.rvs(size=size)
    
    def get_normalization_const(self):
        return self.norm_const
    
    def set_normalization_const(self,norm_const):
        self.norm_const = norm_const
        
    def set_support(self,bounds):
        # current bounds of pdf in the current element
        self.bounds = bounds

    def get_copy(self):
        copy = TruncNormal(self.mean,self.var,self.bounds)
        # save original support
        copy.a = self.a
        copy.b = self.b
        # save original mean and variance
        copy.mean = self.mean
        copy.var  = self.var
        
        copy.norm_const = self.norm_const
        
        return copy
    
    
class Beta():    
    ## ONLY IMPLEMENTED FOR mean=0, sigma=1
    def __init__(self,alpha,beta,mean, sigma, bounds):
    
        self.local_mean = None
        self.local_var  = None
        self.local_median = None
        self.norm_const = None
        self.a = bounds[0]             # original bounds 
        self.b = bounds[1]             # original bounds 
        self.alpha = alpha
        self.beta  = beta
        self.cp_pdf = beta_dist(self.alpha,self.beta,loc=mean,scale=sigma)
        
        self.norm_const = None
            
        self.set_support(bounds)
        self.type = 'beta'
        self.mean = mean               # original mean
        self.var  = sigma              # original variance

        
    def pdf(self,x):
        return self.cp_pdf.pdf(x)
    
    def get_mean(self):
        if self.local_mean is None: 
            def f(x):
                return x*self.pdf(x)/self.get_normalization_const()
            self.local_mean = quad(f,self.bounds[0],self.bounds[1])[0]            
        return self.local_mean
    
    def get_variance(self):
        if self.local_var is None:
            def f(x):
                return (x-self.get_mean())**2*self.pdf(x)/self.get_normalization_const()
            self.local_var = quad(f,self.bounds[0],self.bounds[1])[0]            
        return self.local_var
    
    def get_median(self):
        if self.local_median is None: 
            def f1(x):
                return self.pdf(x)/self.get_normalization_const()
            def f2(y):
                return quad(f1,self.bounds[0],y)[0] - 0.5
            self.local_median = fsolve(f2,(self.bounds[1]+self.bounds[0])/2)[0]            
        return self.local_median
    
    def get_support(self):
        return self.bounds
    
    def rvs(self,size=1):
        # sample from pdf
        return self.cp_pdf.rvs(size=size)
    
    def get_normalization_const(self):
        if not self.norm_const:
            def f(x):
                return self.pdf(x)
            self.norm_const = quad(f,self.bounds[0],self.bounds[1])[0]
        return self.norm_const

        
    def set_support(self,bounds):
        # current bounds of pdf in the current element
        self.bounds = bounds

    def get_copy(self):
        copy = Beta(self.alpha,self.beta,self.mean,self.var,self.bounds)
        # save original support
        copy.a = self.a
        copy.b = self.b
        # save original mean and variance
        copy.mean = self.mean
        copy.var  = self.var
        
        return copy