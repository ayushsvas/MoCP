#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:08:16 2019

@author: ayush
"""

#random walk for {r_i}
import numpy as np
import matplotlib.pyplot as plt 
import random 
from mpl_toolkits.mplot3d import Axes3D

random.seed()
N=64
mu, sigma=0, np.sqrt(1/3*63)
dx=np.ones(N)
dy=np.ones(N)
dz=np.ones(N)
dr=np.zeros(N-1)
#Draw random samples from Gaussian Distribution 

for i in range(N-1):
    
    dx[i+1]=dx[i]+np.random.normal(mu,sigma)
    dy[i+1]=dy[i]+np.random.normal(mu,sigma)
    dz[i+1]=dz[i]+np.random.normal(mu,sigma)
    dr[i]=np.sqrt(dx[i+1]**2+dy[i+1]**2+dz[i+1]**2)
    



fig = plt.figure()
ax = Axes3D(fig)


ax.plot(dx, dy, dz,'ob-')
plt.show()
