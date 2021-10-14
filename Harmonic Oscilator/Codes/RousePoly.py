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
h=0.001
T=1
steps=int(T/h)

N=64
mu, sigma=0, np.sqrt(1/63)
dx=np.ones((N,steps))
dy=np.ones((N,steps))
dz=np.ones((N,steps))
dr=np.zeros(N-1)

#Draw random samples from Gaussian Distribution and forming the initial configuration of positions

dr=np.random.normal(mu,sigma,N)
theta=(np.random.random(N)-0.5)*3.14/2
phi=np.random.random(N)*3.14*2
dx=np.ones((N,steps))
dy=np.ones((N,steps))
dz=np.ones((N,steps))
dx[:,0]=dr*np.cos(theta)*np.cos(phi)
dy[:,0]=dr*np.cos(theta)*np.sin(phi)
dz[:,0]=dr*np.sin(theta)



    
px=np.ones((N,steps))
px[:,0]=np.random.normal(0,1,N) #initial momenta for particles sampled from gaussian distribution

py=np.ones((N,steps))
py[:,0]=np.random.normal(0,1,N) 

pz=np.ones((N,steps))
pz[:,0]=np.random.normal(0,1,N) 


Fx=np.ones((N,steps))
Fy=np.ones((N,steps))
Fz=np.ones((N,steps))

H=np.ones(steps)
#now applying velocity verlet
for t in range(steps-1):
    i=0
    for i in range(N):
        if i==0:
            Fx[i,t]=3/2*(N-1)*(2*dx[i+1,t]-2*dx[i,t])
            Fy[i,t]=3/2*(N-1)*(2*dy[i+1,t]-2*dy[i,t])
            Fz[i,t]=3/2*(N-1)*(2*dz[i+1,t]-2*dz[i,t])
            dx[i,t+1]=dx[i,t]+px[i,t]*h+Fx[i,t]*h**2/2
            dy[i,t+1]=dy[i,t]+py[i,t]*h+Fy[i,t]*h**2/2
            dz[i,t+1]=dz[i,t]+pz[i,t]*h+Fz[i,t]*h**2/2
            Fx[i,t+1]=3/2*(N-1)*(2*dx[i+1,t+1]-2*dx[i,t+1])
            Fy[i,t+1]=3/2*(N-1)*(2*dy[i+1,t+1]-2*dy[i,t+1])
            Fz[i,t+1]=3/2*(N-1)*(2*dz[i+1,t+1]-2*dz[i,t+1])
        elif i==N-1:
            Fx[i,t]=3/2*(N-1)*(2*dx[i-1,t]-2*dx[i,t])
            Fy[i,t]=3/2*(N-1)*(2*dy[i-1,t]-2*dy[i,t])
            Fz[i,t]=3/2*(N-1)*(2*dz[i-1,t]-2*dz[i,t])
            dx[i,t+1]=dx[i,t]+px[i,t]*h+Fx[i,t]*h**2/2
            dy[i,t+1]=dy[i,t]+py[i,t]*h+Fy[i,t]*h**2/2
            dz[i,t+1]=dz[i,t]+pz[i,t]*h+Fz[i,t]*h**2/2
            Fx[i,t+1]=3/2*(N-1)*(2*dx[i-1,t+1]-2*dx[i,t+1])
            Fy[i,t+1]=3/2*(N-1)*(2*dy[i-1,t+1]-2*dy[i,t+1])
            Fz[i,t+1]=3/2*(N-1)*(2*dz[i-1,t+1]-2*dz[i,t+1])
        else:
            
            Fx[i,t]=3/2*(N-1)*(2*dx[i-1,t]+2*dx[i+1,t]-4*dx[i,t])
            Fy[i,t]=3/2*(N-1)*(2*dy[i-1,t]+2*dy[i+1,t]-4*dy[i,t])
            Fz[i,t]=3/2*(N-1)*(2*dz[i-1,t]+2*dz[i+1,t]-4*dz[i,t])
            dx[i,t+1]=dx[i,t]+px[i,t]*h+Fx[i,t]*h**2/2
            dy[i,t+1]=dy[i,t]+py[i,t]*h+Fy[i,t]*h**2/2
            dz[i,t+1]=dz[i,t]+pz[i,t]*h+Fz[i,t]*h**2/2
            Fx[i,t+1]=3/2*(N-1)*(2*dx[i-1,t+1]+2*dx[i+1,t+1]-4*dx[i,t+1])
            Fy[i,t+1]=3/2*(N-1)*(2*dy[i-1,t+1]+2*dy[i+1,t+1]-4*dy[i,t+1])
            Fz[i,t+1]=3/2*(N-1)*(2*dz[i-1,t+1]+2*dz[i+1,t+1]-4*dz[i,t+1])
    for j in range(N):
        px[j,t+1]=px[j,t]+h/2*Fx[j,t]+h/2*Fx[j,t+1]
        py[j,t+1]=py[j,t]+h/2*Fy[j,t]+h/2*Fy[j,t+1]
        pz[j,t+1]=pz[j,t]+h/2*Fz[j,t]+h/2*Fz[j,t+1]
            
for ti in range(steps):
    for k in range(N-1):
        H[ti]=3/2*(N-1)*np.sum(((dx[k+1,ti]-dx[k,ti])**2+(dy[k+1,ti]-dy[k,ti])**2+(dz[k+1,ti]-dz[k,ti])**2))
    H[ti]=H[ti]+3/2*(N-1)*np.sum(px[:,ti]**2/2+py[:,ti]**2/2+pz[:,ti]**2/2)
        

#for k in range(steps):
#  
#    fig = plt.figure()
#    ax = Axes3D(fig)
#    ax.plot(dx[:,k], dy[:,k], dz[:,k],'ob-')
#    plt.show()
#    plt.pause
#plt.figure(1)
plt.plot(range(steps),H)
#plt.figure(2)
#plt.plot(range(steps),dx[7,:])
#plt.plot(dx[7,:],px[7,:])    