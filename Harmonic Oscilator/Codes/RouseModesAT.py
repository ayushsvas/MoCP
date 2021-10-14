#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:06:23 2019

@author: ayush
"""

#Anderson Thermostat on Rouse Model and calculation of rouse modes 
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import time
N=64
h=0.001
T=100
dt=1000
steps=int(T/h)
sigma=np.sqrt(1/63/3)
mu=0
tim=time.time()

dr=np.ones((N,3,steps))


for q in range(N-1):
    dr[q+1,0,0]=dr[q,0,0]+np.random.normal(mu,sigma)
    dr[q+1,1,0]=dr[q,1,0]+np.random.normal(mu,sigma)
    dr[q+1,2,0]=dr[q,2,0]+np.random.normal(mu,sigma)
dp=np.random.normal(mu,1,(N,3,steps))
dp[:,:,1:-1]=1
dp[:,:,-1]=1


F=np.ones((N,3,steps))
F[0,:,0]=3/2*(N-1)*(2*dr[1,:,0]-2*dr[0,:,0])
F[-1,:,0]=3/2*(N-1)*(2*dr[-2,:,0]-2*dr[-1,:,0])
H=np.zeros(steps)

for t in range(steps-1):
    dr[0,:,t+1]=dr[0,:,t]+dp[0,:,t]*h+F[0,:,t]*h**2/2
    dr[-1,:,t+1]=dr[-1,:,t]+dp[-1,:,t]*h+F[-1,:,t]*h**2/2
    for i in range(1,N-1):
        F[i,:,t]=3/2*(N-1)*(2*dr[i+1,:,t]+2*dr[i-1,:,t]-4*dr[i,:,t])
        dr[i,:,t+1]=dr[i,:,t]+dp[i,:,t]*h+F[i,:,t]*h**2/2
        
    F[0,:,t+1]=3/2*(N-1)*(2*dr[1,:,t+1]-2*dr[0,:,t+1])
    dp[0,:,t+1]=dp[0,:,t]+h/2*F[0,:,t]+h/2*F[0,:,t+1]
    F[-1,:,t+1]=3/2*(N-1)*(2*dr[-2,:,t+1]-2*dr[-1,:,t+1])
    dp[-1,:,t+1]=dp[-1,:,t]+h/2*F[-1,:,t]+h/2*F[-1,:,t+1]
    if t%dt==0:
        dp[:,:,t+1]=np.random.normal(mu,1,(N,3))
    else:
        for j in range(1,N-1):
            F[j,:,t+1]=3/2*(N-1)*(2*dr[j+1,:,t+1]+2*dr[j-1,:,t+1]-4*dr[j,:,t+1])
            dp[j,:,t+1]=dp[j,:,t]+h/2*F[j,:,t]+h/2*F[j,:,t+1]
            
for s in range(steps):
    for r in range(N-1):
        H[s]+=3/2*(N-1)*np.sum((dr[r+1,:,s]-dr[r,:,s])**2)
    H[s]=H[s]+np.sum(dp[:,:,s]**2/2)
        
k=0
#for k in range(steps):
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(dr[:,0,k],dr[:,1,k],dr[:,2,k],'ob-')
#ax.scatter(dr[:,0,k],dr[:,1,k],dr[:,2,k])
plt.show()
plt.pause

plt.figure(1)
plt.plot(np.linspace(0,steps-1,steps)*h,H)
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Step Size = {}".format(h))
plt.grid()
plt.show()    
print("Time Elapsed = {}".format(time.time()-tim))

Y=np.ones(steps)
p=1
#Rouse Modes
for p in range(1,4):
    i=np.linspace(1,N,N)
    i=np.reshape(i,(N,1))
    X=np.ones((3,steps))
    for y in range(steps):
        X[0,y]=np.sum(dr[:,0,y]*np.cos(np.pi*p/N*(i-1/2)).T)/N
        X[1,y]=np.sum(dr[:,1,y]*np.cos(np.pi*p/N*(i-1/2)).T)/N
        X[2,y]=np.sum(dr[:,2,y]*np.cos(np.pi*p/N*(i-1/2)).T)/N
        Y[y]=np.sum(np.sqrt(dr[:,0,y]**2+dr[:,1,y]**2+dr[:,2,y]**2)*np.cos(np.pi*p/N*(i-1/2)).T)/N
    plt.figure(2)
    plt.plot(np.linspace(0,steps-1,steps)*h,X[0,:],label='p={}'.format(p))
    plt.xlabel('Time')
    plt.ylabel("Time Eveloution of Rouse Modes X_p along x")
    plt.title("Step Size = {}, p = {},{},{}".format(h,1,2,3))
    plt.grid
    plt.legend()
    plt.figure(3)
    plt.plot(np.linspace(0,steps-1,steps)*h,X[1,:],label='p={}'.format(p))
    plt.xlabel('Time')
    plt.ylabel("Time Eveloution of Rouse Modes X_p along y")
    plt.title("Step Size = {}, p = {},{},{}".format(h,1,2,3))
    plt.grid
    plt.legend()
    plt.figure(4)
    plt.plot(np.linspace(0,steps-1,steps)*h,X[2,:],label='p={}'.format(p))
    plt.xlabel('Time')
    plt.ylabel("Time Eveloution of Rouse Modes X_p along z")
    plt.title("Step Size = {}, p = {},{},{}".format(h,1,2,3))
    plt.grid
    plt.legend()
    plt.figure(5)
    plt.plot(np.linspace(0,steps-1,steps)*h,Y,label='p={}'.format(p))
    plt.xlabel('Time')
    plt.ylabel("Time Eveloution of Rouse Modes X_p")
    plt.title("Step Size = {}, p = {},{},{}".format(h,1,2,3))
    plt.grid
    plt.legend()



psqavg=np.ones(steps)

for e in range(steps):
    psqavg[e]=np.sum(dp[:,0,e]**2)/N
m=str(np.mean(psqavg))
plt.figure(6)
plt.plot(np.linspace(0,steps-1,steps)*h,psqavg)
plt.xlabel("Time")
plt.ylabel("Temperature from variance of velocity")
plt.title("Step Size = {}, Average Temperarture of the plot = {}",format(h,m))
plt.grid()

