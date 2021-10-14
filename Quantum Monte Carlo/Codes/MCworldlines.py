#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:57:55 2020

@author: ayush
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit 
import time 
tim=time.time()

#initital configuration 
n=100
T=100000
t_x=np.zeros((2,n))
t_x[0,:]=np.random.uniform(-1,1,n)
t_x[1,:]=np.linspace(0,T,n)
corr=np.zeros(T)
t_x_prev1=t_x
dt=0.01
t_x[0,-1]=t_x[0,0]
#picking a time slice at random and displacing Rj


for t in range(T):
    
    pick=int(np.random.random()*n)-2
    
    x_old=t_x[0,pick]
    x_pick=t_x[0,pick]+np.random.uniform(-1,1)*np.sqrt(dt)
    
    
    dS=(x_old-x_pick)*(t_x[0,pick+1]+t_x[0,pick-1]-x_old-x_pick)/dt+dt*(-x_old**2+x_pick**2)/2
                  #calculate action at new point 
    
    rho=np.exp(-dS)
    if rho>np.random.random():
        t_x[0,pick]=x_pick
        t_x[0,-1]=t_x[0,0]
        t_x[0,0]=t_x[0,-1]
        
#   
#    if t>100:
#        if t==101:
#            t_x_prev=t_x
    corr[t]=np.correlate(t_x[0,:]-np.mean(t_x[0,:]),t_x_prev1[0,:]-np.mean(t_x_prev1[0,:]))
            
        

print(time.time()-tim)
plt.figure(1)
plt.plot(t_x[0],t_x[1],'-ob')
plt.figure(2)
plt.plot(np.linspace(0,np.size(corr),np.size(corr))/n,corr)
plt.figure(3)
m,c=np.polyfit(np.linspace(1,np.size(corr[0:4500]),np.size(corr[0:4500]))/n,np.log(corr[0:4500]),1)
x=np.linspace(1,np.size(corr[0:4500]),np.size(corr[0:4500]))/n
y=m*x+c
plt.plot(x,corr[0:4500]);plt.plot(x,np.exp(y));plt.plot(np.ones(40)/(-m),range(40),'--')
plt.xlabel("MC step")
plt.ylabel("Autocorrelation")
plt.title("Required MCS steps for independent configurations = {}".format(-1/m))
plt.legend(['Computed Autocorrelation','Linear Curve Fitted','Convergence line'])