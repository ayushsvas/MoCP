#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:08:34 2019

@author: ayush
"""

import numpy as np
import matplotlib.pyplot as plt 

#custom dynamics
q0=1  #Starting point and momentum 
p0=1
h=0.001 #timestep
T=10  #total simulation time 
N=int(T/h)  #No. of timesteps 
q=np.ones(N)  #storage for position values for timesteps 
p=np.ones(N)   #storage for momentum values per timestep
energy=np.ones(N)

q[0]=q0  #Initialising starting point to storage of postion and momentum 
p[0]=p0


for t in range(N-1):
    q[t+1]=q[t]*(1-h**2)+h*p[t] 
    p[t+1]=p[t]-q[t]*h
    energy[t+1]=q[t+1]**2/2+p[t+1]**2/2
txt = "Energy plot of Symplectic Euler" 
plt.figure(1)
plt.plot(q,p)
plt.title("Step Size = {}, Time = {}".format(h,T))
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.savefig("Plots/CustomDyn_{}.png".format(h),dpi=600,bbox_inches='tight')
plt.grid()
plt.axis('square')
plt.show()
plt.figure(2)
plt.plot(np.linspace(1,N-2,N-1)*h,energy[1::])
plt.grid()
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Step Size = {}".format(h))
plt.figtext(0.5,-0.05,txt,ha='center')
plt.savefig("Plots/energyCustomDyn_{}.png".format(h),dpi=600,bbox_inches='tight')
plt.show()
