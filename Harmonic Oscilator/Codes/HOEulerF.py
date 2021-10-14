

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:35:03 2019

@author: ayush
"""
import numpy as np
import matplotlib.pyplot as plt 
import math
#simple harmonic oscillator 
T=1000  #Total run time 
h=0.001  #Single time step
N=int(T/h)  #No. of steps 
q=np.ones(N)
p=np.ones(N)  #Phase space variables and their storage 
energy=np.ones(N)
q0=1
p0=1 
q[0]=q0
p[0]=p0   #starting point  



for t in range(N-1):
    q[t+1]=q[t]+h*p[t]
    p[t+1]=p[t]-h*q[t]
    energy[t+1]=q[t+1]**2/2+p[t+1]**2/2
txt = "Energy plot of Euler Forward Algorithm" 
plt.figure(1)
plt.plot(q,p)
plt.title("Step size = {}, Time = {}".format(h,T))
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.grid()
plt.axis('square')
plt.savefig("Plots/EulerF0_{}.png".format(h),dpi=600,bbox_inches='tight')
plt.show 
plt.figure(2)
plt.plot(np.linspace(0,N-1,N)*h,energy)
plt.title("Step size = {}".format(h))
plt.xlabel("Time")
plt.ylabel("Energy")
plt.grid()
plt.figtext(0.5,-0.01,txt,ha='center')
plt.savefig("Plots/EnergyEulerF_{}.png".format(h),dpi=600,bbox_inches='tight')
plt.show
    
