#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:28:41 2019

@author: ayush
"""
#velocity verlet 
import numpy as np
import matplotlib.pyplot as plt 


q0=1  #Starting point and momentum 
p0=1
h=0.5 #timestep
T=10000 #total simulation time 
N=int(T/h)  #No. of timesteps 
q=np.ones(N)  #storage for position values for timesteps 
p=np.ones(N)   #storage for momentum values per timestep
energy=np.ones(N)
Hs=np.ones(N)
q[0]=q0  #Initialising starting point to storage of postion and momentum 
p[0]=p0


for t in range(N-1):
    
    q[t+1]=q[t]+p[t]*h-q[t]*h**2/2    #version 2 
    p[t+1]=p[t]*(1-h**2/2)+q[t]*(h**3/4-h)
    energy[t+1]=q[t+1]**2/2+p[t+1]**2/2
    Hs[t+1]=(-q[t+1]+q[t+1]*h**2/12)**2/2+(p[t+1]+p[t+1]*h**2/6)**2/2

txt="Energy plot for Velocity-Verlet"
txts="Energy plot for Velocity-Verlet using Shadow Hamiltonian"
plt.figure(1)
plt.plot(q,p)
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.title("Step Size = {}, Time = {}".format(h,T))
plt.grid()
plt.axis('square')
plt.savefig("Plots/VerysmallstepsVel_Verlet_{}.png".format(h),dpi=600,bbox__inches='tight')
plt.show()

plt.figure(2)
plt.plot(np.linspace(1,N-2,N-1)*h,energy[1::])
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Step Size = {}".format(h))
plt.grid()
plt.figtext(0.5,-0.01,txt,ha='center')
plt.savefig("Plots/VerysmallstepsenergyVel_Verlet_{}.png".format(h),dpi=600,bbox__inches='tight')
plt.show()

plt.figure(3)
plt.plot(np.linspace(1,N-2,N-1)*h,Hs[1::])
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Step Size = {}".format(h))
plt.grid()
plt.figtext(0.5,-0.01,txts,ha='center')
plt.savefig("Plots/VerysmallstepsenergyVel_Verlet_{}.png".format(h),dpi=600,bbox__inches='tight')
plt.show()

