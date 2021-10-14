#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:38:23 2019

@author: ayush
"""
import numpy as np
import matplotlib.pyplot as plt
#RG-4th Harmonic Oscillator 


h=0.1
T=1000
N=int(T/h)
q0=0
p0=1
q=np.ones(N)
p=np.ones(N)
q[0]=q0
p[0]=p0
energy=np.ones(N)

for t in range(N-1):
    qk1=p[t]
    pk1=-q[t]
    qk2=p[t]-q[t]*h/2
    pk2=-q[t]-p[t]*h/2
    qk3=p[t]-q[t]*(h/2)-p[t]*h**2/4
    pk3=-q[t]-p[t]*h/2+q[t]*h**2/4
    qk4=p[t]-q[t]*(h)-p[t]*h**2/2+q[t]*h**3/4
    pk4=-q[t]-p[t]*h+q[t]*h**2/2+p[t]*h**3/4
    q[t+1]=q[t]+h/6*(qk1+2*qk2+2*qk3+qk4)
    p[t+1]=p[t]+h/6*(pk1+2*pk2+2*pk3+pk4)
    energy[t+1]=q[t+1]**2/2+p[t+1]**2/2

txt="Energy plot for RK4"
plt.figure(1)
plt.plot(q,p)
plt.title("Step size = {}, Time = {}".format(h,T))
plt.xlabel("Position")
plt.ylabel("Momentum")
plt.grid()
plt.axis('square')
plt.xlim([-0.0005,0.0005])
plt.ylim([0.9988,1.00])
plt.savefig("Plots/vstimeRG4HO_{}.png".format(h),dpi=600,bbox_inches='tight')
plt.show()

plt.figure(2)
plt.plot(np.linspace(1,N-2,N-1)*h,energy[1::])
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Steps Size = {}".format(h))
plt.grid()
plt.figtext(0.5,-0.01,txt,ha='center')
plt.savefig("Plots/vstimeenergyRG4HO_{}.png".format(h),dpi=600,bbox_inches='tight')
plt.show()

