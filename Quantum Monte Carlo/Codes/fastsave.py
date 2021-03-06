#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:06:34 2020

@author: ayush
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:17:50 2020

@author: ayush
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

#Diffusion Quantum Monte Carlo for the Harmonic Oscillator

t=time.time()

tmax = 1000
dtau = 0.3



N = np.zeros(tmax)
N[0] = 1000000 #initial no. of replicas
x0 = 0
x_init = x0*np.ones(int(N[0]))
E = np.zeros(tmax)
E[0] = 0.5*x0**2
sig = np.sqrt(dtau)
d=4
for i in range(tmax-1):
    x_fin = x_init + sig*np.random.normal(0,1,int(N[i])) #advance replicas
    W = np.exp(-(((x_fin**2-d**2)**2)/(8*d**2)-E[i])*dtau) #calculate weights
    Wd = W + np.random.rand(int(N[i])) #calculate multiplicities
    m = Wd.astype(int)
    m = np.minimum(m,3)
    x_fin = np.repeat(x_fin,m) # birth/death according to multiplicity
    N[i+1] = np.size(x_fin)
    if i==0:
        E[i+1] = np.mean(((x_fin**2-d**2)**2)/(8*d**2))
    E[i+1] = E[i] + 1*(1-N[i+1]/N[i])/dtau #New estimate of ground state energy
    x_init = x_fin

    
    
    
    
    
#hist, edge = np.histogram(x_fin,bins=500)
#x_real = np.linspace(edge[0],edge[500],500)
#yd=((x_real**2-d**2)**2/(8*d**2))
#norm = np.sqrt(np.sum(hist**2)*(edge[1]-edge[0]))
#plt.figure(1)
#plt.plot(x_real,hist/norm,linewidth=1)
##plt.plot(x_real,yd,'--b',linewidth=1)
#plt.xlabel('x')
#plt.ylabel("Psi(x)")
#plt.legend(['d=0.1','d=2','d=4','d=6','d=8'])
#
#plt.savefig('Plots/gwfdp.png',dpi=600,bbox_inches='tight')
#
#
tau = np.linspace(0,tmax*dtau,tmax-1)
plt.figure(2)
plt.plot(tau,E[1:]) #Ground state energy
plt.xlabel("Imaginar time (iℏτ)")
plt.ylabel("Ground state energy")
#plt.legend(['d=0.1','d=2','d=4','d=6','d=8'])
#plt.savefig('Plots/gsdp.png',dpi=600,bbox_inches='tight')
    



print('Time Elapsed =',time.time()-t,'seconds')