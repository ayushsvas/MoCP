#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:22:45 2019

@author: ayush
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:34:08 2019

@author: ayush
"""
import numpy as np
import matplotlib.pyplot as plt
from  numba import jit
import random
import time
tim=time.time()


KT=0.01
divs=200    
q1=np.linspace(-1,2,divs)
q2=np.linspace(-1,2,divs)
Q1,Q2=np.meshgrid(q1,q2)
Volt=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
check=Volt<5*KT
sinkA=np.where(check==True)
q1a=q1[sinkA[1][:]]
q2a=q2[sinkA[0][:]]
maskq1=q1a<0.5
maskq2=q2a<0.5
q1A=q1a[maskq1]
q2A=q2a[maskq2]
Tp=100000
@jit(nopython=True)
def pot(Q1,Q2):
    Volt=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
    return Volt
KT=0.001
@jit(nopython=True)
def rare_event():
    j=0
    flux=np.zeros(1000)
    for KT in np.linspace(0.001,1,1000):
        count=0
        for i in range(len(q1A)):
            q1p=0
            q2p=0
            
            pv=pot(q1p,q2p)
            for t in range(Tp):
                Q1=q1p+(random.random()-0.5)*2*0.3*np.sqrt(KT)
                Q2=q2p+(random.random()-0.5)*2*0.3*np.sqrt(KT)
                
                v=pot(Q1,Q2)
                rho=np.exp(-(v-pv)/KT)
                if rho>random.random():
                    q1p=Q1
                    q2p=Q2
                    pv=v
                    if q1p<-1 or q2p<-1:
                        q1p=2-round(q1p%1)
                        q2p=2-round(q2p%1)
                    if q1p>2 or q2p>2:
                        q1p=-1+round(q1p%1)
                        q2p=-1+round(q2p%1)
                    if v-pot(1,1)<5*0.01 and q1p>0.5 and q2p>0.5:
                        count+=1
                        break
        
        flux[j]=count
        j+=1
    return flux
rflux=rare_event()
print(rflux)
print(time.time()-tim)
plt.figure()
plt.plot(np.linspace(1,1000,1000)/1000,rflux/len(q1A))
plt.grid()
plt.xlabel("KT")
plt.ylabel("Reactive Flux")
plt.title("Flux Over-Population Method")
#plt.savefig("Plots/Flux Over-Population.png",dpi=600,bbox_inches="tight")
                     
        
        
            



    

