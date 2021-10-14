#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:31:39 2019

@author: ayush
"""

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  
from numba import jit
import random 
import time 

tim=time.time()

#Monte carlo jumps 

@jit(nopython=True)
def pot(Q1,Q2):
    V=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
    return V
    
    
@jit(nopython=True)
def rare_event():            
    KT=0.1
    divs=100
    q1=np.linspace(-1,2,divs)
    q2=np.linspace(-1,2,divs)
    comp=np.ones((divs,divs))
    Tg=100000
    for i in range(divs):
        for j in range(divs):
            countm=0
            countp=0
            iq1=q1[i]
            iq2=q2[j]
            q1p=q1[i]
            q2p=q2[j]
            pv=pot(q1p,q2p)
            
            for t in range(Tg):
                Q1=q1p+(random.random()-0.5)*2*0.3*np.sqrt(KT)
                Q2=q2p+(random.random()-0.5)*2*0.3*np.sqrt(KT)
                
                v=pot(Q1,Q2)
                rho=np.exp(-(v-pv)/KT)
                if rho>random.random():
                    q1p=Q1
                    q2p=Q2
                    pv=v
                    potty=pot(Q1,Q2)
                    if q1p<-1 or q2p<-1:
                        q1p=2-round(q1p%1)
                        q2p=2-round(q2p%1)
                    if q1p>2 or q2p>2:
                        q1p=-1+round(q1p%1)
                        q2p=-1+round(q2p%1)
                    if potty-pot(0,0)<5*KT and q1p<0.5 and q2p<0.5:
                        countm+=1
                        q1p=iq1
                        q2p=iq2
                        
                    if potty-pot(1,1)<5*KT and q1p>0.5 and q2p>0.5:
                        countp+=1
                        q1p=iq1
                        q2p=iq2                  
                        
        comp[i,j]=countp/(countm+countp)
    return comp
KT=0.1     
                                    
cp=rare_event()
plt.figure(1)
plt.imshow(np.flipud(cp),aspect='equal',cmap='magma',extent=[-1,2,-1,2])
plt.colorbar()
plt.xlabel("q1")
plt.ylabel("q2")
plt.title("KT = {}".format(KT))
plt.savefig("Plots/CommitorProb{}.png".format(KT),dpi=600, bbox_inches="tight")
print(time.time()-tim)    




    