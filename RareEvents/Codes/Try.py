#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 01:31:39 2019

@author: ayush
"""
#Reactive flux. To calculate this firstly remove my concept of 50 trials to each point.
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  
from numba import jit
import random 
import time 

tim=time.time()


KT=0.01
divs=100    
q1=np.linspace(-1,2,divs)
q2=np.linspace(-1,2,divs)
Q1,Q2=np.meshgrid(q1,q2)

#sinkA=np.where(Volt<5*KT)
#q1a=q1[sinkA[0][:]]
#q2a=q2[sinkA[1][:]]
#maskq1=q1a<0.5
#maskq2=q2a<0.5
#q1A=q1a[maskq1]
#q2A=q2a[maskq2]
#comp=np.ones((divs,divs))

Tg=100000

#Monte carlo jumps 

@jit(nopython=True)
def pot(Q1,Q2):
    V=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
    return V
    
Volt=pot(Q1,Q2)

    
@jit(nopython=True)
def rare_event():    
    countp=np.ones(100)
    
    q1p=1.5
    q2p=0.2
    for kt in range(1,101):
        count=0
        
        sinkA=np.where(Volt<5*0.01)
        q1a=q1[sinkA[0][:]]
        q2a=q2[sinkA[1][:]]
        maskq1=q1a<0.5
        maskq2=q2a<0.5
        q1A=q1a[maskq1]
        q2A=q2a[maskq2]
        
        #print(len(q1A))
        for i in range(len(q1A)):
            
            
          
            pv=pot(q1p,q2p)
            for t in range(Tg):
                test1=q1p+(random.random()-0.5)*2*0.3*np.sqrt(0.01)
                test2=q2p+(random.random()-0.5)*2*0.3*np.sqrt(0.01)
                
                v=pot(test1,test2)
                rho=np.exp(-(v-pv)/kt/100)
                if rho>random.random():
                    q1p=test1
                    q2p=test2
                    pv=v
                    potty=pot(test1,test2)
                    if q1p<-1 or q2p<-1:
                        q1p=2-round(q1p%1)
                        q2p=2-round(q2p%1)
                    if q1p>2 or q2p>2:
                        q1p=-1+round(q1p%1)
                        q2p=-1+round(q2p%1)
                    if potty-pot(1,1)<0.05 and q1p**2+q2p**2>(q1p-1)**2+(q2p-1)**2:
                        count+=1
                        break
            testq1=q1A[i]
            testq2=q2A[i]
            
            rho=(-pot(testq1,testq2)/kt/100)
            if rho>random.random():
                q1p=q1A[i]
                q2p=q2A[i]
                pv=pot(q1p,q2p)
                
        countp[kt-1]=count
    return countp
KT=0.1     
                                    
cp=rare_event()
#plt.figure(1)
#plt.imshow(np.flipud(cp),aspect='equal',cmap='magma',extent=[-1,2,-1,2])
#plt.colorbar()
#plt.xlabel("q1")
#plt.ylabel("q2")
#plt.title("KT = {}".format(KT))
#plt.savefig("Plots/CommitorProb{}.png".format(KT),dpi=600, bbox_inches="tight")
print(time.time()-tim)    
print(cp)
plt.plot(np.linspace(1,100,100)/100,cp)