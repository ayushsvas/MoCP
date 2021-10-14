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


#q2=q2[:,np.newaxis]
#q1,q2=np.meshgrid(q1,q2)
#q1p=np.zeros(N**2)
#q2p=np.zeros(N**2)
#V=np.ones((N**2,N**2)) 
#V=(1-np.exp(-(q1-0.3*q2)**2-0.8*(0.3*q1+q2)**2))*(1-np.exp(-(q1-1)**2-8*(q2-1)**2))
#
#plt.figure(1)
#plt.contour(V,50,cmap='magma')
#plt.axis('square')

#Monte carlo jumps 

@jit(nopython=True)
def pot(Q1,Q2):
    V=(1-np.exp(-(Q1-0.3*Q2)**2-0.8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
    return V
    
    
@jit(nopython=True)
def rare_event():
    j=0
    N=50              
    KT=0.01
    q1=np.linspace(-1,2,100)
    q2=np.linspace(-1,2,100)
    comp=np.ones((100,100))
    for i in range(100):
        for j in range(100):
#            countm=0
            countp=0
        
            for p in range(N):
                q1p=q1[i]
                q2p=q2[j]
                pv=pot(q1p,q2p)
                flag=0
                
                while flag==0:
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
#                            countm+=1
                            flag=1
                        if potty-pot(1,1)<5*KT and q1p>0.5 and q2p>0.5:
                            countp+=1
                            #print(countp)
                            flag=1
                            
            comp[i,j]=countp/(N)
            
            
    return comp
                        
                        
cp=rare_event()

plt.imshow(np.flipud(cp))
plt.colorbar()

    




    