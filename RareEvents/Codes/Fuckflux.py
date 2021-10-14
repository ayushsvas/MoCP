#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:58:45 2019

@author: ayush
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time 
import random
tim=time.time()
KT=0.003
divs=200    
q1=np.linspace(-1,2,divs)
q2=np.linspace(-1,2,divs)
Q1,Q2=np.meshgrid(q1,q2)
Tt=100000

c=0
droppings=100
infaces=69
numberofsamplesperc=np.ones(infaces)


#initial conditions 



#potential energy calculation 
@jit(nopython=True)
def pot(Q1,Q2):
    Volt=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
    return Volt

#Applying Monte Carlo and Forward Flux Algorithm 
@jit(nopython=True)
def rare_event(count,numberofsamplesperc):
    saveq1=np.zeros((droppings,infaces))
    saveq2=np.zeros((droppings,infaces))
    c=1
    T=np.zeros(infaces+1)
    for c in range(1,70) :
        
        if c==1:
            q1p=0
            q2p=0
            pv=pot(q1p,q2p)
            flag=0
        else:
            l=0
            q1p=saveq1[l,c-2]
            q2p=saveq2[l,c-2]
            
            pv=pot(q1p,q2p)
            s=0
        for t in range(Tt):
            testq1=q1p+(random.random()-0.5)*2*0.3*np.sqrt(KT)
            testq2=q2p+(random.random()-0.5)*2*0.3*np.sqrt(KT)
            v=pot(testq1,testq2)
            rho=np.exp(-(v-pv)/KT)
            if rho>random.random():
                q1p=testq1
                q2p=testq2     
                pv=v
                if q1p<-1 or q2p<-1:
                    q1p=2-round(q1p%1)
                    q2p=2-round(q2p%1)
                if q1p>2 or q2p>2:
                    q1p=-1+round(q1p%1)
                    q2p=-1+round(q2p%1)            
                if 6*q1p+q2p-c/100>0:
                    saveq1[s,c-1]=q1p
                    saveq2[s,c-1]=q2p
                    s+=1
                    if c==1:             
                        flag+=1
                        if flag==droppings-1:
                            break
                        q1p=0
                        q2p=0       
                    else:
                        l+=1
                        q1p=saveq1[l,c-2]
                        q2p=saveq2[l,c-2]
                        if l==numberofsamplesperc[c-2]:
                            break
        T[c-1]=t
        numberofsamplesperc[c-1]=s      
    for L in range(l):
        q1p=saveq1[L,-1]
        q2p=saveq2[L,-1]
        pv=pot(q1p,q2p)
        for t in range(Tt):
            testq1=q1p+random.random()*0.3*np.sqrt(KT)
            testq2=q2p+random.random()*0.3*np.sqrt(KT)
            v=pot(testq1,testq2)
            rho=np.exp(-(v-pv)/KT)
            if rho>random.random():
                q1p=testq1
                q2p=testq2
                pv=v
                if q1p<-1 or q2p<-1:
                    q1p=2-round(q1p%1)
                    q2p=2-round(q2p%1)
                if q1p>2 or q2p>2:
                    q1p=-1+round(q1p%1)
                    q2p=-1+round(q2p%1)
                if v-pot(1,1)<5*0.01 and q1p>0.5 and q2p>0.5:
                    count+=1
                    T[-1]=T[-1]+t
                    break
    
                
    return (count,numberofsamplesperc,saveq1,saveq2,T)
fflux, arry, qp1,qp2,tim_e=rare_event(c,numberofsamplesperc)
print(fflux)
for j in range(infaces):    
    plt.scatter(qp1[:,j],qp2[:,j])

        
                        
                    
                        
                    

        
    
    


