#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:37:50 2019

@author: ayush
"""
import numpy as np
import matplotlib.pyplot as plt
from  numba import jit
import random
import time
tim=time.time()

#forward flux sampling 

KT=0.001
divs=200    
q1=np.linspace(-1,2,divs)
q2=np.linspace(-1,2,divs)
Q1,Q2=np.meshgrid(q1,q2)
Volt=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
Tp=100000
droppings=100#l
infaces=5
saveq1=np.zeros((droppings,infaces))
saveq2=np.zeros((droppings,infaces))
numberoftrialsatc=np.zeros(5)
#Potential funtion


@jit(nopython=True)
def pot(Q1,Q2):
    Volt=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
    return Volt

@jit(nopython=True)
def rare_event():
    c=0
    saveq1=np.zeros((droppings,infaces))
    saveq2=np.zeros((droppings,infaces))
    numberoftrialsatc=np.zeros(5)
    l=0
    count=0
    for c in range(infaces):
        if c==0:
            
            for i in range(droppings):
                
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
                        if -q2p-5*q1p-c+1>0:
                            print(-q2p-5*q1p+1)
                            saveq1[l,c]=q1p
                            saveq2[l,c]=q2p
                            l+=1
                            print(l)
                            break 
                                #Till here I started from (0,0), did 50 trials. Saved at c=0 interface, the coordinates for each successful trial.
        elif c>0 and c<4:
            numberoftrialsatc[c]=l
            l=0
            for d in range(len(saveq1[:,0])):
                q1p=saveq1[d,c-1]    #Simulation from c=0 starts, successul trails at c=0 are tested for c=1. If they pass, saving takes place at c=1
                q2p=saveq2[d,c-1]
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
                        if -q2p-5*q1p-c-1>0:
                            saveq1[l,c]=q1p
                            saveq2[l,c]=q2p
                            l+=1
                            break
        else:
            for dd in range(l):
                q1p=saveq1[dd,c]    #Simulation from c=0 starts, successul trails at c=0 are tested for c=1. If they pass, saving takes place at c=1
                q2p=saveq2[dd,c]
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
    return count
                
fflux=rare_event()
print(fflux)                           
        

#firstly defining iso-commitor surfaces 
