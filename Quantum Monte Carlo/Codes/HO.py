#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:58:59 2020

@author: ayush
"""

#ground eigenstate and energy eigenvalue of HO
import numpy as np
import matplotlib.pyplot as plt 
from numba import jit 
import time
from scipy.stats import norm  

tim=time.time()

N=10000
T=1000
dt=0.01
reps=np.zeros((T,N))
sumV=0
E_r=np.sum(reps[0,:])**2/2/N
Eval=np.ones(T)
Eval[0]=E_r
Nnot=N
flag=1


for t in range(T-1):
    n=0
    flag=1
    reps[t+1,:]=reps[t,:]+np.random.normal(0,1,N)*np.sqrt(dt)
    sumV=0
    while flag==1:
        
        
        V=0.5*reps[t+1,n]**2
        
        W=np.exp(-(V-E_r)*dt)
        
       
        m=np.minimum(np.int(W+np.random.uniform()),3)
       
        sumV+=V*m
        
        if m==0:
            reps=np.delete(reps,n,axis=1)
            
            Nnot-=1
            n=n-1
            
          
        if m==2:
            b=np.reshape(np.copy(reps[:,n]),(T,1))
            reps=np.append(reps,b,axis=1)
            Nnot+=1
      
            
            
    
        if m==3:
            b=np.reshape(np.copy(reps[:,n]),(T,1))
            reps=np.append(reps,b,axis=1)
            reps=np.append(reps,b,axis=1)
            Nnot+=2
           
            
        n+=1
        if n==Nnot-1:
            flag=0
    Nnot=np.size(reps[t+1,:])       
    E_r=sumV/Nnot
    
#    E_rnew=E_r+(1-Nnot/N)
#    E_r=E_rnew
    Eval[t+1]=E_r
    N=Nnot
print(time.time()-tim)
y,x=np.histogram(reps[-1,:])


plt.figure(1)
plt.plot(np.linspace(0,T-1,T),Eval)

plt.figure(2)
y=np.exp(-x**2/2)/np.sqrt(np.sqrt(np.pi))
plt.hist(aesehi2[-1,:],bins=250)
xmin,xmax=plt.xlim()
#plt.hist(reps[-1,:],bins=250)
y0,x0=np.histogram(aesehi2[-1,:],bins=250)

A=np.trapz(y0**2,dx=(xmax-xmin)/250)
reps1=np.copy(reps)

y1=np.sqrt(3.5)*norm.pdf(x,0.05,1)
plt.figure(2)
plt.plot(x,y1)
plt.plot(x0[1:],y0/np.sqrt(A))
plt.plot(x,y)
        
    
    
        