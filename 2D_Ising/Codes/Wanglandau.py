#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:38:20 2019

@author: ayush
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import math

sump=0
N=16
steps=10*N**2
KB=1
T=1
state =1*np.ones((N,N,steps))
Jc=np.log(1+np.sqrt(2))/2
magstore=np.ones(steps-1)
energystore=np.ones(steps-1)
energydstore=np.ones(steps-1)
energystore1=np.ones(steps-1)
J=0.4
n=0
energyst=0
deltaE=0
Knot=Jc/(KB*T)
K=J/(KB*T)
#Defining energy funtion and then giving this initial state its intitial energy
def energy(s):
    term1=0
    term2=0
    
    for i in range(N):
        for j in range(N):
            if j!=N-1:
                term1+=s[i][j]*s[i][j+1]+s[i][j]*s[i][j-1]
        term1+=s[i][N-1]*s[i][N-2]+s[i][N-1]*s[i][0]
                
       

    for j in range(N):
        for i in range(N):
            if i!=N-1:
                term2+=s[i][j]*s[i+1][j]+s[i][j]*s[i-1][j]
        term2+= s[N-1][j]*s[N-2][j]+s[N-1][j]*s[0][j]
    return(-Jc*(term1+term2)/2);

def diffE(test,r1,r2):
    horiz=0
    verti=0
    if r2!=N-1:
        horiz+=test[r1,r2]*2*(test[r1][r2-1]+test[r1][r2+1])
    else:
        horiz+=test[r1,r2]*2*(test[r1][r2-1]+test[r1][0])
    if r1!=N-1:
        verti+=test[r1,r2]*2*(test[r1-1][r2]+test[r1+1][r2])
    else:
        verti+=test[r1,r2]*2*(test[r1-1][r2]+test[0][r2])
          
    return (-Jc*(horiz+verti))
        

def magnetization(test):
    magperspin=0
    
    magperspin+=np.sum(test)
    return magperspin
    
flag=0
m=0
t=0 
f=1.2
u=0
ii=1
#making storage for book keeping of magnetisation and weights
wmagstore=np.zeros(N**2+1)
wts=np.ones(N**2+1)#initialising all weights to 1
#keeping track here
def wmagnetization(s,wmagstore,wts):
    global m,u,f,flag,ii
    m=int(np.sum(s))
    wmagstore[m]+=ii
    wts[m]=np.log(f)+wts[m]
    u=u+1
    hmax=np.max(wmagstore)
    hmin=np.min(wmagstore)
    if  (hmax-hmin)/(hmax+hmin)<0.2:
        wmagstore=np.zeros(steps)
        f=np.sqrt(f)
        
        if f==1+10e-6:
            flag=1
    
while flag==0:
    r1=int(N*random.random())
    r2=int(N*random.random())
    test=state[:,:,t]
    test[r1][r2]*=-1
    wmagnetization(test,wmagstore,wts)
    
    



















    
iniE=energy(state[:,:,0])

#flipping a spin in the lattice and calculating probablity density for the new state obtained

for i in range(steps-1):
    m=0
    test=np.copy(state[:,:,i])
    magnetization(test)
    #generating a random number
    r1=int(N*random.random())
    r2=int(N*random.random())
    
    #PrevE=energy(state[:,:,i])
    #energystore[i]=PrevE
    
    test[r1,r2]*=-1
    #NewE=energy(test)
    #deltaE=NewE-PrevE
    mi=int(magnetization(state[:,:,i]))
    m=int(magnetization(test))
    p1=np.exp(-(diffE(test,r1,r2))/(KB*T*(-wts[m]+wts[mi])))   
    #print(diffE(test,r1,r2),deltaE)
    #p=np.exp(-(deltaE)/(KB*T))
   
    if p1>=random.random():
        state[:,:,i+1]=test
        
        energydstore[i]=diffE(test,r1,r2)
        
        #sump+=NewE-PrevE
        #PrevE=NewE       
    else:
        state[:,:,i+1]=state[:,:,i]
        energydstore[i]=0
    

#plt.imshow(state[:,:,-1])


