#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:22:42 2019

@author: ayush
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 10:31:44 2019

@author: ayush
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import stats

sump=0
N=16
steps=300*N**2
KB=1
T=1
state =1*np.ones((N,N,steps))
Jc=(np.log(1+np.sqrt(2)))/2
magstore=np.ones(steps-1)
energystore=np.ones(steps-1)
energydstore=np.ones(steps-1)
mpswt=np.ones(steps-1)
normz=np.ones(steps-1)
y=0
J=0.4
n=0
energyst=0
deltaE=0

deltaJ=Jc-J
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
        

def magnetization(test,prevE):
    magperspin=0
    global n
    magperspin+=np.sum(test)/(N**2)
    magstore[n]=magperspin
    givewt(prevE,magperspin)
    n=n+1
    
iniE=energy(state[:,:,0])


def givewt(prevE,magperspin1):
    global y
    mpswt[y]=magperspin1*np.exp(deltaJ*prevE/Jc)
    normz[y]=np.exp(deltaJ*prevE/Jc)
    y=y+1
#flipping a spin in the lattice and calculating probablity density for the new state obtained

for i in range(steps-1):
    test=np.copy(state[:,:,i])
    PrevE=energy(state[:,:,i])
    magnetization(test,PrevE)
    #generating a random number
    r1=int(N*random.random())
    r2=int(N*random.random())
    
    
    energystore[i]=PrevE
    
    test[r1,r2]*=-1
    #NewE=energy(test)
    #deltaE=NewE-PrevE
    p1=np.exp(-(diffE(test,r1,r2))/(KB*T))   
   
    p=np.exp(-(deltaE)/(KB*T))
   
    if p1>=random.random():
        state[:,:,i+1]=test
        
        energydstore[i]=diffE(test,r1,r2)
        
        #sump+=NewE-PrevE
        #PrevE=NewE      
    else:
        state[:,:,i+1]=state[:,:,i]
        energydstore[i]=0
    
#print(np.sum(state[:,:,0]))
#plt.imshow(state[:,:,-1])
#plt.show()
plt.figure(1)
plt.plot(np.linspace(0,steps/N**2,steps-1),magstore)
#plt.xlabel("Steps")
#plt.ylabel("Mag/spin")
#plt.figure(2)
#plt.xlabel("Steps")
#plt.ylabel("Energy")
#plt.plot(np.linspace(0,steps/N**2,steps-1),energystore)
#print(magstore)

print("Net Energy Change = {}, Energy difference between initial and final state = {}".format(np.sum(energydstore),energy(state[:,:,-1])-iniE))
#print(magstore,energystore)
#x=np.copy(energystore)
#y=np.copy(magstore)
#c=np.unique(x)
#c1=np.unique(y)
#print(len(c),len(c1))
#H1=plt.hist2d(x,y,bins=(len(c),len(c1)))
#deltaK=Knot-K
##A=H[0]*(np.exp(-deltaK*(H[1]))).T
##probd=H*(np.exp(-deltaK*(H)))/(np.sum(A))
##print(probd)
#plt.figure(2)
##plt.plot(H[2],probd)
#H=plt.hist(magstore,len(c1))
#H[0]*np.exp(-deltaK*)
#plt.figure(2)
#plt.hist(mpswt/sum(normz),bins=len(np.unique(mpswt)))
#plt.hist(magstore,bins=len(np.unique(magstore)))


