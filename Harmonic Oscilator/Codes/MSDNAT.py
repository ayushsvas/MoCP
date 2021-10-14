#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:06:23 2019

@author: ayush
"""

#rouse model
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import time
N=64
h=0.001
T=100
steps=int(T/h)
sigma=np.sqrt(1/63/3)
mu=0
tim=time.time()

dr=np.ones((N,3,steps))


for q in range(N-1):
    dr[q+1,0,0]=dr[q,0,0]+np.random.normal(mu,sigma)
    dr[q+1,1,0]=dr[q,1,0]+np.random.normal(mu,sigma)
    dr[q+1,2,0]=dr[q,2,0]+np.random.normal(mu,sigma)
dp=np.random.normal(mu,1,(N,3,steps))
dp[:,:,1:-1]=1
dp[:,:,-1]=1


F=np.ones((N,3,steps))
F[0,:,0]=3/2*(N-1)*(2*dr[1,:,0]-2*dr[0,:,0])
F[-1,:,0]=3/2*(N-1)*(2*dr[-2,:,0]-2*dr[-1,:,0])
H=np.zeros(steps)

for t in range(steps-1):
    dr[0,:,t+1]=dr[0,:,t]+dp[0,:,t]*h+F[0,:,t]*h**2/2
    dr[-1,:,t+1]=dr[-1,:,t]+dp[-1,:,t]*h+F[-1,:,t]*h**2/2
    for i in range(1,N-1):
        F[i,:,t]=3/2*(N-1)*(2*dr[i+1,:,t]+2*dr[i-1,:,t]-4*dr[i,:,t])
        dr[i,:,t+1]=dr[i,:,t]+dp[i,:,t]*h+F[i,:,t]*h**2/2
        
    F[0,:,t+1]=3/2*(N-1)*(2*dr[1,:,t+1]-2*dr[0,:,t+1])
    dp[0,:,t+1]=dp[0,:,t]+h/2*F[0,:,t]+h/2*F[0,:,t+1]
    F[-1,:,t+1]=3/2*(N-1)*(2*dr[-2,:,t+1]-2*dr[-1,:,t+1])
    dp[-1,:,t+1]=dp[-1,:,t]+h/2*F[-1,:,t]+h/2*F[-1,:,t+1]
    for j in range(1,N-1):
        F[j,:,t+1]=3/2*(N-1)*(2*dr[j+1,:,t+1]+2*dr[j-1,:,t+1]-4*dr[j,:,t+1])
        dp[j,:,t+1]=dp[j,:,t]+h/2*F[j,:,t]+h/2*F[j,:,t+1]
        
for s in range(steps):
    for r in range(N-1):
        H[s]+=3/2*(N-1)*np.sum((dr[r+1,:,s]-dr[r,:,s])**2)
    H[s]=H[s]+np.sum(dp[:,:,s]**2/2)
        
k=0
#for k in range(steps):
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(dr[:,0,k],dr[:,1,k],dr[:,2,k],'ob-')
#ax.scatter(dr[:,0,k],dr[:,1,k],dr[:,2,k])
plt.show()
plt.pause

plt.figure(1)
plt.plot(np.linspace(0,steps-1,steps)*h,H)
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Step Size = {}".format(h))
plt.grid()
plt.show()    
print("Time Elapsed = {}".format(time.time()-tim))


##g3
#bs=100 #block size
#to=0#book keeping
#bc=int(steps/bs) #block count
#g3=np.ones((bc,bs))  #No. of blocks are the number of columns and their size(time length) are number of rows 
##We run for loop on blocks
#for k in range(bc):
#    for ts in range(bs): #Block length (Time length)
#        g3[k,ts]=(np.sum(dr[:,0,ts+to])/N-np.sum(dr[:,0,to])/N)**2/bs
#        if (ts+1)%bs==0:
#            to=to+bs
#plt.figure(2)
#plt.plot(np.linspace(0,bs-1,bs)*bc*h,np.sum(g3[:,:],axis=0))
#plt.xlabel("Time")
#plt.ylabel("g3(t)")
#plt.title("Step size = {}, Mean Square Displacement of COM".format(h))
#plt.grid()



##g1
#bs=100 #block size
#to=0#book keeping
#bc=int(steps/bs) #block count
#g1=np.ones((bc,bs))  #No. of blocks are the number of columns and their size(time length) are number of rows 
##We run for loop on blocks
#for k in range(bc):
#    for ts in range(bs): #Block length (Time length)
#
#        g1[k,ts]=np.sum((dr[:,0,ts+to]-dr[:,0,to])**2,axis=0)/bs/N
#        if (ts+1)%bs==0:
#            to=to+bs
#plt.plot(np.linspace(0,bs-1,bs)*bc*h,np.sum(g1[:,:],axis=0))
#plt.xlabel("Time")
#plt.ylabel("g2(t)")
#plt.title("Step size = {}, Mean Square Displacement of Segments relative to COM ".format(h))
#plt.grid()

#q2
bs=100 #block size
to=0#book keeping
bc=int(steps/bs) #block count
g2=np.ones((bc,bs))  #No. of blocks are the number of columns and their size(time length) are number of rows 
#We run for loop on blocks
for k in range(bc):
    for ts in range(bs): #Block length (Time length)

        g2[k,ts]=np.sum((dr[:,0,ts+to]-dr[:,0,to]-np.sum(dr[:,0,ts+to])/N+np.sum(dr[:,0,to])/N)**2,axis=0)/bs/N
        if (ts+1)%bs==0:
            to=to+bs
plt.plot(np.linspace(0,bs-1,bs)*bc*h,np.sum(g2[:,:],axis=0))
plt.xlabel("Time")
plt.ylabel("g2(t)")
plt.title("Step size = {}, Mean Square Displacement of Segments relative to COM ".format(h))
plt.grid()