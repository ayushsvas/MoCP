#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:19:58 2020

@author: ayush
"""

#Evolution of eigenvalues with Lanczos steps 

import numpy as np 
import matplotlib.pyplot as plt


n=100
A=np.random.uniform(-1,1,(n,n))
A=(A+A.T)/2
v1=np.ones((n,1))
v1/=np.linalg.norm(v1)
basis=v1 

TriD=np.zeros((n,n))
eigstore=np.zeros(n)

w1_prime=np.matmul(A,v1)
a=np.matmul(w1_prime.T,v1)
w1=w1_prime-a*v1

TriD[0,0]=a
TriD[1,0]=0
TriD[0,1]=0

v_j_1=np.copy(v1)
w_j_1=w1

plt.plot(np.ones(np.int(np.size(np.linalg.eig(TriD)[0])))*0.1,np.linalg.eig(TriD)[0],'.k',markersize=3)
for j in range(1,n):
    b=np.linalg.norm(w_j_1)
    v_j=w_j_1/b
    w1_prime=np.matmul(A,v_j)
    a=np.matmul(w1_prime.T,v_j)
    w1=w1_prime-a*v_j-b*v_j_1
    v_j_1=np.copy(v_j)
    w_j_1=w1
    basis=np.append(basis,v_j,axis=1)
    TriD[j,j]=a
    TriD[j-1,j]=b
    TriD[j,j-1]=b
    eigvals = np.linalg.eig(TriD)[0]
    plt.plot(np.ones(np.int(np.size(eigvals)))*0.1*(j+1), eigvals,'.k',markersize=3)
    plt.xlabel("Lanczos Step")
    plt.ylabel("Eigenvalue Spectrum")
    
plt.plot(np.ones(np.int(np.size(np.linalg.eig(A)[0])))*(j+3)*0.1,np.linalg.eig(TriD)[0],'.y',markersize=10)
plt.savefig(" MinMaxEigvalConv.png",dpi=600,bbox_inches='tight')
OrthNormB=np.matmul(basis.T,basis)

OrthNormB=np.matmul(basis.T,basis)