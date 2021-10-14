#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:24:10 2019

@author: ayush
"""
import numpy as np
import matplotlib.pyplot as plt




def pot(Q1,Q2):
    Volt=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
    return Volt
KT=0.01
divs=200    
q1=np.linspace(-1,2,divs)
q2=np.linspace(-1,2,divs)
Q1,Q2=np.meshgrid(q1,q2)
Volt=(1-np.exp(-(Q1-0.3*Q2)**2-8*(0.3*Q1+Q2)**2))*(1-np.exp(-(Q1-1)**2-8*(Q2-1)**2))
check=Volt<5*KT
sinkA=np.where(check==True)
q1a=q1[sinkA[1][:]]
q2a=q2[sinkA[0][:]]
maskq1=q1a<0.5 #this conditions fails at 0.1
maskq2=q2a<0.5
q1A=q1a[maskq1]
q2A=q2a[maskq2]

plt.figure(1)
plt.contourf(q1,q2,Volt)
plt.scatter(q1A,q2A)




