#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 22:50:22 2020

@author: ayush
"""

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


tim=time.time()

N=500
T=1000
dt=0.002

sumV=0

Eval=np.ones(T)


Estore=[0.42701857, 0.43588855, 0.4510801 , 0.46340596, 0.4670668 ,
       0.47429093, 0.45959453, 0.48042813, 0.47926381, 0.47491098,
       0.46861007, 0.48556042, 0.4903719 , 0.47402549, 0.47299791,
       0.48139038, 0.47574718, 0.49051135, 0.48835483, 0.47966014,
       0.47171493, 0.4860828 , 0.47147494, 0.4765902 , 0.48638796,
       0.47903697, 0.4742429 , 0.49324128, 0.47863385, 0.48789949,
       0.48702467, 0.49004011, 0.48329419, 0.48613716, 0.48324399,
       0.48251115, 0.47642142, 0.48018819, 0.48949491, 0.48068832,
       0.48347981, 0.47969066, 0.48856967, 0.47980859, 0.4757919 ,
       0.4792608 , 0.4767209 , 0.47971857, 0.48639036, 0.48178395,
       0.47415055, 0.4721148 , 0.48349413, 0.4826896 , 0.47908762,
       0.48301932, 0.47310892, 0.47889913, 0.48002591, 0.47494446,
       0.48176325, 0.48036105, 0.47834933, 0.48893346, 0.46933787,
       0.479727  , 0.48444892, 0.47317691, 0.47838058, 0.47818223,
       0.47282698, 0.47618452, 0.47336112, 0.47953195, 0.48191725,
       0.47726024, 0.47994711, 0.47053149, 0.47094486, 0.47750355,
       0.48210863, 0.47139035, 0.47651603, 0.47742953, 0.4662584 ,
       0.49226872, 0.46729887, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ]
for dt in range(10,11):
    N=500
    reps=np.zeros((T,N))
    E_r=np.sum(reps[0,:])**2/2/N
    Eval[0]=E_r
    Nnot=N
    flag=1
    for t in range(T-1):
        n=0
        flag=1
        reps[t+1,:]=reps[t,:]+np.random.normal(0,1,N)*np.sqrt(dt/1000)
        sumV=0
        while flag==1:
            
            
            V=0.5*reps[t+1,n]**2
            
            W=np.exp(-(V-E_r)*dt/1000)
            
           
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
        
        E_rnew=E_r+(1-Nnot/N)
        E_r=E_rnew
        Eval[t+1]=E_r
        N=Nnot
#    Estore[dt-5]=np.mean(Eval)
print(time.time()-tim)

#plt.plot(np.linspace(5,100,95)/1000,Estore)

plt.figure(1)
plt.plot(np.linspace(0,T-1,T)*dt/1000,Eval)

#plt.plot(np.linspace(5,99,94)/1000,Est)
#plt.title("Ground State Energy Estimate for different time slices")
#plt.xlabel("Imaginary Time Slice (iℏΔτ)")
#
#plt.ylabel("Ground state energy")
#plt.grid()

