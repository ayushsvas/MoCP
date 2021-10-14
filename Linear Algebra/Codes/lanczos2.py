import numpy as np
import matplotlib.pyplot as plt
#from numba import jit
import time
from pathlib import Path

#@jit(nopython=True)
def lancz(A,n,m):
    a = np.zeros(m)
    b = np.zeros(m-1)
    V = np.zeros((n,m))
    
    V[:,0] = np.random.uniform(-1,1,n)
    V[:,0] = V[:,0]/np.linalg.norm(V[:,0])
    w1d = A@V[:,0]
    a[0] = w1d.T@V[:,0]
    w1 = w1d - a[0]*V[:,0]
    
    T = np.zeros((m,m))
    T[0,0] = a[0]
    
    for j in range(m-1):
        b[j] = np.linalg.norm(w1)
        if b[j] != 0:
            V[:,j+1] = w1/b[j]
        else:
            V[:,j+1] = np.random.uniform(-1,1,n)
            for i in range(j+1):
                V[:,j+1] = V[:,j+1] - np.dot(V[:,j+1],V[:,i])*V[:,i]
            V[:,j+1] = V[:,j+1]/np.linalg.norm(V[:,j+1])
        wjd = A@V[:,j+1]
        a[j+1] = wjd.T@V[:,j+1]
        wj = wjd-a[j+1]*V[:,j+1]-b[j]*V[:,j]
        w1 = wj
        T[j+1,j] = b[j]
        T[j,j+1] = b[j]
        T[j+1,j+1] = a[j+1]
    return T,V

tt = time.time()
res = 50
avg = 8
n = np.linspace(1,500,res).astype(int)
cpu = np.zeros(res)

s=0
for j in range(avg):
    for i in range(res):
        t = time.time()
        B = np.random.uniform(-1,1,(n[i],n[i]))
        A = (B+B.T)/2
        T,V = lancz(A,n[i],n[i])
        eigvalT, eigvectT = np.linalg.eig(T)
        cpu[i] = time.time()-t
    s=s+cpu/avg
    plt.pause(2)
    
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

b,c,d = np.polyfit(n,cpu,deg=2)
plt.plot(n,b*n**2+c*n+d,'r',linewidth=2);plt.plot(n,cpu,'.k',markersize=5)
plt.xlabel('Size of matrix (n)')
plt.ylabel('CPU Time (in secs)')
plt.legend(['Quadratic fit','Observed'])
#plt.savefig('cpu',dpi=600,bbox_inches='tight')
plt.show()

print('Time Elapsed =',time.time()-tt,'seconds')

