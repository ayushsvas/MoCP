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

t = time.time()

n = 2000
B = np.random.uniform(-1,1,(n,n))
A = (B+B.T)/2
eigvalA, eigvectA = np.linalg.eig(A)
odA = eigvalA.argsort()[::-1]
eigvalA = eigvalA[odA]
eigvectA = eigvectA[:,odA]
mmax = 100
egvext = np.zeros((mmax,2))
eigvalm = []

for i in range(mmax):
    m = i+1
    T,V = lancz(A,n,m)
    eigvalT, eigvectT = np.linalg.eig(T)
    eigvalm.append(eigvalT)
    # odT = eigvalT.argsort()[::-1]
    # eigvalT = eigvalT[odT]
    # eigvectT = eigvectT[:,odT]
    egvext[i,0] = np.min(eigvalT)
    egvext[i,1] = np.max(eigvalT)
    
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

plt.plot(egvext[:,0]-np.min(eigvalA)) #estimate of smallest eigval
plt.plot(egvext[:,1]-np.max(eigvalA)) #estimate of smallest eigval
plt.plot(np.linspace(0,mmax,10),np.zeros(10),'--k',linewidth=0.3)
plt.xlabel('Lanzcos Steps')
plt.ylabel('Deviation')
plt.legend(['Minimum','Maximum'])
#plt.savefig('n='+str(n)+'/lancz_estimate',dpi=600,bbox_inches='tight')
plt.show()

for i in range(mmax):
    plt.plot(eigvalm[i],np.zeros(i+1)+i,'.',markersize = 3)
plt.plot(eigvalA,np.zeros(n)+i+2,'-',markersize = 1)
plt.xlabel('Spectrum')
plt.ylabel('Lanczos Steps')
#plt.savefig('n='+str(n)+'/egvstime',dpi=600,bbox_inches='tight')
plt.show()

np.save('n='+str(n)+'/A',A)
np.save('n='+str(n)+'/eigvalA',eigvalA)
np.save('n='+str(n)+'/egvext',egvext)

print('Time Elapsed =',time.time()-t,'seconds')

