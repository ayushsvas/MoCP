import numpy as np
import matplotlib.pyplot as plt
#from numba import jit
import time
from tabulate import tabulate
#@jit(nopython=True)
def tridiag(A,n,clk):
    tim=time.time()
    Pl = np.identity(n)
    Pr = np.identity(n)
    eps = 1e-15
    # w = np.zeros((n,1))
    for k in range(1,n-1):
        a = -np.sign(A[k+1-1,k-1])*np.sqrt(np.sum(A[k+1-1::,k-1]**2))
        r = np.sqrt(0.5*a**2-0.5*a*A[k+1-1,k-1])
        c1 = np.reshape((A[k+1-1,k-1]-a)/(2*r),(1,1))
        c2 = np.reshape(A[k+2-1::,k-1]/(2*r),(np.size(A[k+2-1::,k-1]),1))
        w = np.concatenate((np.zeros((k,1)),c1,c2))
        
        P = np.identity(n)-2*w*w.T
        A = P @ A @ P
        Pr = Pr @ P
        Pl = P @ Pl
    A[np.abs(A) < eps] = 0
    clk[int(n/8)-1]=time.time()-tim
    return A,Pl,Pr

clk=np.zeros(99)
for n in range(1,100):
    n = n*8
    B = np.random.uniform(-1,1,(n,n)) #generator matrix
    A = (B+B.T)/2 #symmetrized
    
    T,Pl,Pr = tridiag(A,n,clk)




plt.plot(np.linspace(8,800,99),clk)
Ql = np.linalg.inv(Pl)
Qr = np.linalg.inv(Pr)
A_d = Ql@T@Qr
T_d = Pl@A@Pr
eigA = np.sort(np.linalg.eig(A)[0])
eigT = np.sort(np.linalg.eig(T_d)[0])


