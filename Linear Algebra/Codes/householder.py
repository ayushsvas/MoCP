import numpy as np
#import matplotlib.pyplot as plt
#from numba import jit
#import time
from tabulate import tabulate
#@jit(nopython=True)
def tridiag(A,n):
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
        # for i in range(n):
        #     if i <= k-1:
        #         w[i] = 0
        #     elif i == k+1-1:
        #         w[i] = (A[k+1-1,k-1]-a)/(2*r)
        #     else:
        #         w[i] = A[i,k-1]/(2*r)
        P = np.identity(n)-2*w*w.T
        A = P @ A @ P
        Pr = Pr @ P
        Pl = P @ Pl
    A[np.abs(A) < eps] = 0
    return A,Pl,Pr

n = 8
B = np.random.uniform(-1,1,(n,n)) #generator matrix
A = (B+B.T)/2 #symmetrized
T,Pl,Pr = tridiag(A,n)
A=np.array([[-0.08881783, -0.34807589,  0.3776538 , -0.44642712,  0.1906032 ,
        -0.11704179, -0.24663897,  0.27616006],
       [-0.34807589, -0.72109811,  0.32308118,  0.69879731,  0.06454008,
        -0.67712922, -0.02172997,  0.44746676],
       [ 0.3776538 ,  0.32308118, -0.59075846, -0.33315317, -0.76282866,
         0.01161367,  0.40853859, -0.34992508],
       [-0.44642712,  0.69879731, -0.33315317, -0.55354723,  0.52337426,
         0.57542695, -0.54132491,  0.19689057],
       [ 0.1906032 ,  0.06454008, -0.76282866,  0.52337426, -0.16390296,
        -0.03325957, -0.03782796,  0.38667693],
       [-0.11704179, -0.67712922,  0.01161367,  0.57542695, -0.03325957,
        -0.17926821, -0.28940202,  0.42826085],
       [-0.24663897, -0.02172997,  0.40853859, -0.54132491, -0.03782796,
        -0.28940202,  0.660168  , -0.1848833 ],
       [ 0.27616006,  0.44746676, -0.34992508,  0.19689057,  0.38667693,
         0.42826085, -0.1848833 ,  0.472477  ]])

# N = 1000; rep = 1
# cpu = np.zeros(N-3)
# for n in range(3,N,100):
#     s = 0
#     for i in range(rep):
#         B = np.random.uniform(-1,1,(n,n)) #generator matrix
#         A = (B+B.T)/2 #symmetrized
#         t = time.time()
#         T,Pl,Pr = tridiag(A)
#         Ql = np.linalg.inv(Pl)
#         Qr = np.linalg.inv(Pr)
#         cpu[n-3] = s + (time.time()-t)/rep

# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 12}

# plt.rc('font', **font)

# z=cpu[np.nonzero(cpu)]
# x = np.linspace(0,999,np.size(z))
# a,b,c,d = np.polyfit(x,z,deg=3)
# plt.plot(x,a*x**3+b*x**2+c*x+d,'r',linewidth=2);plt.plot(x,z,'.k',markersize=5)
# plt.xlabel('Size of matrix (n)')
# plt.ylabel('CPU Time (in secs)')
# plt.legend(['Cubic fit','Observed'])
# plt.savefig('cpu',dpi=600,bbox_inches='tight')
# plt.show()

Ql = np.linalg.inv(Pl)
Qr = np.linalg.inv(Pr)
A_d = Ql@T@Qr
T_d = Pl@A@Pr
eigA = np.sort(np.linalg.eig(A)[0])
eigvA = np.sort(np.linalg.eig(A)[1])eig
eigT = np.sort(np.linalg.eig(T_d)[0])


