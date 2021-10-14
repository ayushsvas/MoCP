import numpy as np
import random as rnd
import matplotlib.pyplot as plt

Jc = 0.44068679350977147 #critial value
J = Jc #exchange energy
H = 0 #external magnetic field
N = 16 #no. of spins on one side
tmax = 10000*N**2 #no. of monte carlo steps
Rn = 1 #no. of different monte carlo realizations

s = np.ones((N,N,tmax), dtype = int) #spin state cold start
#s[:,:,0] = np.sign(np.random.rand(N,N)-0.5) #hot start random spin state
M = np.zeros((Rn,tmax)) #magnetization array
E = np.zeros((Rn,tmax))

def Energy(state):
    s0 = 0
    for j in range(N):
        for i in range(N):
            s0 = s0 + state[i,j]*state[(i-1)%N,j] + state[i,j]*state[(i+1)%N,j] + state[i,j]*state[i,(j-1)%N] + state[i,j]*state[i,(j+1)%N]
    return -J*s0/2

def diffE(state,i,j):
    nnsum = state[(i-1)%N,j]+state[(i+1)%N,j]+state[i,(j-1)%N]+state[i,(j+1)%N]
    return 2*J*state[i,j]*nnsum

def autocorr(m,t):
    norm = 1/(tmax-t)
    sum1=0;sum2=0;sum3=0;
    for i in range(tmax-t):
        sum1 += m[i]*m[i+t]
        sum2 += m[i]
        sum3 += m[i+t]
    out = norm*sum1-sum2*sum3*norm**2
    return out

E[:,0] = Energy(s[:,:,0])
M[:,0] = np.sum(s[:,:,0])

for y in range(Rn):
    for x in range(tmax-1):
        rnum = int((N**2)*rnd.random())
        row = rnum//N
        col = rnum%N
        delE = diffE(s[:,:,x],row,col)
        delM = -2*s[row,col,x]
        r = rnd.random()
        R = np.exp(-delE)
        if delE < 0 or R >= r:
            s[row,col,x] = -s[row,col,x]
            s[:,:,x+1] = s[:,:,x]
            E[y,x+1] = E[y,x] + delE
            M[y,x+1] = M[y,x] + delM
        else:
            s[:,:,x+1] = s[:,:,x]
            E[y,x+1] = E[y,x]
            M[y,x+1] = M[y,x]

M = (1/N**2)*M
ravgm = (1/Rn)*np.sum(M, axis=0)
tavgm = (1/tmax)*np.sum(M, axis=1)
rtavgm = (1/tmax)*np.sum(ravgm)
eqavg = (1/np.size(ravgm[100:tmax-1]))
his, xed, yed = np.histogram2d(E, M, bins = 257)

plt.figure(1)
plt.plot(np.linspace(0,tmax/N**2,tmax),ravgm)
plt.ylabel('Magnetization per spin')
plt.xlabel('Monte Carlo Steps')
plt.show()

#plt.figure(2)
#plt.plot((1/it)*np.sum(Energy, axis = 0))
#plt.show()

#plt.hist(m, 1)
#plt.imshow(s[:,:,x+1], cmap = 'binary', aspect = 'auto')
#plt.show()
#plt.pause(0.01)
#plt.draw
    
    