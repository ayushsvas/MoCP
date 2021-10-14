import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

#Diffusion Quantum Monte Carlo for the Harmonic Oscillator

t=time.time()

tmax = 1000
dtau = 0.01

N = np.zeros(tmax)
N[0] = 1000000  #initial no. of replicas
x0 = 0
x_init = x0*np.ones(int(N[0]))
E = np.zeros(tmax)
E[0] = 0.5*x0**2
d=1
sig = np.sqrt(dtau)
for i in range(tmax-1):
    x_fin = x_init + sig*np.random.normal(0,1,int(N[i])) #advance replicas
#    W = np.exp(-(((x_fin**2-d**2)**2)/(8*d**2)-E[i])*dtau) #calculate weights
    W = np.exp(-0.5*(x_fin**2-E[i])*dtau)
    Wd = W + np.random.rand(int(N[i])) #calculate multiplicities
    m = Wd.astype(int)
    m = np.minimum(m,3)
    x_fin = np.repeat(x_fin,m) # birth/death according to multiplicity
    N[i+1] = np.size(x_fin)
    if i==0:
        E[i+1] = np.mean(0.5*x_fin**2)
    E[i+1] = E[i] + 1*(1-N[i+1]/N[i])/dtau #New estimate of ground state energy
    x_init = x_fin




## E_T vs size of step
# dtau = np.linspace(0,1,500)
# mean = np.zeros(100)
# var = np.zeros(100)

# for i in range(1,500):
#     N, E, x_fin = dqmc(tmax,dtau[i])
#     mean[i] = np.mean(E[1000::])
#     var[i] = np.var(E[1000::])

# plt.plot(dtau[1::],mean[1::])
# plt.xlabel('Imaginary timestep Δτ')
# plt.ylabel('Groundstate energy E_T')
# plt.savefig('gsevsdt',dpi=600,bbox_inches='tight')
# plt.show()
# plt.plot(dtau[1::],var[1::])
# plt.show()


hist, edge = np.histogram(x_fin,bins=500)
x_real = np.linspace(edge[0],edge[500],500)
psi_real = np.exp(-x_real**2/2)/np.pi**0.25
norm = np.sqrt(np.sum(hist**2)*(edge[1]-edge[0]))
plt.plot(x_real,hist/norm,linewidth=2)
plt.plot(x_real,psi_real,linewidth=2)
plt.xlabel('x')
plt.ylabel('Psi(x)')
plt.legend(['DQMC','Analytical'])
plt.title("Ground state wave function")
#plt.savefig('gwf',dpi=600,bbox_inches='tight')
plt.show()

tau = np.linspace(0,tmax*dtau,tmax)
plt.plot(tau,E) #Ground state energy
#plt.savefig('gse0.1',dpi=600,bbox_inches='tight')
plt.show()

# print(np.mean(E[100::]))
# print(np.var(E[100::]))

print('Time Elapsed =',time.time()-t,'seconds')