import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import jit
import time

zeroethdimension=time.time()

@jit(nopython=True)
def V(q1,q2):
    return (1-np.exp(-(q1-0.3*q2)**2-8*(0.3*q1+q2)**2))*(1-np.exp(-(q1-1)**2-8*(q2-1)**2))

@jit(nopython=True)
def F(x,y):
    Fx = -(-(1-np.exp(-8*(0.3*x+y)**2-(x-0.3*y)**2))*(2-2*x)*np.exp(-(x-1)**2-8*(y-1)**2)-(1-np.exp(-(x-1)**2-8*(y-1)**2))*(-3.44*x-4.2*y)*np.exp(-8*(0.3*x+y)**2-(x-0.3*y)**2))
    Fy = -(-(1-np.exp(-8*(0.3*x+y)**2-(x-0.3*y)**2))*(16-16*y)*np.exp(-(x-1)**2-8*(y-1)**2)-(1-np.exp(-(x-1)**2-8*(y-1)**2))*(-4.2*x-16.18*y)*np.exp(-8*(0.3*x+y)**2-(x-0.3*y)**2))
    return np.array([Fx,Fy])

tmax = 10**3
xval = np.linspace(-1,2,10**3)
X,Y=np.meshgrid(xval,xval)
pot = V(X,Y)
N = 1000
evol = np.zeros((N,2,tmax))
#@jit(nopython=True)
def string(tmax):
    
    
    dt = 1e-2
    Z = np.array([np.linspace(0,1,N),np.linspace(0,1,N)]).T
    for t in range(tmax):
        #apply force on string
        for i in range(N):
            x=Z[i,0];y=Z[i,1]
            Z[i,:] = Z[i,:] + F(x,y)*dt
        #reparametrize
        Delta = np.sum((np.diff(Z,axis=0))**2, axis=1)
        epsilon = np.cumsum(Delta)
        epsilon /= epsilon[N-2]
        epsilon = np.append(0,epsilon)
        cs = CubicSpline(epsilon,Z)
        Z = cs(np.linspace(0,1,N))
        evol[:,:,t] = Z
    return Z

Z = string(tmax)


#@jit(nopython=True)
def loopy(evol):
    for t in range(0,tmax,5):
        plt.imshow(np.flipud(pot), extent=[-1,2,-1,2],cmap='magma')
        plt.colorbar()
        plt.plot(evol[:,0,t],evol[:,1,t],'y',linewidth=2)
        plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'w',linewidth=1)
        plt.show()

loopy(evol)




#plt.imshow(np.flipud(pot), extent=[-1,2,-1,2],cmap='magma')
#plt.colorbar()
#plt.plot(Z[:,0],Z[:,1],'y',linewidth=2)
#plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'w',linewidth=1)
#plt.plot(0.050733,0.563589,'ko')
#plt.savefig('MEP',dpi=600,bbox_inches='tight')
#plt.show()

print('Time Elapsed =',time.time()-zeroethdimension,'seconds')

