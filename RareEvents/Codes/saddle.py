import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

@jit(nopython=True)
def F(x,y):
    Fx = -(-(1-np.exp(-8*(0.3*x+y)**2-(x-0.3*y)**2))*(2-2*x)*np.exp(-(x-1)**2-8*(y-1)**2)-(1-np.exp(-(x-1)**2-8*(y-1)**2))*(-3.44*x-4.2*y)*np.exp(-8*(0.3*x+y)**2-(x-0.3*y)**2))
    Fy = -(-(1-np.exp(-8*(0.3*x+y)**2-(x-0.3*y)**2))*(16-16*y)*np.exp(-(x-1)**2-8*(y-1)**2)-(1-np.exp(-(x-1)**2-8*(y-1)**2))*(-4.2*x-16.18*y)*np.exp(-8*(0.3*x+y)**2-(x-0.3*y)**2))
    return np.array([Fx,Fy])

@jit(nopython=True)
def V(q1,q2):
    return (1-np.exp(-(q1-0.3*q2)**2-8*(0.3*q1+q2)**2))*(1-np.exp(-(q1-1)**2-8*(q2-1)**2))

@jit(nopython=True)
def prop(x,y,tmax):
    traj = np.zeros((tmax+1,2))
    tot = np.zeros(tmax)
    delR = 0.01 #delta R for dimer
    dt = 1e-1 #timestep
    traj[0,0] = x; traj[0,1] = y
#    dth = 1e-7
    for i in range(tmax):
        for theta in np.linspace(0,np.pi,10**3):
            x1, y1 = x + delR*np.cos(theta), y + delR*np.sin(theta);
            x2, y2 = x - delR*np.cos(theta), y - delR*np.sin(theta);
            F1 = F(x1,y1)
            F2 = F(x2,y2)
            tau1 = ((x1-x)*F1[1]-(y1-y)*F1[0])/delR; tau2 = ((x2-x)*F2[1]-(y2-y)*F2[0])/delR
            Fi = tau1 + tau2
            if abs(Fi)<1e-3:
                break
        tot[i] = tau1 + tau2
        Fr = (F1+F2)/2 #avg force
        n1 = np.array([x1-x,y1-y])/delR; #n2 = np.array([x2-x,y2-y])/delR; #unit vectors
        Fp = np.dot(Fr,n1)*n1 #parallel force
        Fdag = Fr-2*Fp;  #modified
        x = x + Fdag[0]*dt
        y = y + Fdag[1]*dt
        traj[i+1,0] = x; traj[i+1,1] = y
    return traj, tot

t=time.time()
tmax = 10**3
#x = 2-3*np.random.random(); y = 2-3*np.random.random(); #start from random point
x,y = 1,0.5
traj, tot = prop(x,y,tmax)
xval = np.linspace(-1,2,10**3)
X,Y=np.meshgrid(xval,xval)
#fx, fy = F(X,Y)
Z = V(X,Y)
plt.plot(traj[:,0],traj[:,1],'k', linewidth=1.5)
#plt.plot(traj1[:,0],traj1[:,1],'k', linewidth=1.5)
#plt.plot(traj2[:,0],traj2[:,1],'k', linewidth=1.5)
plt.imshow(np.flipud(Z), extent=[-1,2,-1,2],cmap='magma')
plt.colorbar()
#plt.grid()
plt.plot(traj[tmax,0],traj[tmax,1],'ok')
plt.savefig('dimer',dpi=600,bbox_inches='tight')
plt.show()
#plt.plot(traj1[:,0],traj1[:,1],'r',traj2[:,0],traj2[:,1],'b')
#plt.show()
print('Time Elapsed =',time.time()-t,'seconds')