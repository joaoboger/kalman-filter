import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import scipy.constants as ct

q = 1 #elementary charge
m = 0.14 #GeV/c^2 = GeV once c=1 here
pT = 0.9 #GeV
realv = (pT/np.sqrt(np.power(m,2)+np.power(pT,2))) #velocity adimensional
gamma = 1/np.sqrt(1-np.power(realv,2)) #gamma adimensional
K = (1.113*1e-8)*q/(gamma*m) # unit conversion to obtain K in C/kg
B = 2*1e7 #Magnetic field in Tesla

print(K*B)

def EofM(icond,t):
    #Variables initialization
    r=icond[0]
    phi=icond[1]
    vr=icond[2]
    vphi=icond[3]
    
    #Equations of motion
    newR = vr
    newPhi = vphi
    newVR = (K*B+vphi)*r*vphi
    newVPhi = -(2*vphi+K*B)*vr/r

    return [newR,newPhi,newVR,newVPhi]

dt=1/(realv) #time steps
nsteps=10000 #number of steps


fig, axs = plt.subplots(2,2)

for i in range(0,10):
    t=np.linspace(i,i+dt,nsteps) #time points
    icond=[i+0.1,np.pi/4,1,0] #Initial values of r,phi,vr,vphi
    x=odeint(EofM,icond,t) #RK4

    axs[0,0].plot(t,x[:,0])
    axs[0,0].set_title(r"$r$ coordinate")
    axs[0,1].plot(t,x[:,1])
    axs[0,1].set_title(r"$\phi$ coordinate")
    axs[1,0].plot(t,x[:,2])
    axs[1,0].set_title(r"$v_r$")
    axs[1,1].plot(t,x[:,3])
    axs[1,1].set_title(r"$v_{\phi}$")
    fig.suptitle(r"$r_{0}=3.0, \phi_{0}=\frac{\pi}{4}, v_{r,0}=1, v_{\phi,0}=0$")

    plt.figure(20)
    plt.plot(x[:,0]*np.cos(x[:,1]),x[:,0]*np.sin(x[:,1]), marker='.', lineStyle='None')
    print(x[:,1])


for i in range(1,11): #Plots circles in MPL
    circle = plt.Circle((0, 0), i*1 , color='r', fill=False)
    plt.gca().add_patch(circle)

plt.title(r"$q=$ %s,$m=$ %s,$p_{T}=$ %s,$B=$ %s,$v_{real}$= %.4s" % (q,m,pT,B,realv))

plt.show()