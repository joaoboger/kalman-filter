import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

q=1
m = 0.14 #pions em GeV/c2
pT = 0.9 #GeV
realv = (pT/np.sqrt(np.power(m,2)+np.power(pT,2))) #velocity adimensional
gamma = 1/np.sqrt(1-np.power(realv,2)) #gamma adimensional
K = (6.072E-3/2)*q/(gamma*m)
B = 20 #Magnetic field in Tesla

def EofMP(icond,t): 
    r=icond[0]
    phi=icond[1]
    vr=icond[2]
    vphi=icond[3]

    newR = vr
    newPhi = vphi
    newVR = (K*B+vphi)*r*vphi
    newVPhi = -(2*vphi+K*B)*vr/r

    return [newR,newPhi,newVR,newVPhi]

def EofMN(icond,t):
    r=icond[0]
    phi=icond[1]
    vr=icond[2]
    vphi=icond[3]

    newR = vr
    newPhi = vphi
    newVR = (-K*B+vphi)*r*vphi
    newVPhi = -(2*vphi-K*B)*vr/r

    return [newR,newPhi,newVR,newVPhi]

    
### This function returns the updated position
### The input parameters are:
### the current position R, Phi
### the current velocity V_R, V_Phi
### the elapsed time interval dt
### newR, newPhi, newVR, newVPhi are the updated position and velocities

def updatePosition(r, phi, vr, vphi, dt, j, q): #Calculates the update position with function scipy.integrate.odeint (RK4)
    dt=dt*1.5
    t=np.linspace(0,dt,10000)
    if q==1:
        icond=[r,phi,vr,vphi]
        x=odeint(EofMP,icond,t)
    elif q==-1:
        icond=[r,phi,vr,vphi]
        x=odeint(EofMN,icond,t)
    bestn = -1

    for i in range(0,t.size):
        if x[i,0]<j-0.01:
            True
        else:
            bestn=i
            break

    return x[i,0],x[i,1],x[i,2],x[i,3]

def kalman2d(n,dt,p_v,q,Z,Charge):
    Q=Charge

    realPhi = Z[0] # Angle Phi used to generate tracks

    initialppx=1/np.sqrt(2) #In the first iteration I'm assuming complete ignorance of the track angle
    initialppy=1/np.sqrt(2)

    seed_r = 0.1 #Seed of our first iteration
    seed_phi = 1/np.sqrt(2)
    seed_vr = realv
    seed_vphi = 0

    track_x = np.array([]) #Arrays where all positions and errors will be stored to fit the track
    track_y = np.array([])
    track_px = np.array([])
    track_py = np.array([])

    measurement_x = np.array([])
    measurement_y = np.array([])
    measurement_px = np.array([])
    measurement_py = np.array([])

    prediction_x = np.array([])
    prediction_y = np.array([])
    prediction_px = np.array([])
    prediction_py = np.array([])

    for j in range(1,n+1):
        #Prediction step#
        pred_r, pred_phi, pred_vr, pred_vphi = updatePosition(seed_r, seed_phi, seed_vr, seed_vphi, dt, j, Q) #Prediction of positions x and y

        pred_x = pred_r*np.cos(pred_phi)
        pred_y = pred_r*np.sin(pred_phi)

        pred_px = initialppx #Prediction of the propagated error
        pred_py = initialppy
        #Measure step#
        me_x = Z[4*j-3]
        me_y = Z[4*j-2]
        me_px = Z[4*j-1]
        me_py = Z[4*j]

        #Update step#
        kgx = pred_px/(pred_px+me_px) #Kalman gain
        kgy = pred_py/(pred_py+me_py)
        kx = pred_x + kgx*(me_x-pred_x) #Kalman positions
        ky = pred_y + kgy*(me_y-pred_y)
        kpx = (1-kgx)*pred_px #Kalman uncertainties
        kpy = (1-kgy)*pred_py
        
        # Add the values to arrays
        track_x = np.append(track_x,kx)
        track_y = np.append(track_y,ky)
        track_px = np.append(track_px,kpx)
        track_py = np.append(track_py,kpy)

        measurement_x = np.append(measurement_x,me_x)
        measurement_y = np.append(measurement_y,me_y)
        measurement_px = np.append(measurement_px,me_px)
        measurement_py = np.append(measurement_py,me_py)

        prediction_x = np.append(prediction_x,pred_x)
        prediction_y = np.append(prediction_y,pred_y)
        prediction_px = np.append(prediction_px,pred_px)
        prediction_py = np.append(prediction_py,pred_py)

        #Update next iteration input values        
        x=kx
        y=ky
        initialppx=kpx
        initialppy=kpy
        seed_r = np.sqrt(np.power(x,2)+np.power(y,2))
        seed_phi = np.arctan2(y,x)
        seed_vr = pred_vr
        seed_vphi = pred_vphi
    
    
    return track_x,track_y,track_px,track_py,measurement_x,measurement_y,measurement_px,measurement_py,prediction_x,prediction_y,prediction_px,prediction_py

### Initialization of the data
fname = "10000Particles_Pt0p9_BField20_errorPhi0p01.txt"
data = np.loadtxt(fname,dtype=float,delimiter='\t',usecols=range(41)) 
outfname = "Det2dKalmanData"+fname
outfname2 = "DetKalmanData"+fname


#Looping Kalman Filter for each particle
f = open(outfname,"w+")
f2 = open(outfname2,"w+")
for i in range(0,5):
    #First we calculate the possible trajectories considering the particle both being positive and negative
    Ptx,Pty,Ptpx,Ptpy,Pmx,Pmy,Pmpx,Pmpy,Ppx,Ppy,Pppx,Pppy = kalman2d(10,1/(realv),0,0,data[i,:],1)
    diffPM = np.sum(np.square(Pmx-Ppx)+np.square(Pmy-Ppy))
    Ntx,Nty,Ntpx,Ntpy,Nmx,Nmy,Nmpx,Nmpy,Npx,Npy,Nppx,Nppy = kalman2d(10,1/(realv),0,0,data[i,:],-1)
    diffNM = np.sum(np.square(Nmx-Npx)+np.square(Nmy-Npy))

    #Then we analyze which one has the least deviation with the measured values and chose this one as our real trajectory
    if diffNM>diffPM:
        FinalPredX,FinalPredY,FinalPredPX,FinalPredPY,FinalMeasX,FinalMeasY,FinalMeasPX,FinalMeasPY,FinalTraX,FinalTraY,FinalTraPX,FinalTraPY = Ppx,Ppy,Pppx,Pppy,Pmx,Pmy,Pmpx,Pmpy,Ptx,Pty,Ptpx,Ptpy
    else:
        FinalPredX,FinalPredY,FinalPredPX,FinalPredPY,FinalMeasX,FinalMeasY,FinalMeasPX,FinalMeasPY,FinalTraX,FinalTraY,FinalTraPX,FinalPY = Npx,Npy,Nppx,Nppy,Nmx,Nmy,Nmpx,Nmpy,Ntx,Nty,Ntpx,Ntpy

    plt.errorbar(FinalTraX,FinalTraY,FinalTraPX,FinalTraPY,fmt='.',markerfacecolor='blue',markersize=8,label='Kalman Points')  
    plt.errorbar(FinalMeasX,FinalMeasY,FinalMeasPX,FinalMeasPY,fmt='.',markerfacecolor='red',markersize=8,label='Measurements')
    plt.errorbar(FinalPredX,FinalPredY,FinalPredPX,FinalPredPY,fmt='.',markerfacecolor='black',markersize=8 ,label='Predictions')   

for i in range(1,11): #Plots circles in MPL
    circle = plt.Circle((0, 0), i*1 , color='r', fill=False)
    plt.gca().add_patch(circle)

plt.gca().set_aspect('equal') # Squared aspect ratio

plt.xlim([-10.5,10.5]) #Defines axis in MPL
plt.ylim([-10.5,10.5])
plt.legend()
plt.show()
f.close()
f2.close()