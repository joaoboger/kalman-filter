import math
import numpy as np
import matplotlib.pyplot as plt

q = 1
m = 0.14 #pions em GeV/c2
pT = 0.9 #GeV
realv = (pT/np.sqrt(np.power(m,2)+np.power(pT,2))) #velocity adimensional
gamma = 1/np.sqrt(1-np.power(realv,2)) #gamma adimensional
K = q/(gamma*m) # 1/GeV
B = 20 #Magnetic field in Tesla


### This function returns the updated position
### The input parameters are:
### the current position x, y
### the current velocity vx, vy
### the elapsed time interval dt
### newX, newY are the updated position

def eomR(vr):
    newR = vr
    return newR

def eomPhi(r,vphi):
    newPhi = vphi/r
    return newPhi

def eomVR(vphi):
    newVR = K*vphi*B
    return newVR

def eomVPhi(r,vr): 
    newVPhi = -K*vr*B/r
    return newVPhi


def updatePosition(r, phi, vr, vphi, dt, j):
    #The positions are updated solving the equations of motion of a charged 
    #particle travelling in a media with a constant magnetic field transverse
    #to it's velocity. 

    #k's are RK4 coefficients of the coordinate r's EDO
    #l's are RK4 coefficients of the coordinate phi's EDO
    #m's are RK4 coefficients of the velocity vr's EDO
    #n's are RK4 coefficients of the velocity vphi's EDO

    radiuserror=0.01
    nosteps=100000
    dt=dt/10000

    for k in range(nosteps):
        if(r<j-radiuserror):
            #print(r,phi,vr,vphi)
            k1 = eomR(vr)
            l1 = eomPhi(r,vphi)
            m1 = eomVR(vphi)
            n1 = eomVPhi(r,vr)

            k2 = eomR(vr+dt*m1/2)
            l2 = eomPhi(r+dt*k1/2,vphi+dt*n1/2)
            m2 = eomVR(vphi+dt*n1/2)
            n2 = eomVPhi(r+dt*k1/2,vr+dt*m1/2)

            k3 = eomR(vr+dt*m2/2)
            l3 = eomPhi(r+dt*k2/2,vphi+dt*n2/2)
            m3 = eomVR(vphi+dt*n2/2)
            n3 = eomVPhi(r+dt*k2/2,vr+dt*m2/2)

            k4 = eomR(vr+dt*m3)
            l4 = eomPhi(r+dt*k3,vphi+dt*n3)
            m4 = eomVR(vphi+dt*n3)
            n4 = eomVPhi(r+dt*k3,vr+dt*m3)

            r = r + dt*(k1 + 2*k2 + 2*k3 + k4)/6
            phi = phi + dt*(l1 + 2*l2 + 2*l3 + l4)/6
            vr = vr + dt*(m1 + 2*m2 + 2*m3 + m4)/6
            vphi = vphi + dt*(n1 + 2*n2 + 2*n3 + n4)/6
            
        else:
            break

    return r, phi, vr, vphi

def kalman2d(n,dt,p_v,q,Z):
    x=0 #Initial position of the particle, in the origin
    y=0
    realPhi = Z[0]
    seed_x = Z[1]
    seed_y = Z[2]
    #seed_x=1/np.sqrt(2)
    #seed_y=1/np.sqrt(2)
    #initialppx = Z[3] # In the beggining we don't have the slightest idea where the particle must be thus 100% error
    #initialppy = Z[4]
    initialppx=1/np.sqrt(2)
    initialppy=1/np.sqrt(2)

    #vx=seed_x/math.sqrt(seed_x**2+seed_y**2) #Initial velocities: x being cossine, and y sine
    #vy=seed_y/math.sqrt(seed_x**2+seed_y**2)
    #vx=1/math.sqrt(2)
    #vy=1/math.sqrt(2)
    seed_r = 0.1
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

        print("entradas rk4: r=%s phi=%s vr=%s vphi=%s"%(seed_r,seed_phi,seed_vr,seed_vphi))
        #Prediction step#
        pred_r, pred_phi, pred_vr, pred_vphi = updatePosition(seed_r, seed_phi, seed_vr, seed_vphi, dt, j) #Prediction of positions x and y

        print("RK4: r=%s phi=%s vr=%s vphi=%s" % (pred_r,pred_phi,pred_vr,pred_vphi))

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
        #print("%s    %s    %s    %s" % (kx,ky,kpx,kpy))
        
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
        
        x=kx
        y=ky
        initialppx=kpx
        initialppy=kpy
        seed_r = np.sqrt(np.power(x,2)+np.power(y,2))
        seed_phi = np.arctan2(y,x)
        seed_vr = pred_vr
        seed_vphi = pred_vphi

        print("Finally Phi: %s"%(seed_phi/np.pi))

        #print("Prediction: %s\t%s\t%s\t%s\n" % (pred_x,pred_y,pred_px,pred_py))
        #print("Filtered: %s\t%s\t%s\t%s\n" % (kx,ky,kpx,kpy))

    #Uncomment to generate plots in MPL with data and fitting#
    #def tfit(t):
    #    return coef_fit[0]*t + coef_fit[1] 
    #
    #t1=np.arange(0.0,10.0,0.01)
    #
    #plt.plot(t1,tfit(t1),'k', linewidth=0.5)
    
    plt.errorbar(track_x,track_y,track_px,track_py,fmt='.',markerfacecolor='blue',markersize=8,label='Kalman Points')  

    plt.errorbar(measurement_x,measurement_y,measurement_px,measurement_py,fmt='.',markerfacecolor='red',markersize=8,label='Measurements')

    plt.errorbar(prediction_x,prediction_y,prediction_px,prediction_py,fmt='.',markerfacecolor='black',markersize=8 ,label='Predictions')
    
    
    return x,y,kpx,kpy

#Our model will say that the robot moves at speed 1 in the first quadrant of a circle.

fname = "10000Particles_Pt0p9_BField20_errorPhi0p01.txt"
data = np.loadtxt(fname,dtype=float,delimiter='\t',usecols=range(41)) #Initialization of the data
outfname = "Det2dKalmanData"+fname
outfname2 = "DetKalmanData"+fname


#Looping Kalman Filter for each particle
f = open(outfname,"w+")
f2 = open(outfname2,"w+")
for i in range(0,1):
    #print("Track %s Phi angle" % (i+1))
    kalman2d(10,1/(10*realv),0,0,data[i,:])

#Uncomment to generate plots in MPL with data and fitting#
for i in range(1,11): #Plots circles in MPL
    circle = plt.Circle((0, 0), i*1 , color='r', fill=False)
    plt.gca().add_patch(circle)

plt.xlim([-10.5,10.5]) #Defines axis in MPL
plt.ylim([-10.5,10.5])
plt.legend()
plt.show()
f.close()
f2.close()