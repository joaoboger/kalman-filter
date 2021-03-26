import math
import numpy as np
import matplotlib.pyplot as plt

### This function returns the updated position
### The input parameters are:
### the current position x, y
### the current position vx, dy
### the elapsed time interval dt
### newX, newY are the updated position
def updatePosition(x, y, vx, vy, q, dt):
    newX = x + vx * dt + q
    newY = y + vy * dt + q
    return newX, newY

def kalman2d(n,dt,p_v,q,Z):
    x=0 #Initial position of the robot, in the origin
    y=0
    realPhi = Z[0]
    seed_x = Z[1]
    seed_y = Z[2]
    print("x: %s y: %s \t" % (seed_x,seed_y))
    initialPhi=np.arctan2(seed_y,seed_x)
    vx=seed_x/math.sqrt(seed_x**2+seed_y**2) #Initial velocities: x being cossine, and y sine
    vy=seed_y/math.sqrt(seed_x**2+seed_y**2)
    initialVarPhi= np.power(0.10 * initialPhi, 2) #10% error
    #vx=1/math.sqrt(2)
    #vy=1/math.sqrt(2)
    track_x = np.array([]) #Arrays all positions and errors will be stored to fit the track
    track_y = np.array([])
    track_px = np.array([])
    track_py = np.array([])

    for j in range(1,n+1):
        #Prediction step#
        pred_x, pred_y = updatePosition(x, y, vx, vy, q, dt) #Prediction of positions x and y
        pred_px = pred_y * np.sqrt(initialVarPhi)#Prediction of the propagated error
        pred_py = pred_x* np.sqrt(initialVarPhi)

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
        
        x=kx
        y=ky
        px=kpx
        py=kpy

        #print("Prediction: %s\t%s\t%s\t%s\n" % (pred_x,pred_y,pred_px,pred_py))
        #print("Filtered: %s\t%s\t%s\t%s\n" % (kx,ky,kpx,kpy))

    coef_fit = np.polyfit(track_x,track_y,1) #coef_fit is an 1x2 array [c0,c1] where the coefficients are from the polynomial "p(x)=c0*x+c1"
    kalmanPhi = np.arctan2(track_y[0],(track_y[0]-coef_fit[1])/coef_fit[0]) #Once we only have the Phi from the fit, and we want only tracks in the first two quadrants for now I take a positive value of Y and discover it's X coordinate in the fit function
    print("OIA OS ANGULO => MEDIDO: %s\t DI VERDADE: %s \n" % (kalmanPhi,realPhi))
    f.write("%s\t%s\n" % (realPhi-kalmanPhi,(realPhi-kalmanPhi)/realPhi))

    #Uncomment to generate plots in MPL with data and fitting#
    def tfit(t):
        return coef_fit[0]*t + coef_fit[1] 
    
    t1=np.arange(0.0,10.0,0.01)
    
    plt.plot(t1,tfit(t1),'k', linewidth=0.5)
    
    plt.errorbar(track_x,track_y,track_px,track_py,'bo',markersize=1.5)  
    
    
    return x,y,kpx,kpy

#Our model will say that the robot moves at speed 1 in the first quadrant of a circle.

data = np.loadtxt("100Particles_errorPhi0p05.txt",dtype=float,delimiter='\t',usecols=range(41)) #Initialization of the data

#Looping Kalman Filter for each particle
f = open("phidiff.txt","w+")
for i in range(0,100):
    #print("Track %s Phi angle" % (i+1))
    kalman2d(10,1,0,0,data[i,:])

#Uncomment to generate plots in MPL with data and fitting#
for i in range(1,11): #Plots circles in MPL
    circle = plt.Circle((0, 0), i*1 , color='r', fill=False)
    plt.gca().add_patch(circle)

plt.xlim([-10.5,10.5]) #Defines axis in MPL
plt.ylim([0,10.5])
plt.show()
f.close()
