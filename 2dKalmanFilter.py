import math
import numpy as np
import matplotlib.pyplot as plt

### This function returns the updated position
### The input parameters are:
### the current position x, y
### the current position vx, dy
### the elapsed time interval dt
### newX, newY are the updated position
def updatePosition(x, y, vx, vy, dt):
    newX = x + vx * dt
    newY = y + vy * dt
    return newX, newY

def kalman2d(n,dt,p_v,q,Z):
    x=0 #Initial position of the robot, in the origin
    y=0
    vx=Z[0]/math.sqrt(Z[0]**2+Z[1]**2) #Initial velocities: x being cossine, and y sine
    vy=Z[1]/math.sqrt(Z[0]**2+Z[1]**2)
    px=0.005*Z[0]/math.sqrt(Z[0]**2+Z[1]**2) #Uncertainty of the initial position
    py=0.005*Z[1]/math.sqrt(Z[0]**2+Z[1]**2)
    track_x = np.array([]) #Arrays all positions and errors will be stored to fit the track
    track_y = np.array([])
    track_px = np.array([])
    track_py = np.array([])

    for j in range(1,n+1):
        #Prediction step#
        pred_x, pred_y = updatePosition(x, y, vx, vy, dt) #Prediction of positions x and y
        pred_px = px + (dt**2)*p_v + q #Prediction of the propagated error
        pred_py = py + (dt**2)*p_v + q

        #Update step#
        kgx = pred_px/(pred_px+Z[4*j-2]) #Kalman gain
        kgy = pred_py/(pred_py+Z[4*j-1])
        kx = pred_x + kgx*(Z[4*j-4]-pred_x) #Kalman positions
        ky = pred_y + kgy*(Z[4*j-3]-pred_y)
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

    coef_fit = np.polyfit(track_x,track_y,1) #coef_fit is an 1x2 array [c1,c0] where the coefficients are from the polynomial "p(x)=c0+c1*x"

    f.write("%s\t%s\n" % (coef_fit[0],coef_fit[1]))

    print("%s\t%s\n" % (coef_fit[0],coef_fit[1]))

    #Uncomment to generate plots with data and fitting#
    def tfit(t):
        return coef_fit[0]*t + coef_fit[1] 
    
    t1=np.arange(0.0,10.0,0.01)
    
    plt.plot(track_x,track_y,'bo')
    
    plt.plot(t1,tfit(t1),'k')

    return x,y,kpx,kpy

#Our model will say that the robot moves at speed 1 in the first quadrant of a circle.

data = np.loadtxt("100Particles_errorPhi0p05.txt",dtype=float,delimiter='\t',usecols=range(40)) #Initialization of the data

#Looping Kalman Filter for each particle
f = open("coef-fit.txt","w+")
for i in range(0,10):
    #print("Track %s Phi angle" % (i+1))
    kalman2d(10,1,0.005,0.005,data[i][:])

for i in range(1,11):
    circle = plt.Circle((0, 0), i*1 , color='r', fill=False)
    plt.gca().add_patch(circle)

plt.xlim([0,10.5])
plt.ylim([0,10.5])
plt.rcParams["figure.figsize"] = (20,20) 
plt.show()

f.close()