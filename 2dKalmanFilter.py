import math
import numpy as np

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


    for j in range(1,n+1):
        #Prediction step#
        new_x, new_y = updatePosition(x, y, vx, vy, dt) #Prediction of positions x and y
        new_px = px + (dt**2)*p_v + q #Prediction of the propagated error
        new_py = py + (dt**2)*p_v + q

        #Update step#
        kgx = new_px/(new_px+Z[4*j-2]) #Kalman gain
        kgy = new_py/(new_py+Z[4*j-1])
        kx = new_x + kgx*(Z[4*j-4]-new_x) #Kalman positions
        ky = new_y + kgy*(Z[4*j-3]-new_y)
        kpx = (1-kgx)*new_px #Kalman uncertainties
        kpy = (1-kgy)*new_py
        print("%s    %s    %s    %s" % (kx,ky,kpx,kpy))
        x=kx
        y=ky
        px=kpx
        py=kpy
    #a=Z[0]/Z[1]
    #b=x/y
    #print("Iter0: %s   Iter10: %s   Error: %.1f %%" % (a,b,100*abs(a-b)/a))
    return new_x,new_y,new_px,new_py

#Our model will say that the robot moves at speed 1 in the first quadrant of a circle.

data = np.loadtxt("100Particles_errorPhi0p05.txt",dtype=float,delimiter='\t',usecols=range(40)) #Initialization of the data

#Looping Kalman Filter for each particle
for i in range(0,100):
    #print("Track %s Phi angle" % (i+1))
    kalman2d(10,1,0.005,0.005,data[i][:])
