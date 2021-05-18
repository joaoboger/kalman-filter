import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import leastsq
from numpy.linalg import norm


q=1
m = 0.14 #pions em GeV/c2
pT = 0.9 #GeV
realv = (pT/np.sqrt(np.power(m,2)+np.power(pT,2))) #velocity adimensional
gamma = 1/np.sqrt(1-np.power(realv,2)) #gamma adimensional
K = (6.072E-3/2)*q/(gamma*m)
B = 20 #Magnetic field in Tesla

def EofMP(icond,t): # Equations of motion for a positive charge particle
    r=icond[0]
    phi=icond[1]
    vr=icond[2]
    vphi=icond[3]

    newR = vr
    newPhi = vphi
    newVR = (K*B+vphi)*r*vphi
    newVPhi = -(2*vphi+K*B)*vr/r

    return [newR,newPhi,newVR,newVPhi]

def EofMN(icond,t): # Equations of motion for a negative charge particle
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

def circleFit(KTrackX, KTrackY,charge,plot,writetofile): # Fits the trajectory of the particle
    # The fit is calculated in polar coordinates so we need to convert them
    KTrackR = np.sqrt(np.square(KTrackX)+np.square(KTrackY)) 
    KTrackPhi = np.arctan2(KTrackY,KTrackX)
    
    s = np.abs(-(KTrackY[-1]/KTrackX[-1])*KTrackX[4]+KTrackY[4])/np.sqrt(1+np.square(KTrackY[-1]/KTrackX[-1])) # saggita
    L = 5 # Half of the distance from te origin until the last hit

    Rinit = np.square(L)/(2*s)

    if charge == 1: # Fit for positive charge particle
        fitfunc = lambda p, r: p[0] + np.arccos(r/(2*p[1])) # General equation for a circle in polar coordinates which passes through the origin and make an angle p with the x-axis
        errfunc = lambda p, r, theta: theta - fitfunc(p, r) # Error function to be minimized
        init = [-np.pi/4, Rinit] # Initial paremeter of the least squares
        out = leastsq(errfunc,init,args=(KTrackR,KTrackPhi)) # Least squares through scipy.optimize's function leastsq
        c = out[0] # Optimized angle p of the circle
    elif charge == -1: # Fit for negative charge particle
        fitfunc = lambda p, r: -p[0] - np.arccos(r/(2*p[1]))  
        errfunc = lambda p, r, theta: theta - fitfunc(p, r)
        init = [3*np.pi/4, Rinit]
        out = leastsq(errfunc,init,args=(KTrackR,KTrackPhi))
        c = out[0]

    # Fit points converted back to cartesian coordiates in order to plot them
    FitTrackX = KTrackR * np.cos(fitfunc(c,KTrackR))
    FitTrackY = KTrackR * np.sin(fitfunc(c,KTrackR))

    if plot == True: 
        plt.plot(FitTrackX,FitTrackY,'--', label = r'$R_{found} = $ %.5s' % (c[1]))

    if writetofile == True:
        f.write("%s\n"%(c[1]))
    
    return 0

def conformalMapping(TrackX,TrackY):
    PolarPhi = np.array([])
    PhiV = np.array([])
    TrackU = np.array([])
    TrackV = np.array([])
    for i in range(len(TrackX)):
        x = TrackX[i]
        y = TrackY[i]
        u = TrackX[i]/(np.square(TrackX[i]) + np.square(TrackY[i]))
        v = TrackY[i]/(np.square(TrackX[i]) + np.square(TrackY[i]))
        TrackU = np.append(TrackU, u)
        TrackV = np.append(TrackV,v)
        PolarPhi = np.append(PolarPhi, np.arctan2(y,x))

        lfitfunc = lambda p,x: p[0]+p[1]*x
        lerrfunc = lambda p,x,y: y-fitfunc(p,x)
        init=[0,np.pi/4]
        out = leastsq(errfunc,init,args=(TrackU,TrackV))
        c=out[0]
        PhiV = np.append(PhiV, c[1])
        print("CM: %f\t%f\n"%(PhiV[i],PolarPhi[i]))
    
    axs.plot(TrackU,TrackV,'bo')
    axs.plot(TrackX,TrackY,'ro')
    return PolarPhi, PhiV

def lineTo2Layer(FirstLayerHit):
    lineTheta = np.arctan2(FirstLayerHit[1],FirstLayerHit[0])

    # So we want our hit (R=2,Theta=lineTheta)
    l2hit = np.array([])
    l2hit = np.append(l2hit, (2)*np.cos(lineTheta))
    l2hit = np.append(l2hit, (2)*np.sin(lineTheta))

    return l2hit

def findNearest2LayerHits(l2projhit,arrayLayerHits,tolerance):
    tol = tolerance

    selectedHits = [] # List with the index of selected hits in arrayLayerHits[1,:]

    for i in range(int(arrayLayerHits.shape[1]/4)):
        testingHit = np.array([arrayLayerHits[1,i*4],arrayLayerHits[1,i*4+1]])
        distance = norm(l2projhit-testingHit)
        if distance <= tolerance:
            selectedHits.append(i)
    
    return selectedHits

def simpleHistogram(data,nbins):
    data.sort()
    hEnd = data[-1]
    bins = np.linspace(0,round(hEnd),nbins)
    plt.figure(2)
    plt.hist(data,bins,density=True,color='#5e090d')
    plt.show()

def kalman2d(n,dt,p_v,q,Z,Charge,seed):
    Q=Charge

    realPhi = Z[0] # Angle Phi used to generate tracks

    initialppx=1/np.sqrt(2) #In the first iteration I'm assuming complete ignorance of the track angle
    initialppy=1/np.sqrt(2)

    seed_r = np.sqrt(np.square(seed[0])+np.square(seed[1])) #Seed of our first iteration
    seed_phi = np.arctan2(seed[1],seed[0])
    seed_vr = seed[2]
    seed_vphi = seed[3]

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
        me_x = Z[4*j-4]
        me_y = Z[4*j-3]
        me_px = Z[4*j-2]
        me_py = Z[4*j-1]

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

### Function which reconstructs trajectories of charged
### particles under a transverse magnetic field (B) to 
### it's, velocity travelling through 10 detectors.
### INPUTS: Particle velocity, array  of measurements,  
### seed of the first hit.
### OUTPUT: No output, it'll plot the reconstruction.

def kalmanFullReconstruction(realv,data,l1hit):
    #First we calculate the possible trajectories considering the particle both being positive and negative
    Ptx,Pty,Ptpx,Ptpy,Pmx,Pmy,Pmpx,Pmpy,Ppx,Ppy,Pppx,Pppy = kalman2d(10,1/(realv),0,0,data[i,:],1,l1hit)

    diffPM = np.sum(np.square(Pmx-Ppx)+np.square(Pmy-Ppy))
    Ntx,Nty,Ntpx,Ntpy,Nmx,Nmy,Nmpx,Nmpy,Npx,Npy,Nppx,Nppy = kalman2d(10,1/(realv),0,0,data[i,:],-1,l1hit)
    diffNM = np.sum(np.square(Nmx-Npx)+np.square(Nmy-Npy))
    charge=0

    #Then we analyze which one has the least deviation with the measured values and chose this one as our real trajectory
    if diffNM>diffPM:

        FinalPredX,FinalPredY,FinalPredPX,FinalPredPY,FinalMeasX,FinalMeasY,FinalMeasPX,FinalMeasPY,FinalTraX,FinalTraY,FinalTraPX,FinalTraPY = Ppx,Ppy,Pppx,Pppy,Pmx,Pmy,Pmpx,Pmpy,Ptx,Pty,Ptpx,Ptpy
        charge = 1
    else:
        FinalPredX,FinalPredY,FinalPredPX,FinalPredPY,FinalMeasX,FinalMeasY,FinalMeasPX,FinalMeasPY,FinalTraX,FinalTraY,FinalTraPX,FinalTraPY = Npx,Npy,Nppx,Nppy,Nmx,Nmy,Nmpx,Nmpy,Ntx,Nty,Ntpx,Ntpy
        charge = -1
    #plt.errorbar(FinalTraX,FinalTraY,FinalTraPX,FinalTraPY,fmt='.',markerfacecolor='blue',markersize=8)  

    #plt.errorbar(FinalTraX,FinalTraY,FinalTraPX,FinalTraPY,fmt='.',markerfacecolor='blue',markersize=8,label='Kalman Points')  
    #plt.errorbar(FinalMeasX,FinalMeasY,FinalMeasPX,FinalMeasPY,fmt='.',markerfacecolor='red',markersize=8,label='Measurements')
    #plt.errorbar(FinalPredX,FinalPredY,FinalPredPX,FinalPredPY,fmt='.',markerfacecolor='black',markersize=8 ,label='Predictions')  
        
    #circleFit(FinalMeasX,FinalMeasY,charge,1,0)


def dataInit(file):### Initialization of the data in which for each detector hit "i" we have 4 columns "x_i","y_i","error-x_i","error-y_i", and the first column gives the initial angle that the particle comes out from the origin
    fname = file.name
    data = np.loadtxt(file.path,dtype=float,delimiter=' ',usecols=range(4)) 
    outfname = "tmp"+fname

    LayerHits = []

    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    tmp5 = []
    tmp6 = []
    tmp7 = []
    tmp8 = []
    tmp9 = []
    tmp10 = []

    for i in range(len(data[:,0])):
        SqrRad = np.square(data[i,0])+np.square(data[i,1])
        Rad = int(np.round(np.sqrt(SqrRad)))

        eval('tmp'+str(Rad)+'.append(data[i,0])')
        eval('tmp'+str(Rad)+'.append(data[i,1])')
        eval('tmp'+str(Rad)+'.append(data[i,2])')
        eval('tmp'+str(Rad)+'.append(data[i,3])')
    
    for i in range(1,11):
        eval('LayerHits.append(tmp'+str(i)+')')

    arrayLayerHits = np.array(LayerHits) # Array of arrays composed of each layer's hits

    return data, outfname, arrayLayerHits

dirloc = r"/home/jboger/2021.1/kalman-filter/CKFdata"

NumbNear=[]

for file in os.scandir(dirloc):
    print(file.name.split)
    if file.name == 'out':
        break
    
    data, outfname, arrayLayerHits = dataInit(file)

    outfolder = r"/home/jboger/2021.1/kalman-filter/CKFdata/out"
    completeOutput = os.path.join(outfolder,outfname)

    f = open(completeOutput,"w+")

    #Looping Kalman Filter for each particle
    #### Loop over tracks
    fig,axs = plt.subplots()
    for i in range(0,100):

        l1hit = np.array([]) # Seed to get the initial conditions for our prediction: x-coordinate, y-coordinate, radial velocity, angular velocity
        l1hit = np.append(l1hit, arrayLayerHits[0,4*i])
        l1hit = np.append(l1hit, arrayLayerHits[0,4*i+1])
        l1hit = np.append(l1hit, arrayLayerHits[0,4*i+2])
        l1hit = np.append(l1hit, arrayLayerHits[0,4*i+3])

        ### Projection a line through origin and L1Hit to the second layer (R=2)
        l2projhit = lineTo2Layer(l1hit) 

        ### Find nearest hits of second layer
        selHits = findNearest2LayerHits(l2projhit,arrayLayerHits,0.05)

        NumbNear.append(len(selHits))
        
        mks=7
        pointsLayer=[[l1hit[0],l2projhit[0]],[l1hit[1],l2projhit[1]]]
        plt.plot(pointsLayer[0],pointsLayer[1],'--',marker='^', color='green',ms=mks)

        for i in range(len(selHits)):
            plt.plot(arrayLayerHits[1,selHits[i]*4],arrayLayerHits[1,selHits[i]*4+1], marker = 'x', color='blue', ms=mks)



for i in range(1,11): #Plots circles in MPL
    circle = plt.Circle((0, 0), i*1 , color='r', fill=False)
    plt.gca().add_patch(circle)

plt.gca().set_aspect('equal') # Squared aspect ratio
plt.title(r"$p_T=0.9$GeV, $m_{\pi}=0.14$GeV/c$^2$, $B=20$T",size=15)

plt.xlim([0,2.5]) #Defines axis in MPL
plt.ylim([0,2.5])
plt.legend()
plt.show()
print(NumbNear)
simpleHistogram(NumbNear,10)
f.close()
