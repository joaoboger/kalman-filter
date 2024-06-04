import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import leastsq
from scipy.signal import find_peaks

data = np.loadtxt("/home/jboger/2021.1/kalman-filter/out/CurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p1.txt",dtype=float,delimiter='\n',usecols=range(1))
data2 = np.loadtxt("/home/jboger/2021.1/kalman-filter/out/CurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p09.txt",dtype=float,delimiter='\n',usecols=range(1))
data3 = np.loadtxt("/home/jboger/2021.1/kalman-filter/out/MeasCurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p1.txt",dtype=float,delimiter='\n',usecols=range(1))
data4 = np.loadtxt("/home/jboger/2021.1/kalman-filter/out/MeasCurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p09.txt",dtype=float,delimiter='\n',usecols=range(1))

fig,axs = plt.subplots(2,2) #Configuring side by side plots

fitfunc  = lambda p, x: (p[2]/(p[0]*np.pi*(np.sqrt(2))))*np.exp(-0.5*((x-p[1])/p[0])**2) #Gaussian Fit Function
errfunc  = lambda p, x, y: (y - fitfunc(p, x))
init  = [15.0, 0.5, 1.0] #Initial conditions of least squares

expecmean1 = 20 #Mean of the plot range
expecmean2 = 20
expecmean3 = 20
expecmean4 = 20
hrange1 = 20 #Range of plot
hrange2 = 20
hrange3 = 20
hrange4 = 20
hirange1,lrange1 = expecmean1+hrange1,expecmean1-hrange1
hirange2,lrange2 = expecmean2+hrange2,expecmean2-hrange2
hirange3,lrange3 = expecmean3+hrange3,expecmean3-hrange3
hirange4,lrange4 = expecmean4+hrange4,expecmean4-hrange4
bins1 = np.linspace(lrange1,hirange1,500) #Number of bins and interval where it's defined
bins2 = np.linspace(lrange2,hirange2,500)
bins3 = np.linspace(lrange3,hirange3,500) #Number of bins and interval where it's defined
bins4 = np.linspace(lrange4,hirange4,500)
funcbins1 = np.linspace(lrange1,hirange1,1000) 
funcbins2 = np.linspace(lrange2,hirange2,1000)
funcbins3 = np.linspace(lrange3,hirange3,1000) 
funcbins4 = np.linspace(lrange4,hirange4,1000)

### treatData returns data without 10% outliers, taken 5% from each side
def treatData(data):
    newData = data[data != 0.]
    dLen = len(newData)
    nRange = int(np.round(0.05*dLen))
    newData = newData[nRange:(dLen-(nRange+1))]

    return newData

### peakAndVariance returns peaks and variance of data in two separate arrays, respectively
def peakAndVariance(data,userProminence):
    peaks = find_peaks(data, prominence = userProminence)
    peaks = peaks[0]

    variance = np.var(treatData(data))

    return peaks, variance

def delOutliers(data):
    data = data[data>0]
    data = data[data<40]
    return data

data, data2, data3, data4 = delOutliers(data), delOutliers(data2), delOutliers(data3), delOutliers(data4)

lsize = 14 #Legend size

n,outbins,_=axs[0,0].hist(data,bins1,density=True,color='#5e090d') #Histogram plot
obins = outbins[:-1]
obins = obins + (2*hrange1/(2*100))
peak1, variance1 = peakAndVariance(n,0.1)
sigmaData = np.sqrt(np.var(data))
out=leastsq(errfunc,init,args=(obins,n))
c=out[0]
axs[0,0].plot(funcbins1,fitfunc(c,funcbins1),label=r"$\frac{A}{\sigma\sqrt{2}\pi} \exp \left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma ^2}\right)$",color='#e6d3ab')
axs[0,0].set_title("Kalman : $\sigma_{\phi} = 0.01$, $A= %.2f$, $\mu = %.2f$, $\sigma = %.2f$" % (c[2],c[1],sigmaData),fontsize=14) #Title of the subplot
axs[0,0].set_ylabel("Distribution of tracks",fontsize=lsize)
axs[0,0].legend()

n,outbins,_=axs[0,1].hist(data2,bins2,density=True,color='#5e090d') #Histogram plot
obins = outbins[:-1]
obins = obins + (2*hrange2/(2*100))
peaks = find_peaks(n,prominence=0.1)
print(obins[peaks[0][0]])
sigmaData2 = np.sqrt(np.var(data2))
out=leastsq(errfunc,init,args=(obins,n))
c=out[0]
axs[0,1].plot(funcbins2,fitfunc(c,funcbins2),label=r"$\frac{A}{\sigma\sqrt{2}\pi} \exp \left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma ^2}\right)$",color='#e6d3ab')
axs[0,1].set_title("Kalman : $\sigma_{\phi} = 0.02$, $A= %.2f$, $\mu = %.2f$, $\sigma = %.2f$" % (c[2],c[1],sigmaData2),fontsize=14) #Title of the subplot
axs[0,1].set_ylabel("Distribution of tracks",fontsize=lsize) #Axis label of plots
axs[0,1].legend()

n,outbins,_=axs[1,0].hist(data3,bins3,density=True,color='#5e090d') #Histogram plot
obins = outbins[:-1]
obins = obins + (2*hrange3/(2*100))
peaks = find_peaks(n,prominence=0.1)
print(obins[peaks[0][0]])
sigmaData3 = np.sqrt(np.var(data3))
out=leastsq(errfunc,init,args=(obins,n))
c=out[0]
axs[1,0].plot(funcbins3,fitfunc(c,funcbins3),label=r"$\frac{A}{\sigma\sqrt{2}\pi} \exp \left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma ^2}\right)$",color='#e6d3ab')
axs[1,0].set_title("Measure : $\sigma_{\phi} = 0.01$, $A= %.2f$, $\mu = %.2f$, $\sigma = %.2f$" % (c[2],c[1],sigmaData3),fontsize=14) #Title of the subplot
axs[1,0].set_xlabel(r"$R_{fitted}$",fontsize=lsize)
axs[1,0].set_ylabel("Distribution of tracks",fontsize=lsize)
axs[1,0].legend()

n,outbins,_=axs[1,1].hist(data4,bins4,density=True,color='#5e090d') #Histogram plot
obins = outbins[:-1]
obins = obins + (2*hrange4/(2*100))
peaks = find_peaks(n,prominence=0.1)
print(obins[peaks[0][0]])
sigmaData4 = np.sqrt(np.var(data4))
out=leastsq(errfunc,init,args=(obins,n))
c=out[0]
axs[1,1].plot(funcbins4,fitfunc(c,funcbins4),label=r"$\frac{A}{\sigma\sqrt{2}\pi} \exp \left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma ^2}\right)$",color='#e6d3ab')
axs[1,1].set_title("Measure : $\sigma_{\phi} = 0.02$, $A= %.2f$, $\mu = %.2f$, $\sigma = %.2f$" % (c[2],c[1],sigmaData4),fontsize=14) #Title of the subplot
axs[1,1].set_xlabel(r"$R_{fitted}$",fontsize=lsize)
axs[1,1].set_ylabel("Distribution of tracks",fontsize=lsize)
axs[1,1].legend()

plt.rcParams["axes.labelsize"] = 12
plt.show()
