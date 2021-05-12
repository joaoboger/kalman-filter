import os
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

data = np.loadtxt("/home/jboger/2021.1/kalman-filter/out/CurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p09.txt",dtype=float,delimiter='\n',usecols=range(1))
data2 = np.loadtxt("/home/jboger/2021.1/kalman-filter/out/CurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p1.txt",dtype=float,delimiter='\n',usecols=range(1))
data3 = np.loadtxt("/home/jboger/2021.1/kalman-filter/out/MeasCurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p09.txt",dtype=float,delimiter='\n',usecols=range(1))
data4 = np.loadtxt("/home/jboger/2021.1/kalman-filter/out/MeasCurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p1.txt",dtype=float,delimiter='\n',usecols=range(1))

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
bins1 = np.linspace(lrange1,hirange1,100) #Number of bins and interval where it's defined
bins2 = np.linspace(lrange2,hirange2,100)
bins3 = np.linspace(lrange3,hirange3,100) #Number of bins and interval where it's defined
bins4 = np.linspace(lrange4,hirange4,100)
funcbins1 = np.linspace(lrange1,hirange1,1000) 
funcbins2 = np.linspace(lrange2,hirange2,1000)
funcbins3 = np.linspace(lrange3,hirange3,1000) 
funcbins4 = np.linspace(lrange4,hirange4,1000)

lsize = 14 #Legend size

n,outbins,_=axs[0,0].hist(data,bins1,density=True,color='#5e090d') #Histogram plot
obins = outbins[:-1]
obins = obins + (2*hrange1/(2*100))
out=leastsq(errfunc,init,args=(obins,n))
c=out[0]
axs[0,0].plot(funcbins1,fitfunc(c,funcbins1),label=r"$\frac{A}{\sigma\sqrt{2}\pi} \exp \left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma ^2}\right)$",color='#e6d3ab')
axs[0,0].set_title("Kalman : $\sigma_{\phi} = 0.09$, $A= %.2f$, $\mu = %.2f$, $\sigma = %.2f$" % (c[2],c[1],c[0]),fontsize=14) #Title of the subplot
axs[0,0].set_ylabel("Distribution of tracks",fontsize=lsize)
axs[0,0].legend()

n,outbins,_=axs[0,1].hist(data2,bins2,density=True,color='#5e090d') #Histogram plot
obins = outbins[:-1]
obins = obins + (2*hrange2/(2*100))
out=leastsq(errfunc,init,args=(obins,n))
c=out[0]
axs[0,1].plot(funcbins2,fitfunc(c,funcbins2),label=r"$\frac{A}{\sigma\sqrt{2}\pi} \exp \left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma ^2}\right)$",color='#e6d3ab')
axs[0,1].set_title("Kalman : $\sigma_{\phi} = 0.1$, $A= %.2f$, $\mu = %.2f$, $\sigma = %.2f$" % (c[2],c[1],c[0])) #Title of the subplot
axs[0,1].set_ylabel("Distribution of tracks",fontsize=lsize) #Axis label of plots
axs[0,1].legend()

n,outbins,_=axs[1,0].hist(data3,bins3,density=True,color='#5e090d') #Histogram plot
obins = outbins[:-1]
obins = obins + (2*hrange3/(2*100))
out=leastsq(errfunc,init,args=(obins,n))
c=out[0]
axs[1,0].plot(funcbins3,fitfunc(c,funcbins3),label=r"$\frac{A}{\sigma\sqrt{2}\pi} \exp \left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma ^2}\right)$",color='#e6d3ab')
axs[1,0].set_title("Measure : $\sigma_{\phi} = 0.09$, $A= %.2f$, $\mu = %.2f$, $\sigma = %.2f$" % (c[2],c[1],c[0])) #Title of the subplot
axs[1,0].set_xlabel(r"$R_{fitted}$",fontsize=lsize)
axs[1,0].set_ylabel("Distribution of tracks",fontsize=lsize)
axs[1,0].legend()

n,outbins,_=axs[1,1].hist(data4,bins4,density=True,color='#5e090d') #Histogram plot
obins = outbins[:-1]
obins = obins + (2*hrange4/(2*100))
out=leastsq(errfunc,init,args=(obins,n))
c=out[0]
axs[1,1].plot(funcbins4,fitfunc(c,funcbins4),label=r"$\frac{A}{\sigma\sqrt{2}\pi} \exp \left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma ^2}\right)$",color='#e6d3ab')
axs[1,1].set_title("Measure : $\sigma_{\phi} = 0.1$, $A= %.2f$, $\mu = %.2f$, $\sigma = %.2f$" % (c[2],c[1],c[0])) #Title of the subplot
axs[1,1].set_xlabel(r"$R_{fitted}$",fontsize=lsize)
axs[1,1].set_ylabel("Distribution of tracks",fontsize=lsize)
axs[1,1].legend()

plt.rcParams["axes.labelsize"] = 12
plt.show()
