import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

data = np.loadtxt("CurvesRfoundData10000Particles_Pt0p9_BField20_errorPhi0p01.txt",dtype=float,delimiter='\n',usecols=range(1)) #Loading file with two columns of data

fig,axs = plt.subplots(1,2) #Configuring side by side plots

hrange = 1.5 #Range of plot
bins = np.linspace(15-hrange,15+hrange,100) #Number of bins and interval where it's defined

#data[0]=15-data[0]
#data[:,1]=15-data[:,1]

mu0, std0 = norm.fit(data) #Normal gaussian distribution fit
axs[0].hist(data,bins,density=True,label=r"$10^5$ events") #Histogram plot
axs[0].set_title("$\sigma_{\phi} = 0.01$ with fitting $\mu=%.5s$, $\sigma_{gaus}=%.5s$"%(mu0,std0)) #Title of the subplot
axs[0].set(xlabel="$R_{fitted}$",ylabel="Distribution of tracks") #Axis label of plots
p = norm.pdf(bins, mu0, std0) #Fitted Gaussian definition
axs[0].plot(bins, p, 'k', linewidth=2) #Gaussian plot

mu1, std1 = norm.fit(0)
axs[1].hist(0,bins,density=True)
axs[1].set_title("$\sigma_{\phi} = X$ with fitting $\mu=%.5s$, $\sigma_{gaus}=%.5s$"%(mu1,std1))
axs[1].set(xlabel="$(\phi_{Kalman}-\phi_{real})/\phi_{real}$",ylabel="Distribution of tracks")
p = norm.pdf(bins, mu1, std1)
axs[1].plot(bins, p, 'k', linewidth=2)

#p = norm.pdf(bins, mu, std)
#plt.plot(bins, p, 'k', linewidth=2)
#title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
#plt.title(title)
#print("mu:%s\tstd:%s\n"%(mu,std,))

plt.show()
