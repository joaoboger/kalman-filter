import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("phidiff.txt",dtype=float,delimiter='\t',usecols=range(2))

plt.hist(data[:,0],100)
plt.show()