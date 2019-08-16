"""
    Author: Masa Hu
    Email: huynshen@msu.edu

    view_hist.py takes a .npy histogram data as input and visualizes the histogram data with a bin size of 50
"""

#Native library imports
import sys

#OPEN SOURCE IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

######################################################################################################
data = np.load(sys.argv[1])
plt.hist(data,normed=False,bins=50)
a,b,c = stats.gamma.fit(data)
xx = np.linspace(0,1,50)
pdf_gamma = stats.gamma.pdf(xx,a,b,c)
plt.plot(xx,pdf_gamma)
plt.yticks()
plt.show()



