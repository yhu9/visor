import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats


data = np.load(sys.argv[1])

plt.hist(data,normed=False,bins=50)
a,b,c = stats.gamma.fit(data)
xx = np.linspace(0,1,50)
pdf_gamma = stats.gamma.pdf(xx,a,b,c)
plt.plot(xx,pdf_gamma)
plt.yticks()
plt.show()



