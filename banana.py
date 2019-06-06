from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

mu = np.array([0.0,0.0])
cov = np.array([[1.0,0.9],[0.9,1.0]])

samples = mvn.rvs(size=1000, mean=mu, cov=cov)
samples[:,1] = samples[:,1] + samples[:,0]**2 + 1
print(samples)
plt.scatter(samples[:,0], samples[:,1], s=10)

plt.show()


