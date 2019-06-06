import matplotlib.pyplot as plt
from autograd.scipy.stats import norm, rv_continuous, randint
from autograd.scipy.stats import multivariate_normal as mvn
import autograd.numpy as np


class gmm(rv_continuous):
	def __init__(self, normals, probs):
		super().__init__()
		self.normals = normals
		self.probs = probs
		self.num = len(probs)

		assert(self.num == len(normals))
	
	def rvs(self, size=1):
		comps = np.random.choice(self.num, size=size, replace=True, p=self.probs)
		return np.array([self.normals[i].rvs().tolist() for i in comps])

	def pdf(self, x):
		return sum([n.pdf(x) for n in self.normals])

def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
		x = np.linspace(*xlimits, num=numticks)
		y = np.linspace(*ylimits, num=numticks)
		X, Y = np.meshgrid(x, y)
		zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
		Z = zs.reshape(X.shape)
		plt.contour(X, Y, Z)
		ax.set_yticks([])
		ax.set_xticks([])
	

probs = [0.3, 0.7]
n1 = mvn(mean=[-1.0,-1.0], cov=np.diag([1.0,1.0]))
n2 = mvn(mean=[1.0,1.0], cov=np.array([[1.0,0.5],[0.5,1.0]]))

mix = gmm([n1,n2], probs)
samples = mix.rvs(10)
print(mix.pdf(samples)) 

fig = plt.figure()
ax = fig.add_subplot(111)
plot_isocontours(ax, mix.pdf, xlimits=[-4,4],ylimits=[-4,4])
plt.show()
