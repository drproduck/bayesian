import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
import matplotlib
matplotlib.use('tkagg')

# def log_density(x):
# 	mu, log_sigma = x[:, 0], x[:, 1]
# 	sigma_density = norm.logpdf(log_sigma, 0, 1.35)
# 	mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
# 	return sigma_density + mu_density

def log_density(x):
	std = np.array([[0.5,1],[1,0.5]])
	return mvn.logpdf(x, mean=np.array([0.0,0.0]), cov=np.matmul(std, std.T))

def variational_density(x, params, D):
	std = params[D:].reshape([D,D])
	cov_mat = np.matmul(std, std.T)
	return mvn.pdf(x, mean=params[:D], cov=cov_mat)
	

def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
	x = np.linspace(*xlimits, num=numticks)
	y = np.linspace(*ylimits, num=numticks)
	X, Y = np.meshgrid(x, y)
	zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
	Z = zs.reshape(X.shape)
	plt.contour(X, Y, Z)
	ax.set_yticks([])
	ax.set_xticks([])


def lower_bound(variational_params, logprob_func, D, num_samples):
	mu, std = variational_params[:D], variational_params[D:].reshape(D,D)
	cov_mat = np.matmul(std, std.T)

	samples = np.matmul(npr.randn(num_samples, D), std) + mu
	return mvn.entropy(mu, cov_mat) + np.mean(logprob_func(samples))


grad_func = grad(lower_bound)
init_mean = np.array([-1,-1])
init_std = np.random.rand(4)
params = np.concatenate((init_mean, init_std))

fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)
lr = 1e-1
D = 2
num_samples = 2000
for i in range(100):
	plt.cla()
	params = params + lr * grad_func(params, log_density, D, num_samples)	
	plot_isocontours(ax, lambda x: np.exp(log_density(x)))
	plot_isocontours(ax, lambda x: variational_density(x, params, D))
	plt.draw()


	plt.pause(1.0 / 40.0)
