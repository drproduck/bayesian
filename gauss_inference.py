import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc.optimizers import adam


def black_box_variational_inference(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, std = params[:D], params[D:].reshape([D,D])
        return mean, std

    def gaussian_entropy(std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + 0.5 * np.log(np.linalg.det(std @ std.T))

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, std = unpack_params(params)
        samples = rs.randn(num_samples, D) @ std + mean
        lower_bound = gaussian_entropy(std) + np.mean(logprob(samples, t))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params



if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-density.
    D = 2
    # def log_density(x, t):
    #     mu, log_sigma = x[:, 0], x[:, 1]
    #     sigma_density = norm.logpdf(log_sigma, 0, 1.35)
    #     mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
    #     return sigma_density + mu_density
    def log_density(x, t):
        std = np.array([[0.5,1],[1,0.5]])
        return mvn.logpdf(x, mean=np.array([0.0,0.0]), cov=np.matmul(std, std.T))

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_density, D, num_samples=2000)

    # Set up plotting code
    def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z)
        ax.set_yticks([])
        ax.set_xticks([])

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t)))

        plt.cla()
        target_distribution = lambda x : np.exp(log_density(x, t))
        plot_isocontours(ax, target_distribution)

        mean, std = unpack_params(params)
        variational_contour = lambda x: mvn.pdf(x, mean, std @ std.T)
        plot_isocontours(ax, variational_contour)
        plt.draw()
        plt.pause(1.0/30.0)

    print("Optimizing variational parameters...")
    init_mean    = -1 * np.ones(D)
    init_std = np.array([0.1,0.2,0.3,0.5])
    init_var_params = np.concatenate([init_mean, init_std])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=2000, callback=callback)
