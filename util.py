import numpy as np
import torch
import matplotlib.pyplot as plt
def plot_contours(ax, func, xlimits=[-10,10], ylimits=[-10,10], numticks=101):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    coor = torch.Tensor(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    zs = func(coor).data.numpy()
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z)
    ax.set_xticks([])
    ax.set_yticks([])

