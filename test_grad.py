import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad


def relu(X):
	return np.maximum(X, 0)
def nn(W):
	return relu(np.matmul(W, X))
def cost(W):
	return np.sum((nn(W) - y)**2)


grad_cost = grad(cost)
W = np.random.randn(1,2)
X = np.array([[0.0,1.0],[1.0,1.0],[0.0,0.0],[1.0,0.0]]).T
y = np.array([1,0,0,1])
print(grad_cost(W))
lr = 1e-2
print(cost(W))
for i in range(100):
	W = W - lr * grad_cost(W)
	print(cost(W))
print('final:',W)
print(nn(W))
