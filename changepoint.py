import collections
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('tkagg')

count_data = np.loadtxt('txtdata.csv')
n_count_data = len(count_data)

with pm.Model() as model:
	alpha = 1.0 / count_data.mean()

	lambda_1 = pm.Exponential('lambda_1', alpha)
	lambda_2 = pm.Exponential('lambda_2', alpha)

	tau = pm.DiscreteUniform('tau', lower=0, upper=n_count_data - 1)

with model:
	idx = np.arange(n_count_data)
	lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:
	observation = pm.Poisson('obs', lambda_, observed=count_data)

with model:
	step = pm.Metropolis()
	trace = pm.sample(10000, tune=5000, step=step)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']
tau_count = collections.Counter(tau_samples)

plt.subplot(311)
plt.hist(lambda_1_samples, bins=30)
plt.subplot(312)
plt.hist(lambda_2_samples, bins=30)
plt.subplot(313)
plt.hist(tau_samples, bins=30)

plt.show()
