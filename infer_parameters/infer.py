#configure plotting
# %matplotlib inline
# %config InlineBack/end.figure_format = 'svg'
# import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)

import numpy as np
from matplotlib import pyplot as plt
import GPy
from IPython.display import display
from scipy import stats
from pylab import *
from datetime import datetime

X=[]
Y=[]

input_file = open("Dec1_2012.csv","r")
counter = 0
for line in input_file:
	if (len(X) > 300):
		break
	if not line:
		break
	contents = line.split(",")
	for idx, value in enumerate(contents):
		if (contents[idx].strip() != "NaN"):
			# print contents[idx]
			# break
			X.append([idx, counter])
			Y.append([float(contents[idx].strip())])
	counter += 1

X = np.array(X)
Y = np.array(Y)

input_file.close()


# use full GP and auto optimise
# start = datetime.now()

# m = GPy.models.GPRegression(X,Y)
# m.optimize('bfgs')

# time_used = datetime.now() - start
# print str(time_used)

# print m
# _=m.plot()
# plt.show();



# use sparse GP and auto optimise
# note that in order for sparse GP to perform well we need to set some inducing point to help optimise it
# start = datetime.now()

# m = GPy.models.SparseGPRegression(X,Y)
# m.optimize('bfgs')

# time_used = datetime.now() - start
# print str(time_used)

# print m
# _=m.plot()
# plt.show();




# use HMC to optimise parameters
start = datetime.now()

m = GPy.models.GPRegression(X,Y)
# m = GPy.models.SparseGPRegression(X,Y)
m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
hmc = GPy.inference.mcmc.HMC(m,stepsize=5e-2)
s = hmc.sample(num_samples=200)

labels = ['kern variance', 'kern lengthscale','noise variance']
samples = s[60:] # cut out the burn-in period

# xmin = samples.min()
# xmax = samples.max()
# xs = np.linspace(xmin,xmax,100)
# for i in xrange(samples.shape[1]):
# 	kernel = stats.gaussian_kde(samples[:,i])
# 	plot(xs,kernel(xs),label=labels[i])
# _ = legend()

time_used = datetime.now() - start
print str(time_used)

m.kern.variance[:] = samples[:,0].mean()
m.kern.lengthscale[:] = samples[:,1].mean()
m.likelihood.variance[:] = samples[:,2].mean()
print m
_=m.plot()
plt.show();
