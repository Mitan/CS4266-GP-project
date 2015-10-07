
__author__ = 'Lifu'

import numpy as np

import GPy
from src import DataReadingUtils


Z = DataReadingUtils.ReadData("data/Dec1_2012.csv")
matrix = np.zeros((360, 180))
for i in xrange(0,360):
    for j in xrange(0,180):
        matrix[i][j] = -100;
# print matrix

for (x, y), z in Z:
    matrix[x][y] = z[0];

interesting_cases = []

counter = 0
for i in xrange(1,359):
    for j in xrange(1,179):
        interesting = True
        X = []
        Y = []
        for m in xrange(-1,2):
            for n in xrange(-1,2):
                if matrix[i+m][j+n] == -100:
                    interesting = False
                else:
                    if not (m==0 and n==0):
                        X.append([i+m,j+n])
                        Y.append([matrix[i+m][j+n]])
        if interesting:
            print "interesting"
            X = np.array(X)
            Y = np.array(Y)
            # print X.shape
            # print Y

            m = GPy.models.GPRegression(X,Y)
            m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
            m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
            m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
            hmc = GPy.inference.mcmc.HMC(m,stepsize=5e-2)
            s = hmc.sample(num_samples=200)
            samples = s[60:]
            m.kern.variance[:] = samples[:,0].mean()
            m.kern.lengthscale[:] = samples[:,1].mean()
            m.likelihood.variance[:] = samples[:,2].mean()

            
            p_mean, p_variance = m.predict(np.array([[i,j]]))
            error = (matrix[i][j] - p_mean[0]) ** 2
            interesting_cases.append(((i,j), matrix[i][j], p_mean[0][0], error[0]))

            print "finish one, error: %f" %error 

            counter += 1


average_errors = 0
for idx, ((x, y), value, prediction, error) in enumerate(interesting_cases):
    print interesting_cases[idx]
    average_errors += error

average_errors /= len(interesting_cases)
print "num of case: %d" % len(interesting_cases)
print "error: %f" % average_errors
            