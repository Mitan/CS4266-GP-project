
__author__ = 'Lifu'

import numpy as np
import GPy
from matplotlib import pyplot as plt

from src import SparseGP
from src import DataReadingUtils


def BuildModel(mode, trainX, trainY):
    model = SparseGP.SparseGP(trainX,trainY,50)

    return model

def sparse_GP_regression_2D(X, Y, num_samples=400, num_inducing=200, max_iters=100, optimize=True, plot=True, nan=False):
    """Run a 2D example of a sparse GP regression."""
    np.random.seed(1234)
    # X = np.random.uniform(-3., 3., (num_samples, 2))
    # Y = np.sin(X[:, 0:1]) * np.sin(X[:, 1:2]) + np.random.randn(num_samples, 1) * 0.05
    if nan:
        inan = np.random.binomial(1,.2,size=Y.shape)
        Y[inan] = np.nan

    # construct kernel
    rbf = GPy.kern.RBF(2)

    # create simple GP Model
    m = GPy.models.SparseGPRegression(X, Y, kernel=rbf, num_inducing=num_inducing)

    # contrain all parameters to be positive (but not inducing inputs)
    m['.*len'] = 2.

    m.checkgrad()

    # m.constrain_fixed('.*variance', 1.)
    # m.inducing_inputs.constrain_fixed()
    # m.Gaussian_noise.variance.constrain_bounded(1e-3, 1e-1)

    # optimize
    if optimize:
        m.optimize('tnc', messages=1, max_iters=max_iters)

    # plot
    if plot:
        print m.plot
        m.plot()
        plt.show()

    print m
    return m


# 1854oct-sst.csv
# 1991Aug-sst.csv
# 1999Feb-sst.csv
# 2007Apr-sst.csv
# Dec1_2012.csv
            
Z = DataReadingUtils.ReadData("data/Dec1_2012.csv")
trainSet, testSet = DataReadingUtils.GenerateTestAndTrainData(Z)
print "trainset size: %d, testSet size: %d" %((len(trainSet)), (len(testSet)))

trainSetX = np.array([mtuple[0]for mtuple in trainSet])
trainSetY = np.array([mtuple[1]for mtuple in trainSet])

model = sparse_GP_regression_2D(trainSetX, trainSetY)

testSetX = np.array([mtuple[0]for mtuple in testSet])
testSetY = np.array([mtuple[1]for mtuple in testSet])


p_mean, p_variance = model.predict(testSetX)
result = zip(p_mean, testSetY, p_variance)

error = ((testSetY - p_mean) ** 2).mean()
print "error:%f"%error

