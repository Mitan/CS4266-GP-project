import GPy
import climin
from SVGP import SVGP
__author__ = 'Dmitrii'
from FullGP_RBF import FullGP_RBF
import numpy as np
"""
X = np.random.uniform(-3.,3.,(10,2))
Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(10,1)*0.05

m = FullGP_RBF(X,Y)
print m.model

m.setParameters([2.0, 3.0, 4.0])
print m.model

x_test = np.asarray([[1,2]])
print m.predict(x_test)
"""

N=5000
X = np.random.rand(N)[:, None]

print X[0]
print X[0].shape
Y1 = np.sin(6*X) + 0.1*np.random.randn(N,1)
Y2 = np.sin(3*X) + 0.1*np.random.randn(N,1)
Y = np.hstack((Y1, Y2))
X_test = np.asarray([[0.333]])

Z = np.random.rand(20,1)
batchsize = 10

m1 = SVGP(X, Y)

print m1.predict(X_test)

