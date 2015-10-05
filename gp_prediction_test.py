__author__ = 'Dmitrii'
from FullGP_RBF import FullGP_RBF
import numpy as np

X = np.random.uniform(-3.,3.,(10,2))
Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(10,1)*0.05

m = FullGP_RBF(X,Y)
print m.model

m.setParameters([2.0, 3.0, 4.0])
print m.model

x_test = np.asarray([[1,2]])
print m.predict(x_test)