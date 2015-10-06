
__author__ = 'Dmitrii'

import numpy as np
import GPy

from src import FullGP_RBF, SVGP
from src import SparseGP


def BuildModel(mode, trainX, trainY):
    if mode == "full":
        model = FullGP_RBF.FullGP_RBF(trainX, trainY)
        model.InferHypersHMC(200)
    elif mode == "svgp":
        model = SVGP.SVGP(trainX, trainY)
    elif mode == "sparsegp":
        model = SparseGP.SparseGP(trainX,trainY,10)
        

    else:
        raise Exception("imcorrect mode")
    return model


X = np.sort(np.random.rand(50,1)*12)
k = GPy.kern.RBF(1)
K = k.K(X)
K+= np.eye(50)*0.01 # add some independence (noise) to K
y = np.random.multivariate_normal(np.zeros(50), K).reshape(50,1)

#model = BuildModel("sparsegp", trainSetX, trainSetY)
model = BuildModel("sparsegp", X, y)
p_mean, p_variance = model.predict(np.random.rand(5,1)*12)
print p_mean
print p_variance