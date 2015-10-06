import GPy
import climin
import numpy as np

__author__ = 'Dmitrii'

class SVGP():
    """
    SVGP model is said to be very quick
    No need to infer the parameters
    """

    def __init__(self, X, Y):
        #Z and batchsize can be changed
        Z = np.random.rand(20,1)
        batchsize = 10
        m = GPy.core.SVGP(X, Y, Z, GPy.kern.RBF(1) + GPy.kern.White(1), GPy.likelihoods.Gaussian(), batchsize=batchsize)
        m.kern.white.variance = 1e-5
        m.kern.white.fix()
        opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)
        opt.minimize_until(self.callback)
        self.model = m

    def callback(self, i):
        #print m.log_likelihood(), "\r",
        #Stop after 5000 iterations
        if i['n_iter'] > 5000:
            return True
        return False


    def predict(self, x_new):
        return self.model.predict(x_new)
