import GPy
import climin
import numpy as np

__author__ = 'Dmitrii'

class SVGP():
    """
    SVGP model is said to be very quick
    No need to infer the parameters
    """

    def __init__(self, X, Y, batch_size = 10, inducing_input_size = 20):
        #Z and batchsize can be changed
        #set correct shape for inducing inputs
        # TODO
        # check if this is correct
        Z_shape = (X[0].shape)[0]
        Z = np.random.rand(inducing_input_size,Z_shape)
        m = GPy.core.SVGP(X, Y, Z, GPy.kern.RBF(1) + GPy.kern.White(1), GPy.likelihoods.Gaussian(), batchsize=batch_size)
        m.kern.white.variance = 1e-5
        m.kern.white.fix()
        opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)
        opt.minimize_until(self.callback)
        self.model = m

    def callback(self, i):
        #print m.log_likelihood(), "\r",
        #Stop after 5000 iterations
        if i['n_iter'] > 50:
            return True
        return False


    def predict(self, x_new):
        return self.model.predict(x_new)
