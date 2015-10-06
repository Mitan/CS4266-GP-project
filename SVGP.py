import GPy
import climin
import numpy as np

__author__ = 'Dmitrii'


class SVGP():
    """
    SVGP model is said to be very quick
    No need to infer the parameters
    """

    def __init__(self, X, Y, batch_size=20, inducing_input_size=60, number_of_iterations=500):
        # Z and batchsize can be changed
        # set correct shape for inducing inputs
        # TODO
        # check if this is correct
        self.iterations = number_of_iterations
        input_dim = (X[0].shape)[0]
        Z = np.random.rand(inducing_input_size, input_dim)
        m = GPy.core.SVGP(X, Y, Z, GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim), GPy.likelihoods.Gaussian(),
                          batchsize=batch_size)
        m.kern.white.variance = 1e-5
        m.kern.white.fix()
        opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)
        opt.minimize_until(self.callback)
        self.model = m

    def callback(self, i):
        # print m.log_likelihood(), "\r",
        # Stop after given number of iterations
        if i['n_iter'] > self.iterations:
            return True
        return False

    def predict(self, x_new):
        return self.model.predict(x_new)
