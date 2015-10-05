__author__ = 'Dmitrii'
import GPy


class FullGP_RBF():
    """
    full GP prediction with rbf kernel
    we can't use optimise method of GPRegression because it work very slowly for input size = 15k
    so we can just create the model, set the parameters infered from HMC and make a prediction
    """

    def __init__(self, X, Y):
        self.model = GPy.models.GPRegression(X, Y)

    def setParameters(self, params):
        # params is a list [rbf.variance, rbf.lengthscale, Gaussian_noise.variance ]
        self.model[:] = params

    def predict(self, x_new):
        return self.model.predict(x_new)
