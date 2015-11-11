from .kern import Kern
import numpy as np
from ...core.parameterization.param import Param

class RationalQuadratic(Kern):
    """
    Testing Custom Kernel 
    """
    def __init__(self,input_dim,nodes_matrix,variance=1.,lengthscale=1.,power=1.,active_dims=None, name='quad'):
        super(RationalQuadratic, self).__init__(input_dim, active_dims, name)
        assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.power = Param('power', power)
        self.nodes_matrix = Param('nodes_matrix', nodes_matrix)
        self.link_parameters(self.variance, self.lengthscale, self.power)
       
        
    def K(self,X,X2):
        print X
        print X2 
        X = np.array(X, dtype=np.int32)
        if X2 is None: X2 = X
        # print self.nodes_matrix.shape
        # print self.nodes_matrix        
        # covariance_matrix = np.zeros((self.nodes_matrix.shape[0], self.nodes_matrix.shape[1]))
#         for i, row in enumerate(self.nodes_matrix):
#             for j, num_nodes in enumerate(row):
#                 if i == j:
#                     # covariance_matrix[i][j] = (self.variance**2)
#                     covariance_matrix[i][j] = 0
#                 else:
#                     covariance_matrix[i][j] = (self.variance**2)*np.exp([-(self.nodes_matrix[i][j]**2)/(2*self.lengthscale**2)])[0]
#         print covariance_matrix
        
        covariance_matrix = np.zeros((X.shape[0], X2.shape[0]))
        for i in xrange(0,X.shape[0]):
            for j in xrange(0,X2.shape[0]):
                if X[i][0] == X2[j][0]:
                    # covariance_matrix[i][j] = (self.variance**2)
                    covariance_matrix[i][j] = 0
                else:
                    covariance_matrix[i][j] = (self.variance**2)*np.exp([-(self.nodes_matrix[X[i][0]][X2[j][0]]**2)/(2*self.lengthscale**2)])[0]
        print covariance_matrix
        return covariance_matrix
        
    def Kdiag(self,X):
        return np.zeros(X.shape[0])
        # return self.variance**2 * np.ones(X.shape[0])
        
    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X
        # dist2 = np.square((X-X2.T)/self.lengthscale)
        dist2 = self.K(X)     

        dvar = (1 + dist2/2.)**(-self.power)
        dl = self.power * self.variance * dist2 * self.lengthscale**(-3) * (1 + dist2/2./self.power)**(-self.power-1)
        dp = - self.variance * np.log(1 + dist2/2.) * (1 + dist2/2.)**(-self.power)

        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)
        self.power.gradient = np.sum(dp*dL_dK)
    