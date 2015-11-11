import GPy
import numpy as np
from GPy.kern._src.custom_kern import RationalQuadratic as ratquadkern


# ker1 = ratquadkern(1,[[0,1, 1],[1,0,1],[1,1,0]])
ker1 = ratquadkern(1,[
    [0,1,1,2,1,2,2,3],
    [1,0,1,1,2,2,3,2],
    [1,1,0,1,1,2,2,2],
    [2,1,1,0,2,1,2,1],
    [1,2,1,2,0,1,1,2],
    [2,2,2,1,1,0,1,1],
    [2,3,2,2,1,1,0,1],
    [3,2,2,1,2,1,1,0],    
])
print "Gpy version: ",GPy.__version__
#print ker1

X = np.array([ [0],[1] ],dtype = np.int32)
Y = np.array([[100],[100]])
m = GPy.models.GPRegression(X,Y, ker1)
# print m
m.optimize(messages=False)
# print m

# for x in xrange(2,8):
#     print m.predict(np.array([[x]], dtype=np.int32))
print m.predict(np.array([[2],[3],[4],[5],[6],[7]], dtype=np.int32))