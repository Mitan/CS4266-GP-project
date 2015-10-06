"""
from SVGP import SVGP
from FullGP_RBF import FullGP_RBF


# X = np.random.uniform(-3.,3.,(10,2))
# Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(10,1)*0.05
#
# m = FullGP_RBF(X,Y)
# print m.model
#
# # m.optimize("bfgs");
# # m.setParameters([2.0, 3.0, 4.0])
# print m.model
#
# x_test = np.asarray([[1,2]])
# print m.predict(x_test)


# N=5000
# X = np.random.rand(N)[:, None]
#
# Y1 = np.sin(6*X) + 0.1*np.random.randn(N,1)
# Y2 = np.sin(3*X) + 0.1*np.random.randn(N,1)
# Y = np.hstack((Y1, Y2))
# X_test = np.asarray([[0.333]])


X=[]
Y=[]

input_file = open("infer_parameters/Dec1_2012.csv","r")
counter = 0
for line in input_file:
	if (len(X) > 2000):
		break
	if not line:
		break
	contents = line.split(",")
	for idx, value in enumerate(contents):
		if (contents[idx].strip() != "NaN"):
			# print contents[idx]
			# break
			X.append([idx, counter])
			Y.append([float(contents[idx].strip())])
	counter += 1

X = np.array(X)
Y = np.array(Y)
print str(X.shape)

input_file.close()


m1 = SVGP(X, Y)
print m1.model


X_test = np.array([[1,1]]);

print m1.predict(X_test)
"""
