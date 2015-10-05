#configure plotting
# %matplotlib inline
# %config InlineBack/end.figure_format = 'svg'
# import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)

import numpy as np
from matplotlib import pyplot as plt
import GPy
from IPython.display import display
from scipy import stats
from pylab import *
from datetime import datetime

X=[]
Y=[]

input_file = open("Dec1_2012.csv","r")
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

input_file.close()
avg_time = 0
for i in range(10):
	# use full GP and auto optimise
	start = datetime.now()

	m = GPy.models.GPRegression(X,Y)
#	print m
	m.optimize('bfgs')

	time_used = datetime.now() - start
	avg_time += time_used.total_seconds()/10 #in seconds
	print str(time_used.total_seconds())
print 'bfgs optimizer: ',avg_time
#print m
avg_time = 0
#optimize using scg optmizer (Scaled Conjugate Gradient Descent)
for i in range(10):

	start = datetime.now()

	m = GPy.models.GPRegression(X,Y)
#	print m
	m.optimize('scg')

	time_used = datetime.now() - start
	print str(time_used.total_seconds())
	avg_time +=time_used.total_seconds()/10.0

	#print m
print 'scaled conjugate GD: ',avg_time
#
# _=m.plot()
# plt.show();



# use sparse GP and auto optimise
# note that in order for sparse GP to perform well we need to set some inducing point to help optimise it
# start = datetime.now()

# m = GPy.models.SparseGPRegression(X,Y)
# m.optimize('bfgs')

# time_used = datetime.now() - start
# print str(time_used)

# print m
# _=m.plot()
# plt.show();



