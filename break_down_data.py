import numpy as np

from random import seed, sample

X=[]
Y=[]

input_file = open("infer_parameters/Dec1_2012.csv","r")
counter = 0
for line in input_file:
	if not line:
		break
	contents = line.split(",")
	for idx, value in enumerate(contents):
		if (contents[idx].strip() != "NaN"):
			# @naheed remember to check whether we need to flip longitude and lattitude 
			X.append([idx, counter])
			Y.append([float(contents[idx].strip())])
	counter += 1


input_file.close()

Z = zip(X, Y)

seed(45)
testSet = sample(Z, len(Z)/3)
trainSet = []
for point in Z:
	if not point in testSet:
		trainSet.append(point)

testSetX = np.array([data[0] for data in testSet]);
testSetY = np.array([data[1] for data in testSet]);
trainSetX = np.array([data[0] for data in trainSet]);
trainSetY = np.array([data[1] for data in trainSet]);



# code for building model comes here


for test_x in testSetX:
	# code for prediction and compare with ground truth comes here
	pass











