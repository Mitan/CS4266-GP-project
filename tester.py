# linear-fit-1.py
# Example program for linear least square fitting

import numpy as np 
import numpy.linalg as la

# Read data in input file, which is in CSV format.
file = open("co2-data.txt")
data = np.genfromtxt(file, delimiter=",")
file.close()
print "data =\n", data

# Arrange data into D matrix and v vector.
n = len(data[:,0])
D = np.matrix(np.empty([n,10]))
for i in range(3):
 D[:,i] = np.matrix(np.square(data[:,i+1])).T
D[:,3]= np.matrix(data[:,1]*data[:,2]).T
D[:,4]= np.matrix(data[:,2]*data[:,3]).T
D[:,5]= np.matrix(data[:,1]*data[:,3]).T
D[:,6:9] = data[:,1:4]
D[:,9] = np.ones([n,1])
v = np.matrix(data[:,0]).T

# Print to verify that data is arranged correctly.
print "D =\n", D
print "v =\n", v

# Solve for least square solution
a,e,r,s = la.lstsq(D, v)
print "a =\n", a

# Compute fitting error
norm = la.norm(D * a - v)
err = norm * norm
print "err =", err
print "lstsq e =", e

# Write fitting result into output file
result = np.matrix(np.empty([n,10]))
result[:,0] = D * a
result[:,1:10] = D[:,0:9]
file = open("co2-result-q1.txt", "w")
np.savetxt(file, result, "%f", ",")
file.close()
