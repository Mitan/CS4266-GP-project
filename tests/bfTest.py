
__author__ = 'Lifu'

import numpy as np

from src import SparseGP
from src import DataReadingUtils


Z = DataReadingUtils.ReadData("data/Dec1_2012.csv")
matrix = np.zeros((360, 180))
for i in xrange(0,360):
    for j in xrange(0,180):
        matrix[i][j] = -100;
# print matrix

for (x, y), z in Z:
    matrix[x][y] = z[0];
        

interesting_cases = []

for i in xrange(1,359):
    for j in xrange(1,179):
        interesting = True
        surrounding_sum = 0
        for m in xrange(-1,2):
            for n in xrange(-1,2):
                if matrix[i+m][j+n] == -100:
                    interesting = False
                else:
                    surrounding_sum += matrix[i+m][j+n]
        if interesting:
            surrounding_sum -= matrix[i][j]
            surrounding_sum /= 8
            error = (surrounding_sum-matrix[i][j])**2
            interesting_cases.append(((i,j), matrix[i][j], surrounding_sum, error))

average_errors = 0
for idx, ((x, y), value, prediction, error) in enumerate(interesting_cases):
    print interesting_cases[idx]
    average_errors += error

average_errors /= len(interesting_cases)
print "num of case: %d" % len(interesting_cases)
print "error: %f" % average_errors
            