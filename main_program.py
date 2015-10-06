__author__ = 'Dmitrii'

import numpy as np

import DataReadingUtils

input_file_name = "./infer_parameters/Dec1_2012.csv"

all_data = DataReadingUtils.ReadData(input_file_name)
print "Finished reading data"
trainSet, testSet = DataReadingUtils.GenerateTestAndTrainData(all_data)

testSetX = np.array([data[0] for data in testSet]);
testSetY = np.array([data[1] for data in testSet]);
trainSetX = np.array([data[0] for data in trainSet]);
trainSetY = np.array([data[1] for data in trainSet]);
print "Finished dividing data into test and train"
