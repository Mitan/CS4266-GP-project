import numpy as np
from random import seed, sample


def ReadData(intput_file_name):
    X = []
    Y = []
    input_file = open(intput_file_name, "r")
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
    return Z


# return trainSet and testSet
def GenerateTestAndTrainData(Z):
    seed(45)
    testSet = sample(Z, len(Z) / 3)
    trainSet = []
    for point in Z:
        if not point in testSet:
            trainSet.append(point)
    testSetX = np.array([data[0] for data in testSet]);
    testSetY = np.array([data[1] for data in testSet]);
    trainSetX = np.array([data[0] for data in trainSet]);
    trainSetY = np.array([data[1] for data in trainSet]);
    return trainSet, testSet

