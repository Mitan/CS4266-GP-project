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
# by default the size of generated test data is 1/3 of the whole sample
def GenerateTestAndTrainData(Z, test_size=0):
    if test_size == 0:
        test_size = len(Z) / 3
    # TODO change to random seed
    #seed(45)
    testSet = sample(Z, test_size)
    trainSet = []
    for point in Z:
        if not point in testSet:
            trainSet.append(point)
    return trainSet, testSet

