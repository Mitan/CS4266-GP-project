
__author__ = 'Dmitrii'

import numpy as np

import DataReadingUtils
import FullGP_RBF
import SVGP
import SparseGP
def BuildModel(mode, trainX, trainY):
    if mode == "full":
        model = FullGP_RBF.FullGP_RBF(trainX, trainY)
        model.InferHypersHMC(200)
    elif mode == "svgp":
        model = SVGP.SVGP(trainX, trainY)
    elif mode == "sparsegp":
        model = SparseGP.SparseGP(trainX,trainY,10)
        

    else:
        raise Exception("imcorrect mode")
    return model


if __name__ == "__main__":
    input_file_name = "./infer_parameters/Dec1_2012.csv"
    
    all_data = DataReadingUtils.ReadData(input_file_name)
    print "Finished reading data"
    trainSet, testSet = DataReadingUtils.GenerateTestAndTrainData(all_data)

    testSetX = np.array([data[0] for data in testSet]);
    testSetY = np.array([data[1] for data in testSet]);
    trainSetX = np.array([data[0] for data in trainSet]);
    trainSetY = np.array([data[1] for data in trainSet]);
    print "Finished dividing data into test and train"

    model = BuildModel("sparsegp", testSetX, testSetY)
    
    print "Finished building model"
    p_mean, p_variance = model.predict(testSetX)
   
    print "Finished prediction"
    print p_mean
    print testSetY
    print p_variance
    error = ((testSetY - p_mean) ** 2).mean()


    print "error is " + str(error)
