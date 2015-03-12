from numpy import *
import scipy.io
import pylab
import random

d = scipy.io.loadmat('mp_3-5_data.mat') # corresponding MAT file
data = d['Xtrain']    # Xtest for test data
labels = d['Ytrain']  # Ytest for test labels

dataLen = len(data)
trainingLen = 2*dataLen/3
indexes = range(0, dataLen)
random.shuffle(indexes)
"""spliting the dataset into training and validation (!done!)"""
#trainingDict = {'Xtrain':[data[i] for i in indexes[:trainingLen]], 'Ytrain': [labels[i] for i in indexes[:trainingLen]]}
#validationDict = {'Xtrain':[data[i] for i in indexes[trainingLen:]], 'Ytrain': [labels[i] for i in indexes[trainingLen:]]}

#scipy.io.savemat('training_4-9.mat', trainingDict)
#scipy.io.savemat('validation_4-9.mat', validationDict)

