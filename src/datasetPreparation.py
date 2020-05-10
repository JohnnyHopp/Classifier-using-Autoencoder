# -*- coding: utf-8 -*-
"""
To prepare the dataset for classification
"""
import numpy as np
from numpy import random


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def countClass(dataDict):
    """
    Count the data numbers in each class
    """
    for i,(key,val) in enumerate(dataDict.items()):
        print('label {} count: {}'.format(key, len(val)))
    print('\n')

# Seed all sources of randomness to 0 for reproducibility
random.seed(0)
    
CLASSES = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 
           5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
#The ratio used to choose from data batch for training data
TRAIN_RATIO = [0.8, 0.8, 0.5, 0.8, 0.5,
               0.8, 0.8, 0.8, 0.8, 0.5]
#The ratio used to choose from data batch for validation data
VALIDATION_RATIO = [0.2]*10

DIRECT = './data/cifar-10-batches-py/'
TRAIN_DATAFILES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TEST_DATAFILES = ['test_batch']
dataTrDict,dataValDict = {},{}

#Initialize the key and sort the key in dataTrDict,dataValDict as CLASSES
for key in CLASSES:
    dataTrDict[key],dataValDict[key]=[],[] 
dataList,labelList = [],[]
dataTeList,labelTeList = [],[]

for trainFile in TRAIN_DATAFILES:
    dataD = unpickle(DIRECT+trainFile)
    dataList.append(dataD[b'data'])
    labelList.append(dataD[b'labels'])

for testFile in TEST_DATAFILES:
    dataD = unpickle(DIRECT+testFile)
    dataTeList.append(dataD[b'data'])
    labelTeList.append(dataD[b'labels'])
    
dataList = np.concatenate(dataList, axis=0)
labelList = np.concatenate(labelList, axis=0)

#Preprocessing and normalize the data
dataList = dataList.reshape(-1,3,32,32)
#dataList = dataList/255.
#dataList = dataList.transpose(0,2,3,1)
#dataList = (dataList - MEAN) / STD
#dataList = dataList.transpose(0,3,1,2)

#print('dataList.shape',dataList.shape,dataList[0])
print(len(dataList),len(labelList))

#random shuffle the dataset
indList = np.arange(len(labelList))
random.shuffle(indList)
dataList,labelList = dataList[indList],labelList[indList]

for ind,label in enumerate(labelList):
    dataTrDict[label].append(ind)

#Separate the dataset into trainDataset(50% for minority class and 80 for majority classes) and validation dataset(20%)
for i,indTrList in enumerate(dataTrDict.values()):
    dataTrDict[i],dataValDict[i] = indTrList[:int(TRAIN_RATIO[i]*len(indTrList))],indTrList[-int(VALIDATION_RATIO[i]*len(indTrList)):]

#Count the amounts for each class
countClass(dataTrDict)
countClass(dataValDict)

dataTrList,labelTrList = [],[]
dataValList,labelValList = [],[]

for i,indTrList in enumerate(dataTrDict.values()):
    dataTrList.append(dataList[indTrList])
    labelTrList.append(labelList[indTrList])
for i,indValList in enumerate(dataValDict.values()):
    dataValList.append(dataList[indValList])
    labelValList.append(labelList[indValList])

dataTrList,labelTrList = np.concatenate(dataTrList, axis=0),np.concatenate(labelTrList, axis=0)
dataValList,labelValList = np.concatenate(dataValList, axis=0),np.concatenate(labelValList, axis=0)

#random shuffle the train and validation dataset
indTrList,indValList = np.arange(len(dataTrList)),np.arange(len(dataValList))
random.shuffle(indTrList)
random.shuffle(indValList)
dataTrList,labelTrList = dataTrList[indTrList],labelTrList[indTrList]
dataValList,labelValList = dataValList[indValList],labelValList[indValList]


dataTr,labelTr = np.asarray(dataTrList),np.asarray(labelTrList)
dataVal,labelVal = np.asarray(dataValList),np.asarray(labelValList)

#Save the prepared data
print(dataTr.shape,labelTr.shape,dataVal.shape,labelVal.shape)
np.save('./data/dataTrain.npy',dataTr)
np.save('./data/labelTrain.npy',labelTr)
np.save('./data/dataValidation.npy',dataVal)
np.save('./data/labelValidation.npy',labelVal)


    
