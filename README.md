# Classifier-using-Autoencoder
 An investigation on Classifier using AutoEncoder

This repository contains a onvolutional neural network Classifier with AutoEncoder implemented in PyTorch.
Note: The code in this repository was tested with torch version 1.4.0

Possible requirements:
torch == 1.4.0
torchvision >= 0.5.0
numpy >= 1.16.5
matplotlib >= 3.1.1
PIL >= 6.2.0

1.Data Preparation
You have to download the CIFAR-10 dataset in pytorch versjion first to DIRECT = './data/cifar-10-batches-py/' and unpack, then use datasetPreparation.py to prepare the training and validation dataset as numpy format(dataTrain.npy, labelTrain.npy, dataValidation.npy, labelValidation.npy) , the default saving directory is './data/'.

2.Implementation
main.py is the main implementation code to run. Following are the arguments:

"-data" : Input data folder
"-saveDir" : Output data folder
"-nThreads" : Number of threads
"-doAugmentaion" : To do augmentaion or not
"-batchSize" : Batch Size
"-LR" : Learn Rate
"-nEpoch" : Number of Epochs
"-dropLR" : Drop LR, after every dropLR num of epoch to drop learning rate
"-valInterval" : Validation Interval
"-loadModel" : if not "none", Load pre-trained model
"-loadModelParts" : Which part of the parameters including the keywords(split by "_") to load, default='encoder_decoder_clf', means to load all 3 model parts params of encoder, decoder and clf(classifier)
"-toTrain" : To train:1 or not:0
"-criterionClassifier" : Criterion for Classifier
"-criterionAutoEncoder" : Criterion for AutoEncoder
"-nBlocks" : Number of NN blocks used in AutoEncoder
"-codeLen" : The flattened code length in AE hidden layer
"-useWeightedSampling" : To use class sampling weitht to combat class imbalance:1 or not:0
"-toFreezeAE" : To freeze the pretrained AE weights for classifier:1 or not:0
"-onlyAutoEncoder" : To only use autoencoder:1, otherwise to use Classifier:0
"-toGenerateHeatmap" : To generate CAM heatmap:1 or not:0
