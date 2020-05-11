# Classifier-using-Autoencoder
 An investigation on Classifier using AutoEncoder

This repository contains a convolutional neural network Classifier with AutoEncoder implemented in PyTorch.
>Note: The code in this repository was tested with torch version 1.4.0

## Possible requirements:
torch == 1.4.0
torchvision >= 0.5.0
numpy >= 1.16.5
matplotlib >= 3.1.1
PIL >= 6.2.0

## 1.Data Preparation
You have to download the CIFAR-10 dataset in pytorch version first to DIRECT = './data/cifar-10-batches-py/' and unpack, then use datasetPreparation.py to prepare the training and validation dataset as numpy format(dataTrain.npy, labelTrain.npy, dataValidation.npy, labelValidation.npy) , the default saving directory is './data/'.

## 2.Implementation
main.py is the main implementation code to run. Following are the arguments:

>"-data" : Input data folder <br>
"-saveDir" : Output data folder <br>
"-nThreads" : Number of threads<br>
"-doAugmentaion" : To do augmentaion or not<br>
"-batchSize" : Batch Size<br>
"-LR" : Learn Rate<br>
"-nEpoch" : Number of Epochs<br>
"-dropLR" : Drop LR, after every dropLR num of epoch to drop learning rate<br>
"-valInterval" : Validation Interval<br>
"-loadModel" : if not "none", Load pre-trained model<br>
"-loadModelParts" : Which part of the parameters including the keywords(split by "_") to load, default='encoder_decoder_clf', means to load all 3 model parts params of encoder, decoder and clf(classifier)<br>
"-toTrain" : To train:1 or not:0<br>
"-criterionClassifier" : Criterion for Classifier<br>
"-criterionAutoEncoder" : Criterion for AutoEncoder<br>
"-nBlocks" : Number of NN blocks used in AutoEncoder<br>
"-codeLen" : The flattened code length in AE hidden layer<br>
"-useWeightedSampling" : To use class sampling weitht to combat class imbalance:1 or not:0<br>
"-toFreezeAE" : To freeze the pretrained AE weights for classifier:1 or not:0<br>
"-onlyAutoEncoder" : To only use autoencoder:1, otherwise to use Classifier:0<br>
"-toGenerateHeatmap" : To generate CAM heatmap:1 or not:0<br>
