import numpy as np
import torch
import random
import argparse
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from torch.utils.data import WeightedRandomSampler
from torchsummary import summary

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pandas

from dataset import Dataset
from model.model_provider import create_model, create_criterion, create_optimizer
from utils.gradcamHeatmap import get_example_params,GradCam
from utils.misc_functions import save_class_activation_images


class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()        
        self.parser.add_argument('-data', default='../data', help='Input data folder')
        self.parser.add_argument('-saveDir', default='./save/classifier_test', help='Output data folder')
        self.parser.add_argument('-nThreads', default=2, type=int, help='Number of threads')
        self.parser.add_argument('-doAugmentaion', default=True, type=bool, help='To do augmentaion or not')
        self.parser.add_argument('-batchSize', default=512, type=int, help='Batch Size')
        self.parser.add_argument('-LR', default=1e-3, type=float, help='Learn Rate')
        self.parser.add_argument('-nEpoch', default=200, type=int, help='Number of Epochs')
        self.parser.add_argument('-dropLR', default=10, type=float, help='Drop LR')
        self.parser.add_argument('-valInterval', default=1, type=int, help='Val Interval')
        self.parser.add_argument('-loadModel', default='none', help='if not none, Load pre-trained model')
        self.parser.add_argument('-loadModelParts', default='encoder_decoder_clf', help='Which part of the parameters including the keywords(split by "_") to load,')
        self.parser.add_argument('-toTrain',  default=1, type=int, help='To train:1 or not:0')
        self.parser.add_argument('-criterionClassifier', default='CrossEntropy', help='Criterion for Classifier')
        self.parser.add_argument('-criterionAutoEncoder', default='mse', help='Criterion for AutoEncoder')
        self.parser.add_argument('-nBlocks', default=3, type=int, help='Number of NN blocks used in AutoEncoder')
        self.parser.add_argument('-codeLen', default=128*4*4, type=int, help='The flattened code length in AE hidden layer')
        self.parser.add_argument('-useWeightedSampling', default=1, type=int, help='To use class sampling weitht to combat class imbalance:1 or not:0')
        self.parser.add_argument('-toFreezeAE',  default=0, type=int, help='To freeze the pretrained AE weights for classifier:1 or not:0')
        self.parser.add_argument('-onlyAutoEncoder', default=0, type=int, help='To only train autoencoder:1, otherwise to train Classifier:0')
        self.parser.add_argument('-toGenerateHeatmap', default=1, type=int, help='To generate CAM heatmap:1 or not:0')
        
        
        self.opt = self.parser.parse_args()   

        #Create the saving directory and logging file for the configuration
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.saveDir):
            os.makedirs(self.opt.saveDir)
        file_name = os.path.join(self.opt.saveDir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        
def create_data_loaders(opt):
    """
    Create the training data loader and test data loader, can choose to use weightedSampling or not to combat data imbalance
    """
    tr_dataset, te_dataset = Dataset(opt.data, opt, 'Train'), Dataset(opt.data, opt, 'Validation')

    #Do weighted sampling to combat data imbalance
    if opt.useWeightedSampling:
        target = tr_dataset.labels.numpy()
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])
        print('class_ample_count',class_sample_count)
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, 40000)#After weighted sampling, each class will be 4000, thun 40000 in total for train
        
    train_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=opt.batchSize,
        shuffle=True if not opt.useWeightedSampling else False,  #with self-defined sampler, shuffle should be False
        drop_last=True,
        num_workers=opt.nThreads,
        pin_memory=True,
        sampler=sampler if opt.useWeightedSampling else None
    )
    test_loader = torch.utils.data.DataLoader(
        te_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=True
    )
    return train_loader, test_loader

def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    """
    To adjust the learning rate in training
    """
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def step(opt, data_loader, model, criterion, to_train=True, optimizer=None):
    """
    Used as a trining step or validation step
    """
    if to_train:
        model.train()
    else:
        model.eval()
    nIters = len(data_loader)
    loss_meter = AverageMeter()
    with tqdm(total=nIters) as t:
        predictList = []
        labelsList = []
        for i, (input_, label) in enumerate(data_loader):
            input_cuda = input_.float().cuda() if torch.cuda.is_available() else input_.float().cpu()
            label_t_cuda = label.long().cuda() if torch.cuda.is_available() else label.long().cpu()
            # ===================forward=====================
            output= model(input_cuda)
            if opt.onlyAutoEncoder:
                loss = criterion(output, input_cuda)
            else:
                loss = criterion(output, label_t_cuda)
                _, predicted = torch.max(output, 1)
                predictList.extend(predicted.cpu().numpy().tolist())
                labelsList.extend(label_t_cuda.cpu().numpy().tolist())
            # ===================backward====================                  
            if to_train:
#                if not opt.onlyAutoEncoder and opt.useClassWeight:
#                    loss = loss*torch.tensor(CLASS_WEIGHT)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_meter.update(loss.data.cpu().numpy())
            t.set_postfix(loss='{:10.8f}'.format(loss_meter.avg))
            t.update()
        accuracy = accuracy_score(labelsList, predictList)
        
    if not opt.onlyAutoEncoder:
        return loss_meter.avg, accuracy
    else:
        return loss_meter.avg

def train_net(opt, train_loader, test_loader, model, criterion, optimizer, n_epochs, val_interval, learn_rate, drop_lr):
    """
    To train the model with the input arguments, and save the plot of training and validation loss and accuracy, model parameters as well
    """
    loss_tr_list, loss_val_list, acc_tr_list, acc_val_list = [], [], [], []
    for epoch in range(1,n_epochs+1):
        print('epoch',epoch)
        if not opt.onlyAutoEncoder:
            # ===================training====================
            loss_tr_avg, acc_tr_avg = step(opt, train_loader, model, criterion, True, optimizer)
            # ===================validation====================
            with torch.no_grad():
                loss_val_avg, acc_val_avg = step(opt, test_loader, model, criterion, False, optimizer)
            acc_tr_list.append(acc_tr_avg)
            acc_val_list.append(acc_val_avg)
            # ===================paramsSaving====================
            if epoch>1 and loss_val_avg<min(loss_val_list):
                torch.save(model.state_dict(), os.path.join(opt.saveDir,'epoch{}loss{}acc{}.pth'.format(epoch,loss_val_avg,acc_val_avg))) 
        else:
            # ===================training====================
            loss_tr_avg = step(opt, train_loader, model, criterion, True, optimizer)
            # ===================training====================
            with torch.no_grad():
                loss_val_avg = step(opt, test_loader, model, criterion, False, optimizer)
            # ===================paramsSaving====================
            if epoch>1 and loss_val_avg<min(loss_val_list):
                torch.save(model.state_dict(), os.path.join(opt.saveDir,'epoch{}loss{}.pth'.format(epoch,loss_val_avg))) 
            
        loss_tr_list.append(loss_tr_avg)
        loss_val_list.append(loss_val_avg)
        # ===================lossAccuracyPlotting====================
        if not opt.onlyAutoEncoder:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
            ax1.plot(loss_tr_list, label='trainLoss')
            ax1.plot(loss_val_list, label='valLoss')
            ax1.legend(loc='upper left')

            ax2.plot(acc_tr_list, label='trainAcc')
            ax2.plot(acc_val_list, label='valAcc')
            ax2.legend(loc='upper left')
#            fig.tight_layout()
        else:
            plt.figure()
            plt.plot(loss_tr_list,label='trainLoss')
            plt.plot(loss_val_list,label='valLoss')            
            plt.legend(loc='upper left')
        plt.savefig(os.path.join(opt.saveDir, 'epoch{}.jpg'.format(epoch)))
        plt.close('all')
                              
#        adjust_learning_rate(optimizer, epoch, drop_lr, learn_rate)
        print('\n')

def imshow(img):
    """
    To show the image from a Pytorch Tensor format
    """
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def device():
    """
    To put the Tensor or model in GPU if available
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASSES = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 
           5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
TRAIN_RATIO = [0.8, 0.8, 0.5, 0.8, 0.5,
               0.8, 0.8, 0.8, 0.8, 0.5]

# CIFAR IMAGE CONSTANTS(RGB order)
MEAN = [125.3/255, 123.0/255, 113.9/255]
STD = [63.0/255, 62.1/255, 66.7/255]

# Class weight used for weighted sampling in traind dataset loader
CLASS_WEIGHT = [1./trainRation for trainRation in TRAIN_RATIO]

def main():
    # Seed all sources of randomness to 0 for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0) if torch.cuda.is_available() else torch.manual_seed(0)
    random.seed(0)
    
    # Set cudnn.benchmark True to spped up training
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled =True
        torch.backends.cudnn.benchmark = True
    opt = Opts().opt

    # Create data loaders
    train_loader, test_loader = create_data_loaders(opt)

    # Create nn
    model = create_model(opt).to(device())
    
    # Create loss criterion
    if opt.onlyAutoEncoder:
        criterion = create_criterion(opt.criterionAutoEncoder).to(device())
    else:
        criterion = create_criterion(opt.criterionClassifier).to(device())
    
    # Choose to train or to test the model
    if opt.toTrain:
        # Create optimizer
        optimizer = create_optimizer(opt, model)
        train_net(opt, train_loader, test_loader, model, criterion, optimizer, opt.nEpoch, opt.valInterval, opt.LR, opt.dropLR)
    
    # Test classifier or AutoEncoder
    if not opt.onlyAutoEncoder:
        # Testing classifier
        predictList = []
        labelsList = []
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device())
                labels = labels.to(device())
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                predictList.extend(predicted.cpu().numpy().tolist())
                labelsList.extend(labels.cpu().numpy().tolist())
        # ===================confusionMatrixGeneration====================
        confusionMatrix=confusion_matrix(labelsList, predictList)
        print('ConfusionMatrix:\n{}\n{}'.format(list(CLASSES.values()),confusionMatrix))
        np.save(os.path.join(opt.saveDir,'confusionMatrix.npy'),confusionMatrix)
        # ===================classificaitonReport: Precision, Recall and f1 score====================
        classReport = classification_report(labelsList, predictList, digits=2)
        df = pandas.DataFrame(classification_report(labelsList, predictList, digits=2, output_dict=True)).transpose()
        df.to_csv(os.path.join(opt.saveDir, 'my_csv_file.csv'))
        print(classReport)
        # ===================heatmapGeneration====================
        if opt.toGenerateHeatmap:
            model.to(torch.device('cpu'))
            #Currently 33 testing images in the folder test_images are chosen for heaimap generation 
            target_example = 33 
            for t in range(target_example):
                # Get params for heatmap generation
                (original_image, prep_img, target_class, file_name_to_export) = get_example_params(t)
                # Grad cam, choose which block and which layer in the block for heatmap generation
                grad_cam = GradCam(model, target_block=2, target_layer=9)
                # Generate cam mask
                cam = grad_cam.generate_cam(prep_img)
                # Save mask
                save_class_activation_images(opt, original_image, cam, file_name_to_export)
                print('Grad cam completed')       
    else:
        # Test AutoEncoder to plot original input images and reconstructed images
        model.eval()
        dataiter = iter(test_loader)
        # Generate images in the first 8 iteration
        for i in range(8):            
            images, labels = dataiter.next()
            print('GroundTruth: ', ' '.join('%5s' % CLASSES[labels[j].item()] for j in range(opt.batchSize)))
            # ===================showGroundTruthImages====================
            imshow(torchvision.utils.make_grid(images))
            images_ = images.to(device())
            # ===================forward=====================
            decoded_imgs = model(images_)
            # ===================showReconstructedImages====================
            imshow(torchvision.utils.make_grid(decoded_imgs.data))        
       
if __name__ == '__main__':
    main()

