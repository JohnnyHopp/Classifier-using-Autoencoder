from .classifier_model import AutoEncoderClassifier
import torch.nn as nn
import torch

def create_criterion(criterion):
    """
    Create the loss criterion, currenly only MSE and CrossEntropy loss are available
    """
    if criterion == 'mse':
        return nn.MSELoss()
    elif criterion == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('Criterion ' + criterion + ' not supported')

def keywordInStr(keywordList, string):
    """
    Decision if there is any keyword from the keywordList in the string, return True or False
    """
    return any(keyword in string for keyword in keywordList)

def create_model(opt):
    """
    Create the mdoel, if the "opt.loadModel" is not "none", load the pretrained parameter according to the keywords in "opt.loadModelParts"
    """
    model = AutoEncoderClassifier(opt)
    if not opt.loadModel=='none':
#        model = torch.load(opt.loadModel, map_location=lambda storage, loc: storage)
        pretrained_dict = torch.load(opt.loadModel)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and keywordInStr(opt.loadModelParts.split('_'), k)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
#        model.load_state_dict(torch.load(opt.loadModel))
        print('Loaded model from '+opt.loadModel)
    
    #To freeze the Auto Encoder weights as a feature extracture for classifier
    if opt.toFreezeAE:
        model.freezeEncoder()
    
    return model

def create_optimizer(opt, model):
    """
    Create the optimizer, currently only use Adam
    """
    return torch.optim.Adam(model.parameters(), opt.LR)
#    return torch.optim.SGD(model.parameters(), opt.LR, momentum=0.9)