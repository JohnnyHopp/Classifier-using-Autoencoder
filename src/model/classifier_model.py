import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        self.block1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ELU(), nn.BatchNorm2d(32), 
                nn.Conv2d(32, 32, 4, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(32) 
                )#->32x16x16
        self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ELU(), nn.BatchNorm2d(64), 
                nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(64)
                )#->64x8x8
        self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(128), 
                nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(128)
                )#->128x4x4
        self.block4 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(256), 
                nn.Conv2d(256, 256, 4, stride=2, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(256)
                )#->256x2x2                    
        self.block5 = nn.Sequential(
                nn.Conv2d(256, 256, 4, stride=2, padding=1), nn.ELU(), nn.Dropout2d(p=0.3), nn.BatchNorm2d(256)
                )#->256x1x1

#        self.encoder = nn.Sequential(self.block1, self.block2, self.block3, self.block4, self.block5)
        blocks = [self.block1, self.block2, self.block3,self.block4, self.block5]
        self.encoder = nn.Sequential(*blocks[:opt.nBlocks])
        
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        # Input size: [batch, 512, 1, 1]
        self.block1 = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), nn.ELU(), nn.Dropout2d(p=0.3), nn.BatchNorm2d(256)
                )#->256x2x2
        self.block2 = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(256), 
                nn.Conv2d(256, 128, 3, stride=1, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(128)
                )
        self.block3 = nn.Sequential(
                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(128), 
                nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.ELU(), nn.Dropout2d(p=0.2), nn.BatchNorm2d(64)
                )
        self.block4 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(64), 
                nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.ELU(), nn.BatchNorm2d(32),
                )
        self.block5 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(32), 
                nn.Conv2d(32, 3, 3, stride=1, padding=1), nn.Sigmoid()
                )
#        self.decoder = nn.Sequential(self.block1, self.block2, self.block3, self.block4, self.block5)
        blocks = [self.block1, self.block2, self.block3,self.block4, self.block5]
        self.decoder = nn.Sequential(*blocks[-opt.nBlocks:])

        
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()        
        self.classifier = nn.Sequential(
                nn.Linear(opt.codeLen, 32),
                nn.ELU(),
                nn.Dropout(0.4),
                nn.Linear(32, 10)
                )

    def forward(self, x):
#        b,c,h,w = x.shape
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AutoEncoderClassifier(nn.Module):
    def __init__(self, opt):
        super(AutoEncoderClassifier, self).__init__()
        self.opt = opt
        self.encoder = Encoder(self.opt)
        if self.opt.onlyAutoEncoder:
            self.decoder = Decoder(self.opt)
        else:
            self.clf = Classifier(self.opt)
        
    def freezeEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        x = self.encoder(x)
        if self.opt.onlyAutoEncoder:
            x = self.decoder(x)
        else:
            x = self.clf(x)
        return x

def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()