from src.efficient_kan import KAN,KANConv2D
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class BaseModel(nn.Module):
    def training_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        return loss
    
    def validation_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        acc = accuracy(out,labels)
        return {"val_loss":loss.detach(),"val_acc":acc}
    
    def validation_epoch_end(self,outputs):
        batch_losses = [loss["val_loss"] for loss in outputs]
        loss = torch.stack(batch_losses).mean()
        batch_accuracy = [accuracy["val_acc"] for accuracy in outputs]
        acc = torch.stack(batch_accuracy).mean()
        return {"val_loss":loss.item(),"val_acc":acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        

def KANconv_shortcut(in_channel,out_channel,stride):
    layers = [KANConv2D(in_channel,out_channel,kernel_size=1,stride=stride),
             nn.BatchNorm2d(out_channel)]
    return nn.Sequential(*layers)

def KANblock(in_channel,out_channel,k_size,stride, conv=False):
    layers = None
    
    first_layers = [KANConv2D(in_channel,out_channel[0],kernel_size=1,stride=1),
                    nn.BatchNorm2d(out_channel[0])]
    if conv:
        first_layers[0].stride=stride
    
    second_layers = [KANConv2D(out_channel[0],out_channel[1],kernel_size=k_size,stride=1,padding=1),
                    nn.BatchNorm2d(out_channel[1])]

    layers = first_layers + second_layers
    
    return nn.Sequential(*layers)
    

class KANMResnet(BaseModel):
    
    def __init__(self,in_channels,num_classes, num_base):
        super().__init__()
        
        self.stg1 = nn.Sequential(
                                   KANConv2D(in_channels=in_channels,out_channels=num_base,kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(num_base),
                                   nn.MaxPool2d(kernel_size=3,stride=2))
        
        ##stage 2
        self.convShortcut2 = KANconv_shortcut(num_base,num_base*2,1)
        
        self.conv2 = KANblock(num_base,[num_base,num_base*2],3,1,conv=True)
        self.ident2 = KANblock(num_base*2,[num_base,num_base*2],3,1)

        
        ##stage 3
        self.convShortcut3 = KANconv_shortcut(num_base*2,num_base*4,2)
        
        self.conv3 = KANblock(num_base*2,[num_base,num_base*4],3,2,conv=True)
        self.ident3 = KANblock(num_base*4,[num_base,num_base*4],3,2)

        
        ##stage 4
        self.convShortcut4 = KANconv_shortcut(num_base*4,num_base*4,2)
        
        self.conv4 = KANblock(num_base*4,[num_base*2,num_base*4],3,2,conv=True)
        self.ident4 = KANblock(num_base*4,[num_base*2,num_base*4],3,2)
        
        
        ##Classify
        self.classifier = nn.Sequential(
                                       nn.AvgPool2d(kernel_size=(4)),
                                       nn.Flatten(),
                                       KAN([num_base*4,num_base*8,num_classes]))
        
    def forward(self,inputs):
        out = self.stg1(inputs)
        
        #stage 2
        out = self.conv2(out) + self.convShortcut2(out)
        out = self.ident2(out) + out

        
        #stage3
        out = self.conv3(out) + self.convShortcut3(out)
        out = self.ident3(out) + out

        
#         stage4             
        out = self.conv4(out) + self.convShortcut4(out)
        out = self.ident4(out) + out

        
        out = self.classifier(out)#100x1024
        
        return out

if __name__ == '__main__':
    custom_kan = KANMResnet(1,10,4)
    custom_kan = custom_kan.cuda()

    print("Number of trainable parameters: ", sum(p.numel() for p in custom_kan.parameters() if p.requires_grad))