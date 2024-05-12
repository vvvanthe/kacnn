import torch
from torchvision.datasets import MNIST
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader

from utils import *
from KANResnet import KANMResnet

epochs = 20
optimizer = torch.optim.SGD
max_lr=0.01
grad_clip = 0.1
weight_decay = 1e-4
scheduler = torch.optim.lr_scheduler.OneCycleLR


stats = (0.5,), (0.5,)#((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    #tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

test_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])

train_data = MNIST(download=True,root="./data",transform=train_transform)
test_data = MNIST(root="./data",train=False,transform=test_transform)

BATCH_SIZE=32
train_dl = DataLoader(train_data,BATCH_SIZE,num_workers=4,pin_memory=True,shuffle=True)
test_dl = DataLoader(test_data,BATCH_SIZE,num_workers=4,pin_memory=True)


device = get_device()


train_dl = ToDeviceLoader(train_dl,device)
test_dl = ToDeviceLoader(test_dl,device)

model = KANMResnet(1,10,4)
model = to_device(model,device)

history = [evaluate(model,test_dl)]


history += fit(epochs=epochs,train_dl=train_dl,test_dl=test_dl,model=model,optimizer=optimizer,max_lr=max_lr,grad_clip=grad_clip,
              weight_decay=weight_decay,scheduler=torch.optim.lr_scheduler.OneCycleLR)