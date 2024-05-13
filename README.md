# Convolutional Kolmogorov-Arnold Neural Network (Convolutional KAN Layer)

This repository contains an Pytorch implementation of Kolmogorov-Arnold Convolutional Neural Network (KACNN). 

## Description

We proposed a Convolutional KAN Layer (KANConv2D) using a kernel of the B_spline function matrix instead of a real number matrix in conventional CNN. Additionally, some modification is needed to adapt to the convolution function.

Based on `KANConv2D` and `KANLinear`, we developed a Kolmogorov-Arnold Convolutional Neural Network for object classification.

With 69K number of parameters, the performance in MNIST dataset is 98.4%.


## Usage

To define the `KANConv2D`, we call the layer as follows,

```python
from src.efficient_kan import KANConv2D

KANConv2D(in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0, 
        dilatation=1, 
        groups=1,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1])

```
The full KACNN is defined, as follows,

```python
from src.efficient_kan import KAN,KANConv2D

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
```

## Note
This is just simple implementation, I am glad to receive any constructive comment. 


## Acknowledgement
The code was mostly based on [efficient-kan
](https://github.com/Blealtan/efficient-kan).
