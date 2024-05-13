# Kolmogorov-Arnold Convolutional Neural Network

This repository contains an implementation of Kolmogorov-Arnold Convolutional Neural Network (KACNN). 

## Description

We proposed a kernel of B_spline function matrix instead of a real number matrix in conventional CNN. Additionally, some modification is needed to adapted into convolution function.

With 69K number of parameters, the performance in MNIST dataset is 98.4%.


## Usage

```python
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

## Note
This is just simple implementation, I am glad to receive any constructive comment. 


## Acknowledgement
The code was mostly provided by [efficient-kan
](https://github.com/Blealtan/efficient-kan).
