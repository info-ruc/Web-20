from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import math
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
from torchvision.utils import save_image
import time
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 256  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # 缩放
    transforms.ToTensor()])  # 变成torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # 必须有个batch维度，因此用unsqueeze(0)增加一个batch维度。
    # 如果是CPU，那么loader得到的是(3, 128, 128)，unsqueeze之后变成(1, 3, 128, 123)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=feature map的数量
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # 把mean和std reshape成 [C x 1 x 1]
        # 输入图片是 [B x C x H x W].
        # 因此下面的forward计算可以是用broadcasting 

        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)

    def forward(self, img): 
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,content_layers=content_layers_default,
                               style_layers=style_layers_default):
    #cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []
    
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers: 
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers: 
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
        content_img, style_img, input_img, num_steps=300,
        style_weight=1000000, content_weight=1): 
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # 图片的值必须在(0,1)之间，但是梯度下降可能得到这个范围之外的值，所有需要clamp到这个范围里
            input_img.data.clamp_(0, 1)

            # 清空梯度
            optimizer.zero_grad()
            # forward计算，从而得到SytleLoss和ContentLoss
            model(input_img)
            style_score = 0
            content_score = 0

            # 计算Loss
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # Loss乘以weight
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            # 反向求loss对input_img的梯度
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            # 返回loss给LBFGS
            return style_score + content_score
        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)
    
    return input_img


begin=int(sys.argv[2])
end=int(sys.argv[3])

for i in range(begin,end):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    zero=2-int(math.log10(i))
    content="checkpoints/"+sys.argv[1]+"/web/images/epoch"+"0"*zero+str(i)+"_real_A.png"
    style="checkpoints/"+sys.argv[1]+"/web/images/epoch"+"0"*zero+str(i)+"_real_B.png"
    style_img = image_loader(style)
    content_img = image_loader(content)
    input_img = content_img.clone()
    begin=time.time()
    output=run_style_transfer(cnn,cnn_normalization_mean,cnn_normalization_std,content_img,
    	style_img,input_img,num_steps=500, style_weight=10000)
    end=time.time()
    print(end-begin)
    save_image(output,"checkpoints/"+sys.argv[1]+"/NST/NST"+str(i)+".png")







