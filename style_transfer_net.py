import torch.nn as nn
from torchvision import models
import torch

def calc_mean_std(x):
    batch_size, num_channels, h, w = x.size()
    x = x.view(batch_size, num_channels, -1)
    mean = x.mean(dim=2).view(batch_size, num_channels, 1, 1)
    std = x.std(dim=2).view(batch_size, num_channels, 1, 1)
    return mean, std

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def forward(self, content, style):
        meanC, stdC = calc_mean_std(content)
        meanS, stdS = calc_mean_std(style)
        return stdS * (content - meanC) / stdC + meanS

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_content = nn.Sequential(*list(models.vgg19(pretrained=True).features.children())[:23]) # check the number of layers
        self.encoder_style = nn.Sequential(*list(models.vgg19(pretrained=True).features.children())[:23]) # check the number of layers
    
    def forward(self, x, y):
        return self.encoder_content(x), self.encoder_style(y)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return x + self.relu(self.conv2(self.relu(self.conv1(x))))  
       
class Residual_Decoder(nn.Module): # check the number of layers, and the choice of the layers
        def __init__(self):
            super(Residual_Decoder, self).__init__()
            self.decoder_content = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                ResBlock(512, 512),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                ResBlock(256, 256),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                ResBlock(128, 128),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

class Decoder(nn.Module): # check the number of layers, and the choice of the layers
        def __init__(self):
            super(Decoder, self).__init__()
            self.decoder_content = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )
                
class StyleTransferNet():
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        self.encoder = Encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False # freeze the encoder
        self.adain = AdaIN()
        self.decoder = Decoder()
        
    def forward(self, content, style):
        content, style = self.encoder(content, style)
        content = self.adain(content, style)
        return self.decoder(content)
