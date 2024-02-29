import torch.nn as nn
from torchvision import models
import torch
from utils import AdaIN

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_content = nn.Sequential(*list(models.vgg19(pretrained=True).features.children())[:23]) # check the number of layers
        self.encoder_style = nn.Sequential(*list(models.vgg19(pretrained=True).features.children())[:23]) # check the number of layers
    
    def forward(self, x, y):
        return self.encoder_content(x), self.encoder_style(y)

class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            # check the number of layers
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

        def forward(self, x):
            return self.decoder_content(x)
        
def StyleTransferNet():
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, content, style):
        content, style = self.encoder(content, style)
        content = AdaIN(content, style)
        return self.decoder(content)