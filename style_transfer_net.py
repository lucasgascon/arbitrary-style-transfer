import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

def calc_mean_std(x):
    batch_size, num_channels, h, w = x.size()
    x_view = x.view(batch_size, num_channels, -1)
    mean_ = x_view.mean(dim=2).view(batch_size, num_channels, 1, 1)
    std_ = x_view.std(dim=2).view(batch_size, num_channels, 1, 1)
    return mean_, std_

def adain(content, style):
    meanC, stdC = calc_mean_std(content)
    meanS, stdS = calc_mean_std(style)
    output = stdS * (content - meanC) / (stdC + 1e-8) + meanS
    return output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        original_model = vgg19(weights=VGG19_Weights.DEFAULT)
        original_features = list(original_model.features.children())[:22]
        modified_features = nn.Sequential()
        for i, layer in enumerate(original_features):
            if isinstance(layer, nn.ReLU) and layer.inplace:
                # If the layer is a ReLU with inplace=True, replace it with a ReLU with inplace=False
                modified_features.add_module(f"relu_{i}", nn.ReLU(inplace=False))
            else:
                # Otherwise, just add the original layer
                modified_features.add_module(str(i), layer)
        self.encoder = modified_features
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module): # check the choice of the layers
        def __init__(self):
            super(Decoder, self).__init__()
            self.decoder_content = nn.Sequential(
                # Start with 32x32 features
                nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),
                
                nn.Upsample(scale_factor=2, mode='nearest'),  # 128x128
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),

                nn.Upsample(scale_factor=2, mode='nearest'),  # 256x256
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),

                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )
        
        def forward(self, x):
            return self.decoder_content(x)

STYLE_LAYERS = [1, 6, 11, 20]
       
class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        self.encoder = Encoder()
        self.style_layers = [self.encoder.encoder[i] for i in STYLE_LAYERS]
        for param in self.encoder.parameters():
            param.requires_grad = False # freeze the encoder
        self.decoder = Decoder()
        
    def forward(self, content, style):
        content_features = self.encoder(content).detach()
        style_features = self.encoder(style).detach()
        t = adain(content_features, style_features).detach()
        return self.decoder(t)
