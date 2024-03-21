import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


def calc_mean_std(x):
    mean_ = torch.mean(x, dim=[2, 3], keepdim=True)
    std_ = torch.std(x, dim=[2, 3], keepdim=True)
    return mean_, std_


def adain(content, style):
    meanC, stdC = calc_mean_std(content)
    meanS, stdS = calc_mean_std(style)
    output = stdS * (content - meanC) / (stdC + 1e-8) + meanS
    return output



class Alternative_Encoder(nn.Module):
    def __init__(self):
        super(Alternative_Encoder, self).__init__()
        
        vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )


        vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        enc_layers = list(vgg.children())
        self.encoder_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.encoder_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.encoder_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.encoder_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        # fix the encoder
        for name in ['encoder_1', 'encoder_2', 'encoder_3', 'encoder_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
                
    def forward(self, x):
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        x = self.encoder_4(x)
        return x
    
vgg_layers_styleloss = {'relu1_1': [0, 1], 'relu2_1': [2, 3, 4, 5, 6], 'relu3_1': [
    7, 8, 9, 10, 11], 'relu4_1': [12, 13, 14, 15, 16, 17, 18, 19, 20]}

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        original_model = vgg19(weights=VGG19_Weights.DEFAULT)
        original_features = list(original_model.features.children())[:21]
        self.encoder_1 = nn.Sequential()
        self.encoder_2 = nn.Sequential()
        self.encoder_3 = nn.Sequential()
        self.encoder_4 = nn.Sequential()
        for i, layer in enumerate(original_features):
            if i in vgg_layers_styleloss['relu1_1']:
                if isinstance(layer, nn.ReLU) and layer.inplace:
                    # If the layer is a ReLU with inplace=True, replace it with a ReLU with inplace=False
                    self.encoder_1.add_module(
                        "relu1_1", nn.ReLU(inplace=False))
                else:
                    # Otherwise, just add the original layer
                    self.encoder_1.add_module(str(i), layer)

            if i in vgg_layers_styleloss['relu2_1']:
                if isinstance(layer, nn.ReLU) and layer.inplace:
                    # If the layer is a ReLU with inplace=True, replace it with a ReLU with inplace=False
                    self.encoder_2.add_module(
                        "relu2_1", nn.ReLU(inplace=False))
                else:
                    # Otherwise, just add the original layer
                    self.encoder_2.add_module(str(i), layer)

            if i in vgg_layers_styleloss['relu3_1']:
                if isinstance(layer, nn.ReLU) and layer.inplace:
                    # If the layer is a ReLU with inplace=True, replace it with a ReLU with inplace=False
                    self.encoder_3.add_module(
                        "relu3_1", nn.ReLU(inplace=False))
                else:
                    # Otherwise, just add the original layer
                    self.encoder_3.add_module(str(i), layer)

            if i in vgg_layers_styleloss['relu4_1']:
                if isinstance(layer, nn.ReLU) and layer.inplace:
                    # If the layer is a ReLU with inplace=True, replace it with a ReLU with inplace=False
                    self.encoder_4.add_module(
                        "relu4_1", nn.ReLU(inplace=False))
                else:
                    # Otherwise, just add the original layer
                    self.encoder_4.add_module(str(i), layer)

    def forward(self, x):
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        x = self.encoder_4(x)
        return x

    
class Decoder(nn.Module):  # check the choice of the layers
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder_4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.decoder_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.decoder_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.decoder_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, x):
        x = self.decoder_4(x)
        x = self.decoder_3(x)
        x = self.decoder_2(x)
        x = self.decoder_1(x)
        return x

class Decoder_1B(nn.Module):  # check the choice of the layers
    def __init__(self):
        super(Decoder_1B, self).__init__()

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, x):
        return self.decoder(x)


class cat_Decoder(nn.Module):  # check the choice of the layers
    def __init__(self):
        super(cat_Decoder, self).__init__()

        self.decoder_4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        

        self.decoder_3_cat = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.decoder_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
        )

        

        self.decoder_2_cat = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )


        self.decoder_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
        )


        self.decoder_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

          

    def forward(self, x):
        x = self.decoder_4(x)
        x = self.decoder_3(x)
        x = self.decoder_3_cat(x)
        x = self.decoder_2(x)
        x = self.decoder_2_cat(x)
        x = self.decoder_1(x)
        return x
    
class StyleTransferNet(nn.Module):
    def __init__(self, skip_connections=0, alpha=1.0, normed_vgg=False, skip_type = None, cat_decoder = False):
        super(StyleTransferNet, self).__init__()
        if not normed_vgg:
            self.encoder = Encoder()
            for param in self.encoder.parameters():
                param.requires_grad = False  # freeze the encoder
        else:
            self.encoder = Alternative_Encoder()
        if cat_decoder:
            self.decoder = cat_Decoder()
            self.cat = True
            
        else:
            self.cat = False
            self.decoder = Decoder()
            
        self.skip_connections = skip_connections
        self.skip_type = skip_type

        assert 0 <= alpha <= 1
        self.alpha = alpha

    def forward(self, content, style):

        content_1 = self.encoder.encoder_1(content)  # 64 channels
        content_2 = self.encoder.encoder_2(content_1)  # 128 channels
        content_3 = self.encoder.encoder_3(content_2)  # 256 channels
        content_4 = self.encoder.encoder_4(content_3)  # 512 channels

        style_1 = self.encoder.encoder_1(style)
        style_2 = self.encoder.encoder_2(style_1)
        style_3 = self.encoder.encoder_3(style_2)
        style_4 = self.encoder.encoder_4(style_3)

        # Style/ content trade-off
        t = adain(content_4, style_4)
        t = self.alpha * t + (1 - self.alpha) * content_4

        if self.cat:
             # Input is 512 channels and output is 256 channels
            g_t = self.decoder.decoder_4(t)
            g_t = self.decoder.decoder_3_cat(g_t) # 128 channels
 
            if self.skip_type == 'content':
                # g_t = F.interpolate(g_t, size=content_2.size()[
                #             2:], mode='bilinear', align_corners=False)
                g_t = torch.cat([g_t, content_2], dim=1)
            elif self.skip_type == 'style':
                # g_t = F.interpolate(g_t, size=style_2.size()[
                #             2:], mode='bilinear', align_corners=False)
                g_t = torch.cat([g_t, style_2], dim=1)
                    


            # Input is 256 channels and output is 128 channels
            g_t = self.decoder.decoder_3(g_t)
            g_t = self.decoder.decoder_2_cat(g_t) # 64 channels
            
            
            if self.skip_type == 'content':
                # g_t = F.interpolate(g_t, size=content_1.size()[
                #             2:], mode='bilinear', align_corners=False)
                g_t = torch.cat([g_t, content_1], dim=1)
            elif self.skip_type == 'style':
                # g_t = F.interpolate(g_t, size=style_1.size()[
                #             2:], mode='bilinear', align_corners=False)
                g_t = torch.cat([g_t, style_1], dim=1)
                

            # Input is 128 channels and output is 64 channels
            g_t = self.decoder.decoder_2(g_t)

            # Input is 64 channels and output is 3 channels
            g_t = self.decoder.decoder_1(g_t)
             
        else:
            # Input is 512 channels and output is 256 channels
            g_t = self.decoder.decoder_4(t)

            if self.skip_connections>0:
                if self.skip_type == 'content':
                    g_t = g_t + content_3*self.skip_connections
                elif self.skip_type == 'style':
                    g_t = g_t + style_3*self.skip_connections
                elif self.skip_type == 'both':
                    g_t = g_t + (content_3 + style_3)*self.skip_connections
                    


            # Input is 256 channels and output is 128 channels
            g_t = self.decoder.decoder_3(g_t)

            if self.skip_connections>0:
                if self.skip_type == 'content':
                    g_t = g_t + content_2*self.skip_connections
                elif self.skip_type == 'style':
                    g_t = g_t + style_2*self.skip_connections
                elif self.skip_type == 'both':
                    g_t = g_t + (content_2 + style_2)*self.skip_connections
                    


            # Input is 128 channels and output is 64 channels
            g_t = self.decoder.decoder_2(g_t)

            if self.skip_connections>0:
                
                if self.skip_type == 'content':
                    g_t = g_t + content_1*self.skip_connections
                elif self.skip_type == 'style':
                    g_t = g_t + style_1*self.skip_connections
                elif self.skip_type == 'both':
                    g_t = g_t + (content_1 + style_1)*self.skip_connections
                    
            # Input is 64 channels and output is 3 channels
            g_t = self.decoder.decoder_1(g_t)

        return g_t, t

# <bound method Module.named_parameters of VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True) # relu1-1
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True) # relu1-2
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True) # relu2-1
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True) # relu2-2
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True) # relu3-1
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True) # relu3-2
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True) # relu3-3
#     (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (17): ReLU(inplace=True) # relu3-4
#     (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True) # relu4-1
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True) # relu4-2
#   )
