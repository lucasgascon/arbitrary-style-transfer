import torch
import torch.nn as nn
from style_transfer_net import StyleTransferNet, calc_mean_std, adain
from tqdm import tqdm
from dataloader import create_dataloader
import argparse
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)

STYLE_LAYERS = [1, 6, 11, 20]

def train(n_epochs, args):
    
    content_trainloader, style_trainloader = create_dataloader(args.train_content_imgs, args.train_style_imgs, trainset=True, batch_size=args.batch_size, shuffle=True)
    content_testloader, style_testloader = create_dataloader(args.test_content_imgs, args.test_style_imgs, trainset=False, batch_size=1, shuffle=False)
    
    model = StyleTransferNet().to(args.device)
    
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    
    for _ in tqdm(range(n_epochs)):
        model.train()
        for content_batch, style_batch in tqdm(zip(content_trainloader, style_trainloader)):
            optimizer.zero_grad()
            
            content_batch = content_batch.to(args.device)
            style_batch = style_batch.to(args.device)

            content_features = model.encoder(content_batch).detach()
            style_features = model.encoder(style_batch).detach()
            t = adain(content_features, style_features)
            output = model.decoder(t)

            invert_output = model.encoder(output)
            
            # compute the content loss
            content_loss = torch.sum(torch.mean((invert_output - t) ** 2, dim=[2, 3]))
            
            # compute the style loss
            style_layer_loss = []
            
            for layer in model.style_layers:
                style_features = layer(style_batch).detach()
                gen_features = layer(output)

                meanS, varS = calc_mean_std(style_features)
                meanG, varG = calc_mean_std(gen_features)

                sigmaS = torch.sqrt(varS + args.epsilon)
                sigmaG = torch.sqrt(varG + args.epsilon)

                l2_mean = torch.sqrt(torch.sum(torch.square(meanG - meanS)))
                l2_sigma = torch.sqrt(torch.sum(torch.square(sigmaG - sigmaS)))

                style_layer_loss.append(l2_mean + l2_sigma)

            style_loss = torch.sum(torch.stack(style_layer_loss))
            
            # decoder_loss = content_loss + args.style_weight * style_loss
            decoder_loss = content_loss
            # decoder_loss = style_loss
            
            # print('content_loss:', content_loss)
            # print('style_loss:', style_loss)
            # print('decoder_loss:', decoder_loss)
             
            decoder_loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            for content_batch, style_batch in zip(content_testloader, style_testloader):
                content_batch = content_batch.to(args.device)
                style_batch = style_batch.to(args.device)

                styled_images = model(content_batch, style_batch)
                
                # Display the styled images
                styled_img = styled_images[0].detach().cpu().numpy().transpose(1, 2, 0)
                styled_img = np.clip(styled_img, 0, 1)  # Ensure the image is in the 0-1 range
                plt.imshow(styled_img)
                plt.show()
            
if __name__ == '__main__':
    args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 1,
    'lr': 1e-3,
    'epsilon': 1e-8,
    'style_weight': 2,
    'train_content_imgs': 'data/content',
    'train_style_imgs': 'data/style',
    'test_content_imgs': 'data/content', # change this to a different directory
    'test_style_imgs': 'data/style' # change this to a different directory
    }
    
    args = argparse.Namespace(**args)
    
    train(1, args)