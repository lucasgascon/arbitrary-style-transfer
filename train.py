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

def train(args):
    
    content_trainloader, style_trainloader = create_dataloader(args.train_content_imgs, args.train_style_imgs, trainset=True, batch_size=args.batch_size, shuffle=True)
    if args.test:
        content_testloader, style_testloader = create_dataloader(args.test_content_imgs, args.test_style_imgs, trainset=False, batch_size=1, shuffle=False)
    
    model = StyleTransferNet().to(args.device)
    
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        for i, (content_batch, style_batch) in tqdm(enumerate(zip(content_trainloader, style_trainloader))):
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

                l2_mean = torch.sum(torch.square(meanG - meanS))
                l2_sigma = torch.sum(torch.square(sigmaG - sigmaS))

                style_layer_loss.append(l2_mean + l2_sigma)

            style_loss = torch.sum(torch.stack(style_layer_loss))
            
            decoder_loss = content_loss + args.style_weight * style_loss
            
            if i == 0:
                print('content_loss:', content_loss)
                print('style_loss:', style_loss)
             
            decoder_loss.backward()
            optimizer.step()
         
        if args.test:
            model.eval()
            with torch.no_grad():
                for content_batch, style_batch in zip(content_testloader, style_testloader):
                    content_batch = content_batch.to(args.device)
                    style_batch = style_batch.to(args.device)
                    
                
                if args.show_prediction and i == 0:
                    styled_images = model(content_batch, style_batch)
                    
                    # Display the styled images
                    styled_img = styled_images[0].detach().cpu().numpy().transpose(1, 2, 0)
                    styled_img = np.clip(styled_img, 0, 1)  # Ensure the image is in the 0-1 range
                    plt.imshow(styled_img)
                    plt.show()
    
    return model
            
if __name__ == '__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device, help='Device to train the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon value')
    parser.add_argument('--style_weight', type=float, default=2, help='Style weight')
    parser.add_argument('--train_content_imgs', type=str, default='data/val2017', help='Path to the training content images')
    parser.add_argument('--train_style_imgs', type=str, default='data/wikiart_small', help='Path to the training style images')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--show_prediction', action='store_true', help='Display the styled images')
    parser.add_argument('--test_content_imgs', type=str, default=None, help='Path to the test content images')
    parser.add_argument('--test_style_imgs', type=str, default=None, help='Path to the test style images')
    parser.add_argument('--model_name', type=str, default='model.pth', help='Path to save the trained model')
    args = vars(parser.parse_args())
    
    model = train(args)
    
    torch.save(model.state_dict(), args.model_name)