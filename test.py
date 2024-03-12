import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from style_transfer_net import StyleTransferNet
from dataloader import create_dataloader

def test(args):
    model = StyleTransferNet().to(args.device)
    
    model_path = 'models/' + args.model_path
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    
    content_testloader, style_testloader = create_dataloader(args.test_content_imgs, args.test_style_imgs, trainset=False, batch_size=1, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for content_batch, style_batch in zip(content_testloader, style_testloader):
            content_batch = content_batch.to(args.device)
            style_batch = style_batch.to(args.device)

            styled_images = model(content_batch, style_batch)
            # for i in range(styled_images.shape[0]):
            #     save_image(styled_images[i], os.path.join(args.output_dir, f'{i}.png'))
            
            # Display the styled images
            styled_img = styled_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            styled_img = np.clip(styled_img, 0, 1)  # Ensure the image is in the 0-1 range
            plt.imshow(styled_img)
            plt.show()
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_content_imgs', type=str, default='data/val2017', help='Path to the testing content images')
    parser.add_argument('--test_style_imgs', type=str, default='data/wikiart_small', help='Path to the testing style images')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device, help='Device to run the model on')
    args = parser.parse_args()
    test(args)