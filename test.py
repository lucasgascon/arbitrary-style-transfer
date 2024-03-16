import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from style_transfer_net import StyleTransferNet
from dataloader import create_dataloader


def test(args):
    model = StyleTransferNet(args.skipco, args.alpha).to(args.device)

    model_path = 'models/' + args.model_name
    model.decoder.load_state_dict(torch.load(
        model_path, map_location=args.device))

    content_testloader, style_testloader = create_dataloader(
        args.test_content_imgs, args.test_style_imgs, trainset=False, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for content_batch, style_batch in zip(content_testloader, style_testloader):
            content_batch = content_batch.to(args.device)
            style_batch = style_batch.to(args.device)

            styled_images = model(content_batch, style_batch)

            content_img = content_batch[0].detach(
            ).cpu().numpy().transpose(1, 2, 0)
            # Ensure the image is in the 0-1 range
            content_img = np.clip(content_img, 0, 1)

            style_img = style_batch[0].detach(
            ).cpu().numpy().transpose(1, 2, 0)
            # Ensure the image is in the 0-1 range
            style_img = np.clip(style_img, 0, 1)

            # Display the styled images
            styled_img = styled_images[0].detach(
            ).cpu().numpy().transpose(1, 2, 0)
            # Ensure the image is in the 0-1 range
            styled_img = np.clip(styled_img, 0, 1)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(content_img)
            ax[0].set_title('Content Image')
            ax[0].axis('off')
            ax[1].imshow(style_img)
            ax[1].set_title('Style Image')
            ax[1].axis('off')
            ax[2].imshow(styled_img)
            ax[2].set_title('Model output')
            ax[2].axis('off')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_content_imgs', type=str,
                        default='data/val2017', help='Path to the testing content images')
    parser.add_argument('--test_style_imgs', type=str,
                        default='data/wikiart_small', help='Path to the testing style images')
    parser.add_argument('--model_name', type=str,
                        default='model.pth', help='Name of the trained model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device,
                        help='Device to run the model on')
    parser.add_argument('--skipco', action='store_true',
                        help='Use skip connections in the decoder')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Alpha value for style/content tradeoff')
    args = parser.parse_args()
    test(args)
