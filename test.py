import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from style_transfer_net import StyleTransferNet, adain, calc_mean_std
from dataloader import create_dataloader
from torchvision import transforms
import cv2
from utils import vizualize_preds
from dataloader import stats

def load_one_img(img_path):
    
        img = cv2.imread(img_path)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        
        if img is None:
            print(f"Error reading {img_path}")
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      

        h, w, _ = img.shape
        if h < w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))
        img = cv2.resize(img, (new_w, new_h))
        img = transform(img)
        return img


def test(args):
    model = StyleTransferNet(args.skipco, args.alpha).to(args.device)

    model_path = 'models/' + args.model_name
    model.decoder.load_state_dict(torch.load(
        model_path, map_location=args.device))

    content_testloader, style_testloader = create_dataloader(
        args.valid_content_imgs, args.valid_style_imgs, trainset=False, batch_size=1, shuffle=False)
    mse_loss = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for content_batch, style_batch in zip(content_testloader, style_testloader):
            content_batch = content_batch.to(args.device)
            style_batch = style_batch.to(args.device)

            styled_images = model(content_batch, style_batch)
            eval_decoder_loss = []
            eval_content_loss = []
            eval_style_loss = []
            with torch.no_grad():
                for content_batch, style_batch in zip(content_testloader, style_testloader):
                    content_batch = content_batch.to(args.device)
                    style_batch = style_batch.to(args.device)

                    content_features = model.encoder(content_batch)
                    style_features = model.encoder(style_batch)
                    t = adain(content_features, style_features)
                    output = model.decoder(t)

                    invert_output = model.encoder(output)

                    # compute the content loss
                    assert (t.requires_grad is False)
                    content_loss = mse_loss(invert_output, t)

                    # compute the style loss
                    style_loss = 0
                    for j in range(4):
                        # Take the accurate layer from the encoder
                        layer = getattr(
                            model.encoder, 'encoder_{:d}'.format(j + 1))
                        style_batch = layer(style_batch).detach()
                        output = layer(output)
                        assert (style_batch.requires_grad is False)
                        meanS, stdS = calc_mean_std(style_batch)
                        meanG, stdG = calc_mean_std(output)
                        style_loss += mse_loss(meanS, meanG) + \
                            mse_loss(stdS, stdG)
                    decoder_loss = content_loss + args.style_weight * style_loss
                    eval_decoder_loss.append(decoder_loss.item())
                    eval_content_loss.append(content_loss.item)
                    eval_style_loss.append(style_loss.item)

                print('Valid Content loss: ', eval_content_loss.mean(), 'Valid Style loss: ', eval_style_loss.mean(),
                    'Valid Total loss: ', eval_decoder_loss.mean())
                if args.show_images:
                    content_img = load_one_img(args.test_content_img).to(args.device)
                    style_img = load_one_img(args.test_style_img).to(args.device)
    
                    content_features = model.encoder(content_img)
                    style_features = model.encoder(style_img)
                    t = adain(content_features, style_features)
                    styled_img = model.decoder(t)

                    print('Displaying the styled images')
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    if args.normalize:
                        content_img = normalize(transforms.ToTensor(content_img))
                        style_img = normalize(transforms.ToTensor(style_img))


                    fig, ax = vizualize_preds(content_img, style_img, styled_img, normalize = args.normalize)
                    fig.savefig('results/Images_test_'+'model_name'+'.png')
                    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_content_imgs', type=str,
                        default='data/val2017', help='Path to the testing content images')
    parser.add_argument('--valid_style_imgs', type=str,
                        default='data/wikiart_small', help='Path to the testing style images')
    parser.add_argument('--testt_content_img', type=str,
                        default='data/val2017', help='Path to the testing content images')
    parser.add_argument('--test_style_img', type=str,
                        default='data/wikiart_small/miriam-schapiro_fanfare-1968.jpg', help='Path to the one testing style')
    parser.add_argument('--test_content_img', type=str,
                        default='data/val2017/000000581357.jpg', help='Path to the one testing content')
    parser.add_argument('--model_name', type=str,
                        default='model.pth', help='Name of the trained model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device,
                        help='Device to run the model on')
    parser.add_argument('--skipco', action='store_true',
                        help='Use skip connections in the decoder')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Alpha value for style/content tradeoff')
    parser.add_argument('--normalize',action="store_true", default=False,
                        help="Normalize with ImageNet stats")
    args = parser.parse_args()
    test(args)
