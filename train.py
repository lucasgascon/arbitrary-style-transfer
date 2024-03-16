import torch
import torch.nn as nn
from style_transfer_net import StyleTransferNet, calc_mean_std, adain
from tqdm import tqdm
from dataloader import create_dataloader
import argparse
import matplotlib.pyplot as plt
import numpy as np
import wandb

torch.autograd.set_detect_anomaly(True)


def adjust_learning_rate(optimizer, lr, lr_decay, iteration_count):
    """Imitating the original implementation"""
    lr /= (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):
    """Enable to monitor results during training
    """
    if (args.wandb):
        wandb.init(
            name=args.model_name,
            entity=args.wandb_entity,
            project="AdaIN",
            config={
                "architecture": args.model_name,
                "learning_rate": args.lr,
                "learning_rate_decay": args.lr_decay,
                "batch_size": args.batch_size,
                "alpha": args.alpha,
                "style_weight": args.style_weight,
            })

        wandb.config = {
            "architecture": args.model_name,
            "learning_rate": args.lr,
            "learning_rate_decay": args.lr_decay,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "style_weight": args.style_weight,
        }

    content_trainloader, style_trainloader = create_dataloader(
        args.train_content_imgs, args.train_style_imgs, trainset=True, batch_size=args.batch_size, shuffle=True)
    if args.test:
        content_testloader, style_testloader = create_dataloader(
            args.test_content_imgs, args.test_style_imgs, trainset=False, batch_size=1, shuffle=False)

    len_data = min(len(content_trainloader), len(style_trainloader))
    model = StyleTransferNet(args.skipco, args.alpha).to(args.device)

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    mse_loss = torch.nn.MSELoss()

    model.to(args.device)

    for epoch in range(args.n_epochs):
        model.train()
        for i, (content_batch, style_batch) in tqdm(enumerate(zip(content_trainloader, style_trainloader))):
            adjust_learning_rate(optimizer, args.lr,
                                 args.lr_decay, (epoch*len_data+i))
            optimizer.zero_grad()
            content_batch = content_batch.to(args.device)
            style_batch = style_batch.to(args.device)

            content_features = model.encoder(content_batch).detach()
            style_features = model.encoder(style_batch).detach()
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
                layer = getattr(model.encoder, 'encoder_{:d}'.format(j + 1))
                style_batch = layer(style_batch).detach()
                output = layer(output)
                assert (style_batch.requires_grad is False)
                meanS, stdS = calc_mean_std(style_batch)
                meanG, stdG = calc_mean_std(output)
                style_loss += mse_loss(meanS, meanG) + mse_loss(stdS, stdG)

            decoder_loss = content_loss + args.style_weight * style_loss

            if i == 0:
                print('Epoch: ', epoch, 'Content loss: ', content_loss.item(), 'Style loss: ', style_loss.item(),
                      'Total loss: ', decoder_loss.item())
            # Logging to Weights and Biases
            if (args.wandb):
                wandb.log({'Train content Loss': content_loss.item(), ' Train style loss': style_loss.item(),
                           'Train overall Loss': decoder_loss.item()}, step=(epoch*len_data+i))

            if (epoch*len_data+i) % args.save_model_interval == 0:
                state_dict = model.decoder.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, 'models/'+args.model_name +
                           'decoder_epoch_{:d}.pth.tar'.format(epoch))

            decoder_loss.backward()
            optimizer.step()

        if args.test and epoch % 2 == 0:
            model.eval()
            eval_decoder_loss = []
            eval_content_loss = []
            eval_style_loss = []
            with torch.no_grad():
                for content_batch, style_batch in zip(content_testloader, style_testloader):
                    content_batch = content_batch.to(args.device)
                    style_batch = style_batch.to(args.device)

                    content_features = model.encoder(content_batch).detach()
                    style_features = model.encoder(style_batch).detach()
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

                print('Epoch: ', epoch, 'Valid Content loss: ', eval_content_loss.mean(), 'Valid Style loss: ', eval_style_loss.mean(),
                      'Valid Total loss: ', eval_decoder_loss.mean())
                # Logging to Weights and Biases
                if (args.wandb):
                    wandb.log({'Valid content loss': content_loss.item(), 'Valid style loss': style_loss.item(),
                               'Valid overall loss': decoder_loss.item()}, step=(epoch*len_data+i))

                if args.show_prediction:
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
                    fig.savefig('results/Images_{:d}.png'.format(epoch))

    if args.wandb:
        wandb.finish()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Whether to log metrics to Weights & Biases")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb username or team name to which runs are attributed"
                        )
    parser.add_argument('--n_epochs', type=int,
                        default=100, help='Number of epochs')
    parser.add_argument('--save_model_interval', type=int, default=10000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device,
                        help='Device to train the model')
    parser.add_argument('--train_content_imgs', type=str,
                        default='data/val2017', help='Path to the training content images')
    parser.add_argument('--train_style_imgs', type=str,
                        default='data/wikiart', help='Path to the training style images')
    parser.add_argument('--test', action='store_true', help='Test the model')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float,
                        default=1e-8, help='Epsilon value')

    parser.add_argument('--style_weight', type=float,
                        default=10, help='Style weight')

    parser.add_argument('--show_prediction', action='store_true',
                        help='Display the styled images')
    parser.add_argument('--test_content_imgs', type=str,
                        default='data/test2017', help='Path to the test content images')
    parser.add_argument('--test_style_imgs', type=str,
                        default='data/wikiart_small', help='Path to the test style images')
    parser.add_argument('--model_name', type=str,
                        default='model.pth', help='Path to save the trained model')

    parser.add_argument('--skipco', action='store_true',
                        help='Use skip connections in the decoder')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Alpha value for style/content tradeoff')

    args = parser.parse_args()

    model = train(args)

    model_path = 'models/' + args.model_name
    state_dict = model.decoder.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, 'models/'+args.model_name+'decoder_final.pth.tar')
