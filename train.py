import torch
from style_transfer_net import StyleTransferNet, calc_mean_std
from tqdm import tqdm
from dataloader import create_dataloader

STYLE_LAYERS = [1, 6, 11, 20]

args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 8,
    'num_epochs': 10,
    'lr': 1e-3,
    'epsilon': 1e-8,
    'style_weight': 2,
    'content_imgs_path': 'data/content_imgs',
    'style_imgs_path': 'data/style_imgs'
}

def train(content_imgs_path, style_imgs_path, args):
    
    content_trainloader, style_trainloader = create_dataloader(content_imgs_path, style_imgs_path, trainset=True, batch_size=args.batch_size, shuffle=True)
    
    model = StyleTransferNet()
    
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    
    for epoch in tqdm(range(args.num_epochs)):
        for content_batch, style_batch in zip(content_trainloader, style_trainloader):
            optimizer.zero_grad()
            
            content_batch = content_batch.to(args.device)
            style_batch = style_batch.to(args.device)

            content_features, style_features = model.encoder(content_batch, style_batch)
            t = model.adain(content_features, style_features)
            output = model.decoder(t)
            
            invert_output = model.encoder(output)
            
            # compute the content loss
            content_loss = torch.sqrt(torch.sum(torch.square(invert_output - t), dim=[1, 2]))

            # compute the style loss
            style_layer_loss = []
            
            for name, _ in model.encoder.encoder_style.named_children()[STYLE_LAYERS]:
                enc_style_feat = model.encoder.encoder_style[name]
                enc_gen_feat = model.encoder.encoder_content[name] # Should we use the same layer for the generated image?

                meanS, varS = calc_mean_std(enc_style_feat)
                meanG, varG = calc_mean_std(enc_gen_feat)

                sigmaS = torch.sqrt(varS + args.epsilon)
                sigmaG = torch.sqrt(varG + args.epsilon)

                l2_mean = torch.sqrt(torch.sum(torch.square(meanG - meanS)))
                l2_sigma = torch.sqrt(torch.sum(torch.square(sigmaG - sigmaS)))

                style_layer_loss.append(l2_mean + l2_sigma)

            style_loss = torch.sum(torch.stack(style_layer_loss))
            
            decoder_loss = content_loss + args.style_weight * style_loss
             
            decoder_loss.backward()
            optimizer.step()