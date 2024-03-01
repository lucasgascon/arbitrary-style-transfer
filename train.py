import torch
from style_transfer_net import StyleTransferNet, calc_mean_std
from tqdm import tqdm
from dataloader import create_dataloader

def train(content_imgs_path, style_imgs_path, args):
    
    content_trainloader, style_trainloader = create_dataloader(content_imgs_path, style_imgs_path, trainset=True, batch_size=args.batch_size, shuffle=True)
    
    model = StyleTransferNet()
    
    criterion_content = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(lr=args.lr, params=model.decoder.parameters())
    
    for epoch in tqdm(range(args.num_epochs)):
        for content_batch, style_batch in zip(content_trainloader, style_trainloader):
            optimizer.zero_grad()
            
            content_batch = content_batch.to(args.device)
            style_batch = style_batch.to(args.device)

            content_features, style_features = model.encoder(content_batch, style_batch)
            t = model.adain(content_features, style_features)
            output = model.decoder(t)
            
            invert_output = model.encoder(output)
            
            # style_loss = criterion_style(output, style_batch)
            
            # compute the content loss
            # content_loss = criterion_content(invert_output, t)
            content_loss = torch.sum(torch.mean(torch.square(invert_output - t), dim=[1, 2]))

            # compute the style loss
            style_layer_loss = []
            
            for name, layer in model.encoder.encoder_style.named_children():
                enc_style_feat = model.encoder.encoder_style[name]
                enc_gen_feat = model.encoder.encoder_content[name] # Should we use the same layer for the generated image?

                meanS, varS = calc_mean_std(enc_style_feat)
                meanG, varG = calc_mean_std(enc_gen_feat)

                sigmaS = torch.sqrt(varS + args.epsilon)
                sigmaG = torch.sqrt(varG + args.epsilon)

                l2_mean  = torch.sum(torch.square(meanG - meanS))
                l2_sigma = torch.sum(torch.square(sigmaG - sigmaS))

                style_layer_loss.append(l2_mean + l2_sigma)

            style_loss = torch.sum(torch.stack(style_layer_loss))
            
            
            decoder_loss = content_loss + args.style_weight * style_loss
             
            
            decoder_loss.backward()
            optimizer.step()
            
            
            
            