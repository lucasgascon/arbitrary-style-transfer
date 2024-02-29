import torch
from style_transfer_net import StyleTransferNet
from tqdm import tqdm
from dataloader import create_dataloader

def train(content_imgs_path, style_imgs_path, args):
    
    content_trainloader, style_trainloader = create_dataloader(content_imgs_path, style_imgs_path, trainset=True, batch_size=args.batch_size, shuffle=True)
    
    model = StyleTransferNet()
    
    criterion_content = torch.nn.MSELoss()
    criterion_style = torch.nn.MSELoss()
    
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
            loss_content = criterion_content(invert_output, t)
            loss_style = criterion_style(output, style_batch)
            
            loss_decoder = loss_content + args.lamb * loss_style
            
            
            loss_decoder.backward()
            optimizer.step()
            
            
            
            