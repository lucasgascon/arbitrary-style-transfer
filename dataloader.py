import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        img = cv2.imread(img_path)

        h, w, _ = img.shape
        if h < w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))

        img = cv2.resize(img, (new_w, new_h))

        return img

def create_dataloader(content_path, style_path, trainset=True, batch_size=8, shuffle=True):
    if trainset:
        transform = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])
        content_dataset = CustomDataset(content_path)
        content_dataloader = DataLoader(content_dataset, batch_size=batch_size, shuffle=shuffle, transform=transform)
        
        style_dataset = CustomDataset(style_path)
        style_dataloader = DataLoader(style_dataset, batch_size=batch_size, shuffle=shuffle, transform=transform)

        return content_dataloader, style_dataloader
    else:
        content_dataset = Dataset(content_path)
        content_dataloader = DataLoader(content_dataset, batch_size=batch_size, shuffle=shuffle)
        
        style_dataset = Dataset(style_path)
        style_dataloader = DataLoader(style_dataset, batch_size=batch_size, shuffle=shuffle)

        return content_dataloader, style_dataloader
    