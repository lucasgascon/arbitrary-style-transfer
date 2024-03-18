import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        img = cv2.imread(img_path)
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


        if self.transform is not None:
            img = self.transform(img)
        return img


def create_dataloader(content_path, style_path, trainset, batch_size=1, shuffle=True):
    if trainset:
        transform = transforms.Compose([
            # Since OpenCV images are NumPy arrays, convert to PIL Image first
            transforms.ToPILImage(),
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    content_dataset = CustomDataset(content_path, transform=transform)
    content_dataloader = DataLoader(
        content_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    style_dataset = CustomDataset(style_path, transform=transform)
    style_dataloader = DataLoader(
        style_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return content_dataloader, style_dataloader
