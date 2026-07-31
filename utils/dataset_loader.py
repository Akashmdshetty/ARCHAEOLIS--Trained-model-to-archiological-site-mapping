import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for loading processed satellite imagery.
    Supports both unlabeled (returns two views) and labeled modes.
    """
    def __init__(self, root_dir, transform=None, labels=None, mode='unlabeled'):
        """
        Args:
            root_dir (str): Path to data/processed
            transform (callable, optional): Augmentation pipeline.
            labels (list, optional): List of labels if mode is 'labeled'.
            mode (str): 'unlabeled' for BYOL (returns 2 views) or 'labeled'.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Load filenames from the processed directory
        self.image_files = sorted([
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            # For BYOL, the transform handles generating multiple views
            image = self.transform(image)
            
        if self.mode == 'labeled' and self.labels is not None:
            label = self.labels[idx]
            return image, label
            
        return image
