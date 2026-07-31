import torch
import yaml
import os
import sys, pathlib
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.resnet_encoder import get_resnet_encoder
from ssl_training.augmentations import get_inference_augmentations
from utils.dataset_loader import SatelliteDataset

def extract_embeddings():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    data_dir = config['dataset']['path']
    
    # Dataset
    transform = get_inference_augmentations(img_size=config['dataset']['image_size'])
    dataset = SatelliteDataset(data_dir, transform=transform, mode='unlabeled')
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Load Encoder
    encoder = get_resnet_encoder(pretrained=False).to(device)
    ckpt_dir = config['model']['checkpoint_dir']
    
    if os.path.exists(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0:
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
        latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
        encoder.load_state_dict(torch.load(latest_ckpt, map_location=device))
        print(f"Loaded encoder from {latest_ckpt}")
    else:
        print("Warning: No SSL checkpoint found. Extracting random features.")

    encoder.eval()
    
    embeddings = []
    filenames = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for i, images in enumerate(tqdm(dataloader)):
            images = images.to(device)
            _, feats = encoder(images)
            embeddings.append(feats.cpu())
            
            # Keep track of which image produced these features
            batch_files = dataset.image_files[i*config['training']['batch_size'] : (i+1)*config['training']['batch_size']]
            filenames.extend(batch_files)

    embeddings = torch.cat(embeddings).numpy()
    
    # Save to CSV for clustering
    df = pd.DataFrame(embeddings)
    df['filename'] = filenames
    
    output_path = "data/processed/embeddings.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(filenames)} embeddings to {output_path}")

if __name__ == "__main__":
    extract_embeddings()
