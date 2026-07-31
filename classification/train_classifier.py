import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import sys, pathlib
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.resnet_encoder import get_resnet_encoder
from models.classifier_head import ClassifierHead
from ssl_training.augmentations import get_inference_augmentations
from utils.dataset_loader import SatelliteDataset

def train_classifier():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup directories
    data_dir = config['dataset']['path']
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"Error: No processed data found at {data_dir}. Run data/prepare_dataset.py first.")
        return

    # Mock Labels (In production, load from a CSV or folder structure)
    num_samples = len([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))])
    mock_labels = torch.randint(0, 5, (num_samples,)).tolist() 

    # Dataset
    transform = get_inference_augmentations(img_size=config['dataset']['image_size'])
    dataset = SatelliteDataset(data_dir, transform=transform, labels=mock_labels, mode='labeled')
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Encoder (Initialize and load SSL checkpoint if it exists)
    encoder = get_resnet_encoder(pretrained=False).to(device)
    
    # Try to load latest BYOL checkpoint
    ckpt_dir = config['model']['checkpoint_dir']
    if os.path.exists(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0:
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
        latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
        encoder.load_state_dict(torch.load(latest_ckpt, map_location=device))
        print(f"Loaded SSL Encoder from {latest_ckpt}")
    else:
        print("No SSL checkpoint found. Training from scratch (not recommended).")

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # Classifier Head
    classifier = ClassifierHead(
        input_dim=encoder.embedding_dim,
        num_classes=5, # Example classes
        hidden_dim=config['classifier']['hidden_dim']
    ).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=float(config['classifier']['learning_rate']))

    # Training Loop
    epochs = config['classifier']['epochs']
    os.makedirs(config['classifier']['checkpoint_dir'], exist_ok=True)

    print("Starting Classifier Training (Linear Probing)...")
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                _, features = encoder(images)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

    # Save classifier head
    final_path = os.path.join(config['classifier']['checkpoint_dir'], "classifier_final.pth")
    torch.save(classifier.state_dict(), final_path)
    print(f"Classifier saved to {final_path}")

if __name__ == "__main__":
    train_classifier()
