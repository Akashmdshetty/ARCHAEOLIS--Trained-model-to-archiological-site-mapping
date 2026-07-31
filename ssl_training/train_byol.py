"""
train_byol.py
-------------
BYOL self-supervised pre-training on unlabeled satellite images.
Saves the online encoder to models/checkpoints/ssl/ every 10 epochs.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import sys, pathlib
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from ssl_training.byol_model import BYOL
from ssl_training.augmentations import get_byol_augmentations, BYOLViewGenerator
from models.resnet_encoder import get_resnet_encoder
from utils.dataset_loader import SatelliteDataset


def train(resume_from: str = None):
    # ── Load Config ───────────────────────────────────────────────────────
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[BYOL] Using device: {device}")

    processed_dir = config['dataset']['path']
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
        print(f"[ERROR] No images found at '{processed_dir}'.")
        print("       Run:  python data/prepare_dataset.py  first.")
        return

    # ── Dataset ───────────────────────────────────────────────────────────
    img_size  = config['dataset']['image_size']
    base_aug  = get_byol_augmentations(img_size=img_size)
    transform = BYOLViewGenerator(base_aug)
    dataset   = SatelliteDataset(processed_dir, transform=transform, mode='unlabeled')

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),   # 0 = safe on Windows
        pin_memory=torch.cuda.is_available()
    )
    print(f"[BYOL] Dataset: {len(dataset)} images  |  {len(dataloader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────
    backbone = get_resnet_encoder(pretrained=False)
    model = BYOL(
        backbone,
        projection_dim=config['model']['projection_dim'],
        moving_average_decay=0.99
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    start_epoch = 0
    # Optional: resume from checkpoint
    if resume_from and os.path.exists(resume_from):
        state = torch.load(resume_from, map_location=device)
        model.online_encoder.load_state_dict(state)
        print(f"[BYOL] Resumed from: {resume_from}")

    checkpoint_dir = config['model']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    epochs = config['training']['epochs']
    print(f"[BYOL] Starting training for {epochs} epoch(s)...")

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images in pbar:
            view1, view2 = [img.to(device) for img in images]

            optimizer.zero_grad()
            loss = model(view1, view2)
            loss.backward()
            optimizer.step()
            model.update_target_network()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"  → Epoch {epoch+1:>3}/{epochs}  avg_loss={avg_loss:.4f}")

        # Save every 10 epochs and at the final epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"byol_epoch_{epoch+1}.pth")
            torch.save(model.online_encoder.state_dict(), ckpt_path)
            print(f"  [✓] Checkpoint saved → {ckpt_path}")

    # Always save the final encoder
    final_path = os.path.join(checkpoint_dir, "byol_final.pth")
    torch.save(model.online_encoder.state_dict(), final_path)
    print(f"\n[BYOL] Training complete. Final encoder → {final_path}")
    return final_path


if __name__ == "__main__":
    train()
