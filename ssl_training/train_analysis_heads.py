"""
train_analysis_heads.py
-----------------------
Fine-tunes the MultiTaskArchaeologist analysis heads on top of a
frozen BYOL-trained ResNet18 encoder using unsupervised proxy tasks.

Proxy task targets (no labels needed):
  • Segmentation  → reconstruct local texture/gradient features
  • Erosion       → proxy from normalised green-channel ratio (NDVI-like)
  • Faults        → Canny edge pseudo-labels (structural discontinuities)
  • Landslide     → composite of erosion proxy + edge density scalar

Run after train_byol.py completes.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import yaml
import os
import sys, pathlib
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.resnet_encoder import get_resnet_encoder
from models.analysis_heads import MultiTaskArchaeologist
from utils.dataset_loader import SatelliteDataset
import torchvision.transforms as T


# ── Encoder with intermediate feature hooks ────────────────────────────────
class EncoderWithFeatures(nn.Module):
    """
    Wraps ResNetEncoder (which stores ResNet18 under self.model)
    to expose layer1–layer4 intermediate feature maps.
    """
    def __init__(self, base_encoder):
        super().__init__()
        resnet = base_encoder.model   # the underlying torchvision ResNet18
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1   # [B, 64,  56, 56]
        self.layer2 = resnet.layer2   # [B, 128, 28, 28]
        self.layer3 = resnet.layer3   # [B, 256, 14, 14]
        self.layer4 = resnet.layer4   # [B, 512,  7,  7]

    def forward(self, x):
        x  = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


# ── Proxy target generation (CPU, numpy) ──────────────────────────────────
def make_proxy_targets(imgs_tensor: torch.Tensor, out_size: int = 224):
    """
    Given a batch of normalised images (B,3,H,W), compute tighter proxy targets:
      - erosion_map  : Bare soil but NO sharp geometry [B,1,H,W]
      - segmentation : Class 1 (Ruins) based on SHARP geometry (corners/edges) [B,H,W]
      - fault_mask   : Macro Canny edge structures [B,1,H,W]
      - landslide    : composite 
    """
    B = imgs_tensor.size(0)
    erosion_maps  = []
    fault_masks   = []
    ruin_masks    = []
    veg_masks     = []
    landslide_scores = []

    # Denormalise from ImageNet stats → [0,255] uint8
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    denorm = (imgs_tensor.cpu() * std + mean).clamp(0, 1)

    for i in range(B):
        img_np = (denorm[i].permute(1,2,0).numpy() * 255).astype(np.uint8)

        # 1. Base colour properties
        R = img_np[:,:,0].astype(np.float32)
        G = img_np[:,:,1].astype(np.float32)
        B_ch = img_np[:,:,2].astype(np.float32)
        denom = R + G + B_ch + 1e-5
        green_ratio = G / denom                         # [0,1]
        
        # 2. Structural properties (Geometry)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray_float = np.float32(gray)
        dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        corners = (dst > 0.01 * dst.max()).astype(np.float32)
        
        edges = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
        
        # Blur them out slightly so the CNN learns regions, not 1px points
        corner_density = cv2.GaussianBlur(corners, (15, 15), 0)
        if corner_density.max() > 0: corner_density /= corner_density.max()
        
        edge_density = cv2.GaussianBlur(edges, (15, 15), 0)
        if edge_density.max() > 0: edge_density /= edge_density.max()
        
        # 3. New stricter proxies
        # Ruins = High structure + low vegetation
        veg_mask = (green_ratio > 0.38).astype(np.float32)
        veg_masks.append(veg_mask)
        ruins_proxy = ((corner_density + edge_density) / 2.0) * (1.0 - veg_mask)
        ruins_proxy = np.clip(ruins_proxy * 1.5, 0, 1) # Boost weak signals
        ruin_masks.append(ruins_proxy)
        
        # Erosion = Bare ground (low veg) MINUS sharp structural areas
        erosion_proxy = (1.0 - green_ratio) * (1.0 - veg_mask)
        erosion_proxy = erosion_proxy * (1.0 - (corner_density > 0.3).astype(np.float32))
        erosion_maps.append(erosion_proxy)

        # Fault proxy: Just raw Canny edges (different thresholding)
        macro_edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        fault_masks.append(macro_edges)

        # Landslide score: overall erosion + macro edge density
        ls_score = 0.5 * float(erosion_proxy.mean()) + 0.5 * float(macro_edges.mean())
        landslide_scores.append(ls_score)

    erosion_t   = torch.tensor(np.stack(erosion_maps), dtype=torch.float32).unsqueeze(1)
    fault_t     = torch.tensor(np.stack(fault_masks),  dtype=torch.float32).unsqueeze(1)
    ruin_t      = torch.tensor(np.stack(ruin_masks),   dtype=torch.float32).unsqueeze(1)
    veg_t       = torch.tensor(np.stack(veg_masks),    dtype=torch.float32).unsqueeze(1)
    landslide_t = torch.tensor(landslide_scores,       dtype=torch.float32).unsqueeze(1)

    return erosion_t, fault_t, ruin_t, veg_t, landslide_t


# ── Training ──────────────────────────────────────────────────────────────
def train():
    with open('configs/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Analysis] Using device: {device}")

    processed_dir = cfg['dataset']['path']
    if not os.path.exists(processed_dir) or not os.listdir(processed_dir):
        print("[ERROR] No processed images found. Run data/prepare_dataset.py first.")
        return

    # Determine latest BYOL checkpoint
    ssl_dir = cfg['model']['checkpoint_dir']
    final_ckpt = os.path.join(ssl_dir, "byol_final.pth")
    byol_ckpt  = None
    if os.path.exists(final_ckpt):
        byol_ckpt = final_ckpt
    else:
        # Pick the latest epoch checkpoint
        ckpts = sorted([f for f in os.listdir(ssl_dir) if f.endswith('.pth')]) if os.path.isdir(ssl_dir) else []
        if ckpts:
            byol_ckpt = os.path.join(ssl_dir, ckpts[-1])

    # ── Build Encoder ─────────────────────────────────────────────────────
    base = get_resnet_encoder(pretrained=False)
    if byol_ckpt:
        # The BYOL encoder state dict contains the full online encoder
        # (stem + projector). We only need the ResNet backbone layers.
        state = torch.load(byol_ckpt, map_location='cpu')
        # Filter to only ResNet layers (exclude projector)
        backbone_state = {k: v for k, v in state.items()
                         if not k.startswith('projector') and not k.startswith('fc')}
        missing, unexpected = base.load_state_dict(backbone_state, strict=False)
        print(f"[Analysis] Loaded BYOL encoder from {byol_ckpt}")
        print(f"           Missing keys: {len(missing)}  Unexpected: {len(unexpected)}")
    else:
        print("[Analysis] No BYOL checkpoint found — using random initialisation.")

    encoder = EncoderWithFeatures(base).to(device)
    for param in encoder.parameters():           # freeze encoder
        param.requires_grad = False
    encoder.eval()
    print("[Analysis] Encoder frozen.")

    # ── Analysis Heads ────────────────────────────────────────────────────
    heads = MultiTaskArchaeologist(encoder_channels=[64, 128, 256, 512]).to(device)

    # ── Data ──────────────────────────────────────────────────────────────
    transform = T.Compose([
        T.Resize((cfg['dataset']['image_size'], cfg['dataset']['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset    = SatelliteDataset(processed_dir, transform=transform, mode='unlabeled')
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )
    print(f"[Analysis] Dataset: {len(dataset)} images  |  {len(dataloader)} batches/epoch")

    # ── Loss Functions ────────────────────────────────────────────────────
    seg_loss_fn      = nn.CrossEntropyLoss()
    erosion_loss_fn  = nn.MSELoss()
    fault_loss_fn    = nn.BCELoss()
    landslide_loss_fn = nn.MSELoss()

    optimizer = optim.Adam(heads.parameters(),
                           lr=float(cfg['analysis_heads']['learning_rate']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['analysis_heads']['epochs'])

    out_dir = cfg['analysis_heads']['checkpoint_dir']
    os.makedirs(out_dir, exist_ok=True)

    epochs = cfg['analysis_heads'].get('epochs', 5)
    print(f"[Analysis] Training for {epochs} epoch(s)...\n")

    for epoch in range(epochs):
        heads.train()
        totals = dict(seg=0, erosion=0, fault=0, landslide=0, total=0)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for imgs in pbar:
            # imgs is a single tensor (unlabeled mode returns the image directly)
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]      # handle any view-wrapper
            imgs = imgs.to(device)

            # Generate proxy targets (CPU ops)
            erosion_targets, fault_targets, ruin_targets, veg_targets, landslide_targets = make_proxy_targets(
                imgs, out_size=cfg['dataset']['image_size'])
            erosion_targets  = erosion_targets.to(device)
            fault_targets    = fault_targets.to(device)
            ruin_targets     = ruin_targets.to(device)
            veg_targets      = veg_targets.to(device)
            landslide_targets = landslide_targets.to(device)

            # Encoder forward (no grad)
            with torch.no_grad():
                features = encoder(imgs)

            # Heads forward
            outputs = heads(features)         # dict of all outputs
            seg_out   = outputs['segmentation']   # [B,3,H,W]
            eros_out  = outputs['erosion']         # [B,1,H,W]
            fault_out = outputs['faults']          # [B,1,H,W]

            # ── Losses ────────────────────────────────────────────────────
            # Segmentation uses the dedicated ruin and veg geometric targets
            H, W = seg_out.shape[2], seg_out.shape[3]
            ruin_resized = torch.nn.functional.interpolate(
                ruin_targets, size=(H, W), mode='bilinear', align_corners=False)
            veg_resized = torch.nn.functional.interpolate(
                veg_targets, size=(H, W), mode='bilinear', align_corners=False)
            
            # Create a 3-class target: 0=bg, 1=ruins, 2=veg
            B = seg_out.size(0)
            seg_pseudo = torch.zeros((B, H, W), dtype=torch.long, device=device)
            seg_pseudo[veg_resized.squeeze(1) > 0.4] = 2   # Vegetation first
            seg_pseudo[ruin_resized.squeeze(1) > 0.4] = 1  # Ruins overwrite vegetation (stricter structure)
            
            l_seg = seg_loss_fn(seg_out, seg_pseudo)

            # Erosion heatmap
            eros_resized = torch.nn.functional.interpolate(
                erosion_targets, size=eros_out.shape[2:], mode='bilinear', align_corners=False)
            l_erosion = erosion_loss_fn(eros_out, eros_resized)

            # Fault mask
            fault_resized = torch.nn.functional.interpolate(
                fault_targets, size=fault_out.shape[2:], mode='bilinear', align_corners=False)
            l_fault = fault_loss_fn(fault_out, fault_resized)

            # Landslide scalar — pool erosion head output
            landslide_pred = eros_out.mean(dim=[2, 3])               # [B,1]
            l_landslide = landslide_loss_fn(landslide_pred, landslide_targets)

            loss = l_seg + l_erosion + l_fault + l_landslide

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals['seg']       += l_seg.item()
            totals['erosion']   += l_erosion.item()
            totals['fault']     += l_fault.item()
            totals['landslide'] += l_landslide.item()
            totals['total']     += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        n = len(dataloader)
        print(f"  Epoch {epoch+1:>2}/{epochs} | "
              f"total={totals['total']/n:.4f}  "
              f"seg={totals['seg']/n:.4f}  "
              f"erosion={totals['erosion']/n:.4f}  "
              f"fault={totals['fault']/n:.4f}  "
              f"landslide={totals['landslide']/n:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            ckpt = os.path.join(out_dir, f"analysis_epoch_{epoch+1}.pth")
            torch.save(heads.state_dict(), ckpt)
            print(f"  [✓] Saved → {ckpt}")

    final = os.path.join(out_dir, "analysis_heads_final.pth")
    torch.save(heads.state_dict(), final)
    print(f"\n[Analysis] Training complete. Final checkpoint → {final}")


if __name__ == "__main__":
    train()
