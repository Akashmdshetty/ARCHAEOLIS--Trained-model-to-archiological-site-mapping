import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_proxy(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)

    # Denormalise
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    denorm = (tensor * std + mean).clamp(0, 1)
    img_np = (denorm[0].permute(1,2,0).numpy() * 255).astype(np.uint8)

    # 1. Vegetation Proxy (Green ratio)
    R = img_np[:,:,0].astype(np.float32)
    G = img_np[:,:,1].astype(np.float32)
    B = img_np[:,:,2].astype(np.float32)
    denom = R + G + B + 1e-5
    green_ratio = G / denom
    
    # OLD Ruins / Erosion Proxy was: 1.0 - green_ratio
    # This was way too broad!
    old_erosion_proxy = 1.0 - green_ratio
    
    # NEW PROPOSAL FOR RUINS: Look for very sharp geometric corners/edges combined with low green
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Harris Corner Detection
    # Structures and ruins have high corner density
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    corners = (dst > 0.01 * dst.max()).astype(np.float32)
    
    # Edge Detection (Canny)
    edges = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
    
    # New Ruins Proxy: High edges/corners, low vegetation
    # Blur the corners slightly to create a region
    corner_density = cv2.GaussianBlur(corners, (15, 15), 0)
    corner_density = corner_density / (corner_density.max() + 1e-5)
    
    edge_density = cv2.GaussianBlur(edges, (15, 15), 0)
    edge_density = edge_density / (edge_density.max() + 1e-5)
    
    # Mask out areas that are very green
    veg_mask = (green_ratio > 0.38).astype(np.float32)
    
    new_ruins_proxy = ((corner_density + edge_density) / 2.0) * (1.0 - veg_mask)
    new_ruins_proxy = np.clip(new_ruins_proxy * 1.5, 0, 1) # boost signals
    
    new_erosion_proxy = (1.0 - green_ratio) * (1.0 - veg_mask)
    # Remove sharp geometric areas from erosion
    new_erosion_proxy = new_erosion_proxy * (1.0 - (corner_density > 0.3).astype(np.float32))

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img_np)
    axs[0].set_title("Original")
    axs[1].imshow(old_erosion_proxy, cmap='hot')
    axs[1].set_title("OLD Ruins/Erosion Proxy (Broad)")
    axs[2].imshow(new_ruins_proxy, cmap='hot')
    axs[2].set_title("NEW Ruins Proxy (Corners + Edges - Veg)")
    axs[3].imshow(new_erosion_proxy, cmap='hot')
    axs[3].set_title("NEW Erosion (Bare - Corners - Veg)")
    
    out_path = f"test_output/proxy_test.jpg"
    plt.savefig(out_path)
    print(f"Saved proxy comparison to {out_path}")
    
    # Scalar stats
    print(f"Old Mean Prob: {old_erosion_proxy.mean():.3f}")
    print(f"New Ruins Mean: {new_ruins_proxy.mean():.3f}")
    print(f"New Erosion Mean: {new_erosion_proxy.mean():.3f}")

if __name__ == "__main__":
    processed_dir = "data/processed"
    samples = [f for f in os.listdir(processed_dir) if f.endswith('.jpg')]
    test_proxy(os.path.join(processed_dir, samples[0]))
