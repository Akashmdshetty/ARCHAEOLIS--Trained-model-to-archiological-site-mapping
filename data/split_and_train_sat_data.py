"""
split_and_train_sat_data.py
---------------------------
1. Stratified splitting of 5,944 satellite images into Train (70%), Val (15%), Test (15%).
2. Applies satellite-specific data augmentations (flips, rotations, color jitter).
3. Computes class weights to solve class imbalance (e.g. vehicles vs cloudy).
4. Trains a high-accuracy ResNet18 classifier backed by BYOL pretrained features.
5. Evaluates Test Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
6. Saves the best model checkpoint to models/checkpoints/classifier/sat_classifier_best.pth.
"""

import os
import sys
import shutil
import random
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.resnet_encoder import get_resnet_encoder
from models.classifier_head import ClassifierHead

# ── 1. STRATIFIED DATASET SPLITTING ─────────────────────────────────────────────
SOURCE_DIR = r"C:\Users\aakas\OneDrive\Desktop\sat_data\data"
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "sat_data_split")
CLASSES    = ["cloudy", "desert", "green_area", "vehicles", "water"]
SEED       = 42

def create_stratified_split(src_dir, tgt_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    random.seed(SEED)
    np.random.seed(SEED)

    if os.path.exists(tgt_dir):
        print(f"[Dataset Split] Clean existing target split at: {tgt_dir}")
        shutil.rmtree(tgt_dir)

    for split in ["train", "val", "test"]:
        for cls_name in CLASSES:
            os.makedirs(os.path.join(tgt_dir, split, cls_name), exist_ok=True)

    split_stats = {"class": [], "train": [], "val": [], "test": [], "total": []}

    print("\n" + "="*60)
    print("  STRATIFIED DATASET SPLITTING (70% Train / 15% Val / 15% Test)")
    print("="*60)

    for cls_name in CLASSES:
        cls_folder = os.path.join(src_dir, cls_name)
        if not os.path.exists(cls_folder):
            print(f"[WARN] Class folder {cls_folder} missing, skipping.")
            continue

        images = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        total = len(images)
        n_train = int(total * train_ratio)
        n_val   = int(total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train + n_val]
        test_imgs  = images[n_train + n_val:]

        for img_name in train_imgs:
            shutil.copy2(os.path.join(cls_folder, img_name), os.path.join(tgt_dir, "train", cls_name, img_name))
        for img_name in val_imgs:
            shutil.copy2(os.path.join(cls_folder, img_name), os.path.join(tgt_dir, "val", cls_name, img_name))
        for img_name in test_imgs:
            shutil.copy2(os.path.join(cls_folder, img_name), os.path.join(tgt_dir, "test", cls_name, img_name))

        split_stats["class"].append(cls_name)
        split_stats["train"].append(len(train_imgs))
        split_stats["val"].append(len(val_imgs))
        split_stats["test"].append(len(test_imgs))
        split_stats["total"].append(total)

    df_stats = pd.DataFrame(split_stats)
    print(df_stats.to_string(index=False))
    print(f"\n[OK] Stratified dataset created at: {tgt_dir}\n")
    return df_stats


class SatelliteClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        _, embedding = self.encoder(x)
        return self.classifier(embedding)


# ── 2. HIGH ACCURACY TRAINING PIPELINE ──────────────────────────────────────────
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    train_dir = os.path.join(TARGET_DIR, "train")
    val_dir   = os.path.join(TARGET_DIR, "val")
    test_dir  = os.path.join(TARGET_DIR, "test")

    # Satellite-Specific Heavy Data Augmentations
    train_transforms = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),   # Satellite imagery is rotation-invariant
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = ImageFolder(val_dir, transform=eval_transforms)
    test_dataset  = ImageFolder(test_dir, transform=eval_transforms)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Data Loaders] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"[Class Mapping] {train_dataset.class_to_idx}")

    # Compute Class Weights to handle class imbalance (e.g. vehicles)
    class_counts = [0] * len(CLASSES)
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(CLASSES) * c) for c in class_counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[Class Weights] {dict(zip(CLASSES, [round(w, 2) for w in class_weights]))}")

    # Load Backbone (ResNet18) & Classifier Head
    encoder = get_resnet_encoder(pretrained=True).to(device)
    
    # Check if BYOL SSL Checkpoint exists
    byol_ckpt = os.path.join(PROJECT_ROOT, "models", "checkpoints", "ssl", "byol_final.pth")
    if os.path.exists(byol_ckpt):
        try:
            state = torch.load(byol_ckpt, map_location=device)
            encoder.load_state_dict(state, strict=False)
            print(f"[Model] Successfully initialized ResNet18 with BYOL SSL checkpoint: {byol_ckpt}")
        except Exception as e:
            print(f"[WARN] Failed to load BYOL checkpoint ({e}), using pretrained ResNet18.")

    # Fine-tune layer4 + classifier head for maximum accuracy
    for name, param in encoder.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    classifier = ClassifierHead(
        input_dim=encoder.embedding_dim,
        num_classes=len(CLASSES),
        hidden_dim=512
    ).to(device)

    full_model = SatelliteClassifier(encoder, classifier).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, full_model.parameters()), lr=3e-4, weight_decay=1e-4)
    epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints", "classifier")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "sat_classifier_best.pth")

    print("\n" + "="*60)
    print(f"  STARTING MODEL TRAINING ({epochs} Epochs)")
    print("="*60)

    for epoch in range(epochs):
        full_model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:>2}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = full_model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*train_correct/train_total:.2f}%")

        scheduler.step()

        # Validation Phase
        full_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = full_model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        epoch_val_acc = 100.0 * val_correct / val_total
        epoch_val_loss = val_loss / val_total
        print(f"  --> Epoch {epoch+1:>2}/{epochs} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state": full_model.state_dict(),
                "val_acc": best_val_acc,
                "classes": CLASSES
            }, best_model_path)
            print(f"  [BEST MODEL SAVED] Validation Accuracy: {best_val_acc:.2f}% --> {best_model_path}")

    # ── 3. HELD-OUT TEST EVALUATION ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EVALUATING BEST MODEL ON HELD-OUT TEST SET")
    print("="*60)

    checkpoint = torch.load(best_model_path, map_location=device)
    full_model.load_state_dict(checkpoint["model_state"])
    full_model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            logits = full_model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = 100.0 * np.mean(all_preds == all_labels)
    print(f"\n[RESULTS] FINAL HELD-OUT TEST ACCURACY: {test_acc:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASSES], columns=[f"Pred_{c}" for c in CLASSES])
    print(cm_df)

    # Save summary report to data/sat_data_split/test_results.txt
    res_file = os.path.join(TARGET_DIR, "test_results.txt")
    with open(res_file, "w") as f:
        f.write(f"ARCHAEOLIS Satellite Classifier - Test Results\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())

    print(f"\n[OK] Results saved to: {res_file}")

if __name__ == "__main__":
    create_stratified_split(SOURCE_DIR, TARGET_DIR)
    train_model()
