#!/usr/bin/env python3
"""
trainer_95.py

Train three small models on RTX4050 with limited GPU memory.
Uses simple weighted voting and test-time augmentation.
"""

import os
import gc
import glob
import cv2
import torch
import timm
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn

# --- Setup and memory cleanup ---
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# --- Paths ---
BASE = "/home/rithish/Downloads/PALS_ML"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train", "train")
TEST_IMG_DIR = os.path.join(BASE, "test")
OUTPUT_FILE = "final_submission_95plus.csv"

# --- Load labels ---
df = pd.read_csv(TRAIN_CSV)
df.columns = ['filename', 'label']
classes = sorted(df['label'].unique())
label_to_idx = {c: i for i, c in enumerate(classes)}
print(f"{len(df)} samples across {len(classes)} classes")

# --- Data transforms ---
IMG_SIZE = 224
train_transform = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(30),
    T.ColorJitter(0.4, 0.4, 0.4, 0.2),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    T.RandomErasing(p=0.3, scale=(0.02,0.1)),
])
val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# --- Dataset class ---
class RiceDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.root, row['filename'])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = self.transform(Image.fromarray(img))
        label = label_to_idx[row['label']]
        return img, label

# --- Train/validation split ---
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
train_loader = DataLoader(
    RiceDataset(train_df, TRAIN_IMG_DIR, train_transform),
    batch_size=4, shuffle=True, num_workers=1, pin_memory=True
)
val_loader = DataLoader(
    RiceDataset(val_df, TRAIN_IMG_DIR, val_transform),
    batch_size=8, shuffle=False, num_workers=1, pin_memory=True
)

# --- Models to train ---
model_names = ['tf_efficientnetv2_s', 'tf_efficientnet_b0', 'mobilenetv3_large_100']
trained_models = []
model_weights = []

for name in model_names:
    print(f"\nTraining {name}")
    torch.cuda.empty_cache()
    gc.collect()

    model = timm.create_model(name, pretrained=True, num_classes=len(classes))
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    best_f1, patience = 0, 0

    for epoch in range(30):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels) / 4
            loss.backward()
            if (optimizer._step_count + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    out = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
                    preds.extend(out)
                    targets.extend(labels.numpy())
            f1 = f1_score(targets, preds, average='micro') * 100
            print(f" Epoch {epoch+1} F1: {f1:.2f}%")
            if f1 > best_f1:
                best_f1, patience = f1, 0
                torch.save(model.state_dict(), f"best_{name}.pth")
            else:
                patience += 1
                if patience >= 10:
                    print(" Early stopping")
                    break

    # Load best and record weight
    model.load_state_dict(torch.load(f"best_{name}.pth"))
    trained_models.append(model)
    model_weights.append(best_f1)
    print(f"{name} best F1: {best_f1:.2f}%")

# Normalize weights
total = sum(model_weights)
model_weights = [w/total for w in model_weights]
print("Ensemble weights:", ["%.2f" % w for w in model_weights])

# --- Test-time augmentation and inference ---
tta = [
    val_transform,
    lambda x: T.Compose([T.RandomHorizontalFlip(p=1.0), val_transform])(x),
    lambda x: T.Compose([T.RandomVerticalFlip(p=1.0), val_transform])(x),
    lambda x: T.Compose([T.RandomRotation(15), val_transform])(x),
]

test_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
all_preds = []

for fp in tqdm(test_files, desc="Predicting"):
    img = Image.open(fp).convert("RGB")
    weighted_scores = None

    for m, w in zip(trained_models, model_weights):
        m.eval()
        tta_scores = []
        with torch.no_grad():
            for tfm in tta:
                inp = tfm(img).unsqueeze(0).to(DEVICE)
                tta_scores.append(m(inp).softmax(1).cpu())
        avg_score = torch.stack(tta_scores).mean(0) * w
        weighted_scores = avg_score if weighted_scores is None else weighted_scores + avg_score

    pred = weighted_scores.argmax(1).item()
    all_preds.append(classes[pred])
    torch.cuda.empty_cache()

# --- Save submission ---
sub = pd.DataFrame({
    "ID": [os.path.basename(x) for x in test_files],
    "TARGET": all_preds
})
sub.to_csv(OUTPUT_FILE, index=False)
print("Saved submission to", OUTPUT_FILE)
