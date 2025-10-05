#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import cv2
import pickle
from datetime import datetime

# Disable GPU to prevent memory issues on my system
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Rice Classification Model Training")
print("Running on CPU (more stable for my setup)")

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

print("Libraries loaded successfully")

# Check if we're in the right directory
if not os.path.exists('./train.csv'):
    print("Error: train.csv not found in current directory")
    print("Please navigate to the data folder first")
    exit(1)

# Look for training images in common locations
image_dir = None
search_paths = ['train/train', './train/train', 'train', './train']

for path in search_paths:
    if os.path.exists(path):
        image_files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(image_files) > 100:  # Reasonable threshold for training data
            image_dir = path
            print(f"Found {len(image_files)} training images in: {path}")
            break

if image_dir is None:
    print("Could not locate training images!")
    exit(1)

# Load the training data
df = pd.read_csv('train.csv')
print(f"Loaded training metadata: {len(df)} entries")

# Validate that images exist for the labels
print("Checking image-label correspondence...")
valid_data = []
missing_files = 0

for idx, row in df.iterrows():
    image_path = os.path.join(image_dir, row['ID'])
    if os.path.exists(image_path):
        valid_data.append((image_path, row['TARGET']))
    else:
        missing_files += 1

print(f"Valid training pairs: {len(valid_data)}")
if missing_files > 0:
    print(f"Warning: {missing_files} images are missing")

# Encode the class labels
target_labels = [item[1] for item in valid_data]
encoder = LabelEncoder()
encoded_targets = encoder.fit_transform(target_labels)
class_count 
