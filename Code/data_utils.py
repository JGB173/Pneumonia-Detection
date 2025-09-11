# Josh Burgess
# 300652214
# Data utility functions for chest X-ray preprocessing

import os
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image

def create_splits():
    
    # All training images
    train_normal = "chest_xray/train/NORMAL"
    train_pneumonia = "chest_xray/train/PNEUMONIA"
    val_normal = "chest_xray/val/NORMAL" 
    val_pneumonia = "chest_xray/val/PNEUMONIA"
    
    # File paths
    normal_files = []
    pneumonia_files = []
    
    # Add og train files
    for f in os.listdir(train_normal):
        normal_files.append(os.path.join(train_normal, f))

    for f in os.listdir(train_pneumonia):
        pneumonia_files.append(os.path.join(train_pneumonia, f))
        
    # Add og val files  
    for f in os.listdir(val_normal):
        normal_files.append(os.path.join(val_normal, f))

    for f in os.listdir(val_pneumonia):
        pneumonia_files.append(os.path.join(val_pneumonia, f))
    
    return normal_files, pneumonia_files

def get_train_val_splits(normal_files, pneumonia_files, test_size=0.2, random_state=42):
    
    # Split normal and pneumonia files into train/val sets
    normal_train, normal_val = train_test_split(normal_files, test_size=test_size, random_state=random_state)
    pneumonia_train, pneumonia_val = train_test_split(pneumonia_files, test_size=test_size, random_state=random_state)
    
    return normal_train, normal_val, pneumonia_train, pneumonia_val

"""
Preprocesses image: resize + padding to square, convert to tensor
Maintains aspect ratio and preserves anatomical information
"""

def preprocess_image(image_path, target_size=224):

    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB 
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get og dimensions
    original_width, original_height = img.size
    
    # Calc new size maintaining aspect ratio
    if original_width > original_height:
        new_width = target_size
        new_height = int((target_size * original_height) / original_width)
        
    else:
        new_height = target_size
        new_width = int((target_size * original_width) / original_height)
    
    # Resize maintaining aspect ratio
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new square image with black padding
    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    
    # Center the resized image
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    new_img.paste(img, (x_offset, y_offset))
    
    return new_img

# Preprocessing pipeline with augmentation
def get_transforms():
    # Returns train and val transforms
    
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: preprocess_image(x) if isinstance(x, str) else x),
        transforms.RandomRotation(10),  # ±10° rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # ±20% brightness/contrast
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Validation transforms (no augmentation to provide consistent repeatable evaluation)
    val_transform = transforms.Compose([
        transforms.Lambda(lambda x: preprocess_image(x) if isinstance(x, str) else x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Create 500-image subset for testing
def create_subset(normal_files, pneumonia_files, total_size=500):
    
    # Calc proportional sizes (25% normal, 75% pneumonia)
    normal_subset_size = int(total_size * 0.25)  
    pneumonia_subset_size = total_size - normal_subset_size
    
    normal_subset = normal_files[:normal_subset_size]
    pneumonia_subset = pneumonia_files[:pneumonia_subset_size]
    
    return normal_subset, pneumonia_subset