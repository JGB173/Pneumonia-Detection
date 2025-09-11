# Josh Burgess
# 300652214
# PyTorch dataset class for chest X-ray pneumonia detection

import torch
from torch.utils.data import Dataset, DataLoader
from data_utils import preprocess_image

"""
Custom dataset for chest X-ray images
normal_files: List of file paths to normal X-ray images
pneumonia_files: List of file paths to pneumonia X-ray images  
transform: PyTorch transforms to apply to images
"""

class ChestXrayDataset(Dataset):
    
    def __init__(self, normal_files, pneumonia_files, transform=None):
    
        # Combine file paths and create corresponding labels
        # Creates dataset that is unified
        self.image_paths = normal_files + pneumonia_files
        self.labels = [0] * len(normal_files) + [1] * len(pneumonia_files)  # 0=Normal, 1=Pneumonia
        self.transform = transform
    
    def __len__(self):
        # Return total number of images
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get a single image and its label
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = preprocess_image(image_path)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label



"""
Create PyTorch data loaders for training and validation

normal_train, normal_val: Lists of normal image file paths
pneumonia_train, pneumonia_val: Lists of pneumonia image file paths
train_transform, val_transform: PyTorch transform pipelines
batch_size: Number of images per batch
num_workers: Number of worker processes for data loading

Returns:
train_loader, val_loader: PyTorch DataLoader objects
"""
def create_data_loaders(normal_train, normal_val, pneumonia_train, pneumonia_val, train_transform, val_transform, batch_size=32, num_workers=0):
    
    # Create datasets
    train_dataset = ChestXrayDataset(normal_train, pneumonia_train, transform=train_transform)
    val_dataset = ChestXrayDataset(normal_val, pneumonia_val, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle training data for better learning
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers
    )
    
    return train_loader, val_loader


"""
Create a data loader for a subset of the data (for testing)

normal_files, pneumonia_files: Lists of image file paths
transform: PyTorch transform pipeline
total_size: Total number of images in subset
batch_size: Number of images per batch

Returns:
subset_loader: PyTorch DataLoader for the subset
"""
def create_subset_loader(normal_files, pneumonia_files, transform, total_size=500, batch_size=32):
 
    from data_utils import create_subset
    
    # Create subset maintaining class proportions
    normal_subset, pneumonia_subset = create_subset(normal_files, pneumonia_files, total_size)
    
    # Create dataset and loader
    subset_dataset = ChestXrayDataset(normal_subset, pneumonia_subset, transform=transform)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return subset_loader