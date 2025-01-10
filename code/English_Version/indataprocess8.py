"""
MNIST Data Preprocessing Module
This code implements data preprocessing for MNIST dataset with the following features:
- Loads and filters MNIST dataset (digits 2-4 only)
- Balances class distribution
- Resizes images from 28x28 to 7x7
- Binarizes images using threshold
- Provides batch processing capabilities
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import cv2
import random

class MNISTDataProcessor:
    """
    MNIST dataset loading and preprocessing class
    1. Uses OpenCV to resize 28x28 images to 7x7
    2. Binarizes images (converts to 0 and 1)
    """
    def __init__(self, batch_size=64):

        """Initialize data processor"""

        self.batch_size = batch_size
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        
        # Filter samples for digits 2-4
        train_indices = []
        test_indices = []
        
        # Count samples for each class in training set
        class_counts = {i: 0 for i in range(2, 5)}  # Only count samples for digits 2,3,4
        for i, (_, label) in enumerate(train_dataset):
            if 1 < label < 5:  # Modified condition to select only 2,3,4
                class_counts[label] += 1
        
        # Find the minimum class count
        min_samples = min(class_counts.values())
        
        # Create separate index lists for each class
        train_class_indices = {i: [] for i in range(2, 5)}  # Changed to 2,3,4
        for i, (_, label) in enumerate(train_dataset):
            if 1 < label < 5:  # Modified condition to select only 2,3,4
                train_class_indices[label].append(i)
        
        # Randomly sample equal numbers from each class
        for label in range(2, 5):  # Changed to 2,3,4
            selected_indices = random.sample(train_class_indices[label], min_samples)
            train_indices.extend(selected_indices)
            # Adjust label values (2,3,4 become 0,1,2)
            for idx in selected_indices:
                train_dataset.targets[idx] = label - 2  # Modified label mapping
        
        # Apply same processing to test set
        test_class_indices = {i: [] for i in range(2, 5)}  # Changed to 2,3,4
        for i, (_, label) in enumerate(test_dataset):
            if 1 < label < 5:  # Modified condition to select only 2,3,4
                test_class_indices[label].append(i)
        
        # Find minimum class count in test set
        test_min_samples = min(len(indices) for indices in test_class_indices.values())
        
        # Randomly sample equal numbers from each test class
        for label in range(2, 5):
            selected_indices = random.sample(test_class_indices[label], test_min_samples)
            test_indices.extend(selected_indices)
            # Adjust label values (2,3,4 become 0,1,2)
            for idx in selected_indices:
                test_dataset.targets[idx] = label - 2
        
        # Create data loaders
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    @staticmethod
    def preprocess_batch(batch): 
        """
        Preprocess MNIST batch data
        1. Uses OpenCV to resize 28x28 images to 7x7
        2. Applies binarization
        """
        images, labels = batch
        batch_size = images.size(0)
    
        images_np = images.squeeze().numpy()
        processed_images = np.zeros((batch_size, 7, 7), dtype=np.float32)
        
        # Use fixed threshold 0.3 (recommended starting point)
        threshold = 0.3  
        
        for i in range(batch_size):
            img = images_np[i]
            # Use INTER_AREA for downsampling to maintain stroke continuity
            resized = cv2.resize(img, (7, 7), interpolation=cv2.INTER_AREA)
            binary = (resized > threshold).astype(np.float32)
            processed_images[i] = binary
        
        return torch.from_numpy(processed_images).view(batch_size, -1), labels
    
    def get_processed_loaders(self):
        """
        Return processed data loaders
        
        """
        class ProcessedDataLoader:
            def __init__(self, original_loader):
                self.loader = original_loader
                self.batch_size = original_loader.batch_size
            
            def __iter__(self):
                for batch in self.loader:
                    yield MNISTDataProcessor.preprocess_batch(batch)
                    
            def __len__(self):
                return len(self.loader)
        
        return (ProcessedDataLoader(self.train_loader),
                ProcessedDataLoader(self.test_loader))

def verify_preprocessing(processor):
    # Get one batch of data
    for batch, labels in processor.train_loader:
        processed, _ = processor.preprocess_batch((batch, labels))
        # Commented out all print statements
        break

def main():
    # Test data loading and preprocessing
    processor = MNISTDataProcessor(batch_size=32)
    processed_train_loader, processed_test_loader = processor.get_processed_loaders()
    
    # Get and print first batch statistics
    for data, targets in processed_train_loader:
        processor.print_batch_statistics(data, targets)
        break

if __name__ == "__main__":
    processor = MNISTDataProcessor(batch_size=32)
    verify_preprocessing(processor)