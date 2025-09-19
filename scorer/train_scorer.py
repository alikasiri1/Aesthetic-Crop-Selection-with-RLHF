"""
Training script for the Aesthetic Scorer
Since no labeled data is available, this demonstrates the training structure
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from .aesthetic_scorer import AestheticScorer, AestheticScorerPipeline


class SyntheticAestheticDataset(Dataset):
    """
    Synthetic dataset for demonstration
    In practice, you would load your actual aesthetic dataset here
    """
    
    def __init__(self, image_dir: str, annotations_file: str = None, 
                 transform=None, size: int = 1000):
        self.image_dir = image_dir
        self.transform = transform
        self.size = size
        
        # Create synthetic data for demonstration
        self.samples = self._create_synthetic_samples()
        
    def _create_synthetic_samples(self):
        """
        Create synthetic samples for demonstration
        In practice, load from your actual dataset
        """
        samples = []
        
        # Create synthetic images with different aesthetic qualities
        for i in range(self.size):
            # Generate synthetic image
            image = self._generate_synthetic_image(i)
            
            # Generate synthetic score based on image characteristics
            score = self._generate_synthetic_score(image)
            
            samples.append((image, score))
        
        return samples
    
    def _generate_synthetic_image(self, idx: int):
        """Generate a synthetic image with varying aesthetic qualities"""
        # Create different patterns based on index
        if idx % 4 == 0:
            # High aesthetic - smooth gradients
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(224):
                for j in range(224):
                    image[i, j] = [int(255 * i / 224), int(255 * j / 224), 128]
        elif idx % 4 == 1:
            # Medium aesthetic - structured patterns
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            # Add some structure
            image[50:150, 50:150] = [255, 255, 255]
        elif idx % 4 == 2:
            # Low aesthetic - random noise
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            # Very high aesthetic - geometric patterns
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(0, 224, 20):
                for j in range(0, 224, 20):
                    image[i:i+10, j:j+10] = [255, 0, 0]
                    image[i+10:i+20, j+10:j+20] = [0, 255, 0]
        
        return image
    
    def _generate_synthetic_score(self, image: np.ndarray):
        """Generate synthetic aesthetic score based on image characteristics"""
        # Simple heuristic based on image properties
        variance = np.var(image)
        mean_brightness = np.mean(image)
        
        # Higher variance and balanced brightness = higher score
        score = min(10, max(1, (variance / 10000) + (255 - abs(mean_brightness - 127)) / 50))
        
        return score
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, score = self.samples[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(score, dtype=torch.float32)


class AestheticTrainer:
    """
    Trainer class for the aesthetic scorer
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for images, scores in tqdm(dataloader, desc="Training"):
            images, scores = images.to(self.device), scores.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            combined_score, _, _, _ = self.model(images)
            combined_score = combined_score.squeeze()
            
            # Scale scores to match model output
            target_scores = (scores - 1) / 9  # Scale 1-10 to 0-1
            
            loss = self.criterion(combined_score, target_scores)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, scores in tqdm(dataloader, desc="Validating"):
                images, scores = images.to(self.device), scores.to(self.device)
                
                combined_score, _, _, _ = self.model(images)
                combined_score = combined_score.squeeze()
                
                target_scores = (scores - 1) / 9  # Scale 1-10 to 0-1
                loss = self.criterion(combined_score, target_scores)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=50, save_path='scorer/checkpoints'):
        """Full training loop"""
        os.makedirs(save_path, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(save_path, 'best_model.pth'))
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('scorer/training_curves.png')
        plt.show()


def main():
    """Main training function"""
    print("Starting Aesthetic Scorer Training")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = AestheticScorer(backbone_name='efficientnet_b0')
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    train_dataset = SyntheticAestheticDataset(
        image_dir="data/train",  # This would be your actual data directory
        size=2000
    )
    val_dataset = SyntheticAestheticDataset(
        image_dir="data/val",  # This would be your actual data directory
        size=500
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create trainer
    trainer = AestheticTrainer(model, device)
    
    # Train
    trainer.train(train_loader, val_loader, epochs=50)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
