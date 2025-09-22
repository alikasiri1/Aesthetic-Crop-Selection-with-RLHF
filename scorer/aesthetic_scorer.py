"""
Aesthetic Scorer - Neural network for beauty scoring of image crops
Uses pre-trained models since no labeled data is available
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional
import os


class AestheticScorer(nn.Module):
    """
    Aesthetic scoring model that combines multiple pre-trained features
    for beauty assessment of image crops
    """
    
    def __init__(self, backbone_name: str = 'efficientnet_b0', num_classes: int = 1, gray_mode: bool = False):
        super(AestheticScorer, self).__init__()
        
        self.gray_mode = gray_mode
        
        # Use pre-trained backbone
        # if backbone_name.startswith('efficientnet'):
        #     # For grayscale, we need to modify the first layer to accept 1 channel instead of 3
        #     self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        #     if gray_mode:
        #         # Modify first conv layer to accept 1 channel input
        #         original_conv = self.backbone.conv_stem
        #         self.backbone.conv_stem = nn.Conv2d(1, original_conv.out_channels, 
        #                                            kernel_size=original_conv.kernel_size,
        #                                            stride=original_conv.stride,
        #                                            padding=original_conv.padding,
        #                                            bias=original_conv.bias is not None)
        #         # Initialize with average of RGB weights
        #         with torch.no_grad():
        #             self.backbone.conv_stem.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        #     feature_dim = self.backbone.num_features
        # elif backbone_name.startswith('resnet'):
        #     self.backbone = models.__dict__[backbone_name](pretrained=True)
        #     if gray_mode:
        #         # Modify first conv layer to accept 1 channel input
        #         original_conv = self.backbone.conv1
        #         self.backbone.conv1 = nn.Conv2d(1, original_conv.out_channels, 
        #                                        kernel_size=original_conv.kernel_size,
        #                                        stride=original_conv.stride,
        #                                        padding=original_conv.padding,
        #                                        bias=original_conv.bias is not None)
        #         # Initialize with average of RGB weights
        #         with torch.no_grad():
        #             self.backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        #     self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final FC
        #     feature_dim = 2048 if '50' in backbone_name else 512
        # else:
        #     raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, in_chans=1 if gray_mode else 3)
        feature_dim = self.backbone.num_features

        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Aesthetic scoring head
        self.aesthetic_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Composition analysis head
        self.composition_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Color harmony head
        self.color_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
            
        # Multiple scoring heads
        aesthetic_score = self.aesthetic_head(features)
        composition_score = self.composition_head(features)
        color_score = self.color_head(features)
        
        # Combine scores (weighted average)
        combined_score = (aesthetic_score * 0.6 + 
                         composition_score * 0.25 + 
                         color_score * 0.15)
        
        return combined_score, aesthetic_score, composition_score, color_score


class ImageCropper:
    """
    Utility class for creating sliding window crops from large images
    """
    
    def __init__(self, crop_size: Tuple[int, int] = (224, 224), 
                 stride: int = 112, min_overlap: float = 0.1):
        self.crop_size = crop_size
        self.stride = stride
        self.min_overlap = min_overlap
        
    def create_crops(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Create sliding window crops from image
        Returns list of (crop, (x, y, w, h))
        """
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size
        
        crops = []
        
        # Calculate stride to ensure minimum overlap
        stride_h = max(1, int(crop_h * (1 - self.min_overlap)))
        stride_w = max(1, int(crop_w * (1 - self.min_overlap)))
        
        for y in range(0, h - crop_h + 1, stride_h):
            for x in range(0, w - crop_w + 1, stride_w):
                # Ensure we don't go out of bounds
                end_y = min(y + crop_h, h)
                end_x = min(x + crop_w, w)
                start_y = max(0, end_y - crop_h)
                start_x = max(0, end_x - crop_w)
                
                crop = image[start_y:end_y, start_x:end_x]
                
                # Resize if necessary
                if crop.shape[:2] != self.crop_size:
                    crop = cv2.resize(crop, self.crop_size)
                
                crops.append((crop, (start_x, start_y, end_x - start_x, end_y - start_y)))
        
        return crops
    
    def create_multi_scale_crops(self, image: np.ndarray, 
                                scales: List[float] = [0.5, 0.75, 1.0, 1.25]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Create crops at multiple scales
        """
        all_crops = []
        
        for scale in scales:
            # Resize image
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h))
            
            # Create crops at this scale
            crops = self.create_crops(scaled_image)
            
            # Adjust coordinates back to original scale
            for crop, (x, y, w, h) in crops:
                orig_coords = (int(x / scale), int(y / scale), 
                              int(w / scale), int(h / scale))
                all_crops.append((crop, orig_coords))
        
        return all_crops


class AestheticScorerPipeline:
    """
    Complete pipeline for aesthetic scoring of image crops
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 backbone_name: str = 'efficientnet_b0',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 gray_mode: bool = False):
        
        self.device = device
        self.gray_mode = gray_mode
        self.model = AestheticScorer(backbone_name=backbone_name, gray_mode=gray_mode).to(device)
        
        # Image preprocessing
        if gray_mode:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with pre-trained weights (no fine-tuning for now)
            self._initialize_pretrained_weights()
    
    def _initialize_pretrained_weights(self):
        """
        Initialize aesthetic scoring with pre-trained aesthetic models
        """
        # For now, we'll use the pre-trained backbone features
        # In a real scenario, you'd fine-tune on aesthetic datasets
        print("Using pre-trained backbone features for aesthetic scoring")
        
    def load_model(self, model_path: str):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    
    def save_model(self, model_path: str, epoch: int = 0, loss: float = 0.0):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'loss': loss
        }, model_path)
        print(f"Saved model to {model_path}")
    
    def score_image(self, image: np.ndarray) -> float:
        """
        Score a single image crop
        """
        self.model.eval()
        with torch.no_grad():
            # Preprocess image
            if self.gray_mode:
                # For grayscale mode, ensure we have a single channel image
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Convert BGR to grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif len(image.shape) == 2:
                    # Already grayscale
                    pass
                else:
                    raise ValueError(f"Unexpected image shape for grayscale mode: {image.shape}")
            else:
                # RGB mode
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get score
            combined_score, _, _, _ = self.model(tensor)
            score = torch.sigmoid(combined_score).item()  # Convert to 0-1 range
            
            # Scale to 1-10 range
            return score * 9 + 1
    
    def score_crops(self, image: np.ndarray, 
                   cropper: ImageCropper) -> List[Tuple[float, Tuple[int, int, int, int], np.ndarray]]:
        """
        Score all crops from an image
        Returns list of (score, coordinates, crop)
        """
        crops = cropper.create_crops(image)
        print(f"Created {len(crops)} crops")
        scored_crops = []
        
        for crop, coords in crops:
            score = self.score_image(crop)
            scored_crops.append((score, coords, crop))
        
        return scored_crops
    
    def get_top_crops(self, image: np.ndarray, 
                     cropper: ImageCropper, 
                     top_k: int = 10) -> List[Tuple[float, Tuple[int, int, int, int], np.ndarray]]:
        """
        Get top K highest scoring crops
        """
        scored_crops = self.score_crops(image, cropper)
        
        # Sort by score (descending)
        scored_crops.sort(key=lambda x: x[0], reverse=True)
        
        return scored_crops[:top_k]


def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    Since no labeled data is available, we'll create synthetic examples
    """
    # This would typically load your actual dataset
    # For now, we'll return empty lists
    return [], []


if __name__ == "__main__":
    # Example usage
    print("Aesthetic Scorer - Step 1 Implementation")
    
    # Initialize scorer
    scorer = AestheticScorerPipeline()
    
    # Create sample image (you would load your actual image here)
    sample_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    
    # Create cropper
    cropper = ImageCropper(crop_size=(224, 224), stride=112)
    
    # Score crops
    top_crops = scorer.get_top_crops(sample_image, cropper, top_k=5)
    
    print(f"Found {len(top_crops)} top crops:")
    for i, (score, coords, crop) in enumerate(top_crops):
        print(f"Crop {i+1}: Score={score:.2f}, Coords={coords}")
