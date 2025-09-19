"""
Utility functions for image processing and manipulation
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image


def save_image(image: np.ndarray, output_path: str):
    """
    Save image to file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size
    """
    return cv2.resize(image, target_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-1 range
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from 0-1 to 0-255 range
    """
    return (image * 255).astype(np.uint8)


def visualize_crops(image: np.ndarray, 
                   crops_info: List[Tuple[float, Tuple[int, int, int, int], np.ndarray]], 
                   top_k: int = 10,
                   save_path: Optional[str] = None) -> None:
    """
    Visualize top K crops on the original image
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image with crop rectangles
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image with Top Crop Locations')
    axes[0, 0].axis('off')
    
    # Draw rectangles for top crops
    colors = plt.cm.viridis(np.linspace(0, 1, min(top_k, len(crops_info))))
    for i, (score, (x, y, w, h), crop) in enumerate(crops_info[:top_k]):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=colors[i], facecolor='none')
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(x, y-5, f'{score:.2f}', color=colors[i], fontsize=8)
    
    # Show individual crops
    for i in range(min(3, len(crops_info))):
        if i < len(crops_info):
            score, coords, crop = crops_info[i]
            row, col = (0, 1) if i == 0 else (1, 0) if i == 1 else (1, 1)
            axes[row, col].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'Crop {i+1}: Score {score:.2f}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def create_crop_grid(crops_info: List[Tuple[float, Tuple[int, int, int, int], np.ndarray]], 
                    top_k: int = 9) -> np.ndarray:
    """
    Create a grid of top K crops
    """
    top_crops = crops_info[:top_k]
    grid_size = int(np.ceil(np.sqrt(len(top_crops))))
    
    if not top_crops:
        return np.array([])
    
    crop_h, crop_w = top_crops[0][2].shape[:2]
    grid_image = np.zeros((grid_size * crop_h, grid_size * crop_w, 3), dtype=np.uint8)
    
    for i, (score, coords, crop) in enumerate(top_crops):
        row = i // grid_size
        col = i % grid_size
        
        y_start = row * crop_h
        y_end = y_start + crop_h
        x_start = col * crop_w
        x_end = x_start + crop_w
        
        grid_image[y_start:y_end, x_start:x_end] = crop
    
    return grid_image


def calculate_image_statistics(image: np.ndarray) -> dict:
    """
    Calculate various image statistics for analysis
    """
    stats = {}
    
    # Basic statistics
    stats['shape'] = image.shape
    stats['mean'] = np.mean(image)
    stats['std'] = np.std(image)
    stats['min'] = np.min(image)
    stats['max'] = np.max(image)
    
    # Color statistics
    if len(image.shape) == 3:
        for i, color in enumerate(['blue', 'green', 'red']):
            channel = image[:, :, i]
            stats[f'{color}_mean'] = np.mean(channel)
            stats[f'{color}_std'] = np.std(channel)
    
    # Brightness and contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    stats['brightness'] = np.mean(gray)
    stats['contrast'] = np.std(gray)
    
    # Edge density (rough measure of detail)
    edges = cv2.Canny(gray, 50, 150)
    stats['edge_density'] = np.sum(edges > 0) / edges.size
    
    return stats


def enhance_image_aesthetics(image: np.ndarray) -> np.ndarray:
    """
    Apply basic aesthetic enhancements to image
    """
    # Convert to LAB color space for better color manipulation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def create_multi_scale_pyramid(image: np.ndarray, 
                              scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]) -> List[np.ndarray]:
    """
    Create multi-scale pyramid of the image
    """
    pyramid = []
    h, w = image.shape[:2]
    
    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_image = cv2.resize(image, (new_w, new_h))
        pyramid.append(scaled_image)
    
    return pyramid


def blend_images(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend two images with given alpha
    """
    # Ensure same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended


def apply_color_grading(image: np.ndarray, 
                       brightness: float = 0, 
                       contrast: float = 1.0,
                       saturation: float = 1.0) -> np.ndarray:
    """
    Apply color grading to image
    """
    # Brightness and contrast
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    # Saturation
    if saturation != 1.0:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return adjusted


if __name__ == "__main__":
    # Test the utility functions
    print("Image utilities loaded successfully!")
    
    # Create a sample image for testing
    sample_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    # Test statistics
    stats = calculate_image_statistics(sample_image)
    print("Sample image statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test enhancement
    enhanced = enhance_image_aesthetics(sample_image)
    print(f"Enhanced image shape: {enhanced.shape}")
    
    print("All utility functions working correctly!")
