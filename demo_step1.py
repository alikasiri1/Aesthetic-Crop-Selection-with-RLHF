"""
Demo script for Step 1 - Aesthetic Scorer
Simple demonstration of how to use the aesthetic scoring system
"""

"""
Demo script for Step 1 - Aesthetic Scorer
Simple demonstration of how to use the aesthetic scoring system
"""

import numpy as np
import cv2
from scorer import AestheticScorerPipeline, ImageCropper
from utils import visualize_crops, save_image


def quick_demo():
    """
    Quick demonstration of the aesthetic scorer
    """
    print("Step 1 Demo - Aesthetic Scorer")
    print("=" * 40)
    
    # 1. Initialize the scorer
    print("Initializing aesthetic scorer...")
    scorer = AestheticScorerPipeline()
    
    # 2. Load an image from your computer
    print("Loading image from disk...")
    image_path = "image.jpg" 
    sample_image = cv2.imread(image_path)
    
    if sample_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # 3. Create cropper for sliding window
    print("Setting up sliding window cropper...")
    cropper = ImageCropper(crop_size=(224, 224), stride=112)
    
    # 4. Score all crops and get top 5
    print("Scoring crops and finding top 5...")
    top_crops = scorer.get_top_crops(sample_image, cropper, top_k=5)
    
    # 5. Display results
    print(f"\nTop 5 crops found:")
    for i, (score, coords, crop) in enumerate(top_crops):
        x, y, w, h = coords
        print(f"  {i+1}. Score: {score:.2f}/10, Position: ({x}, {y})")
    
    # 6. Save results
    print("\nSaving results...")
    save_image(sample_image, 'demo_results/sample_image.jpg')
    
    for i, (score, coords, crop) in enumerate(top_crops):
        save_image(crop, f'demo_results/top_crop_{i+1}.jpg')
    
    # 7. Visualize
    print("Creating visualization...")
    visualize_crops(sample_image, top_crops, top_k=5, 
                   save_path='demo_results/visualization.png')
    
    print("\nDemo completed! Check 'demo_results/' folder for outputs.")


if __name__ == "__main__":
    import os
    os.makedirs('demo_results', exist_ok=True)
    quick_demo()

