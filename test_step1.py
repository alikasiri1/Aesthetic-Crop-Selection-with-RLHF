"""
Test script for Step 1 - Aesthetic Scorer
Demonstrates the functionality of the aesthetic scoring system
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scorer.aesthetic_scorer import AestheticScorerPipeline, ImageCropper
from utils.image_utils import visualize_crops, create_crop_grid, save_image


def create_sample_images():
    """
    Create sample images with different aesthetic qualities for testing
    """
    images = {}
    
    # High aesthetic - smooth gradient
    gradient = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(400):
        for j in range(400):
            gradient[i, j] = [int(255 * i / 400), int(255 * j / 400), 128]
    images['gradient'] = gradient
    
    # Medium aesthetic - structured pattern
    pattern = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    # Add some geometric structure
    cv2.rectangle(pattern, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(pattern, (200, 200), 50, (0, 0, 0), -1)
    images['pattern'] = pattern
    
    # Low aesthetic - random noise
    noise = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    images['noise'] = noise
    
    # Very high aesthetic - artistic composition
    artistic = np.zeros((400, 400, 3), dtype=np.uint8)
    # Create a sunset-like gradient
    for i in range(400):
        for j in range(400):
            artistic[i, j] = [int(255 * (1 - i / 400)), 
                             int(255 * (1 - i / 400) * 0.5), 
                             int(255 * (1 - i / 400) * 0.2)]
    # Add some elements
    cv2.circle(artistic, (200, 300), 30, (255, 255, 100), -1)  # Sun
    cv2.rectangle(artistic, (0, 350), (400, 400), (50, 50, 50), -1)  # Ground
    images['artistic'] = artistic
    
    return images


def test_single_image_scoring():
    """
    Test scoring of individual images
    """
    print("=== Testing Single Image Scoring ===")
    
    # Initialize scorer
    scorer = AestheticScorerPipeline()
    
    # Create sample images
    sample_images = create_sample_images()
    
    print("\nScoring individual images:")
    for name, image in sample_images.items():
        score = scorer.score_image(image)
        print(f"  {name}: {score:.2f}/10")
        
        # Save sample images
        os.makedirs('results/sample_images', exist_ok=True)
        save_image(image, f'results/sample_images/{name}.jpg')


def test_sliding_window_scoring():
    """
    Test sliding window scoring on a large image
    """
    print("\n=== Testing Sliding Window Scoring ===")
    
    # Initialize scorer and cropper
    scorer = AestheticScorerPipeline()
    cropper = ImageCropper(crop_size=(448, 448), stride=112)
    
    # Create a large composite image
    # large_image = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # # Add different regions with varying aesthetics
    # sample_images = create_sample_images()
    
    # # Place images in different quadrants
    # large_image[0:400, 0:400] = sample_images['artistic']  # Top-left: high aesthetic
    # large_image[0:400, 400:800] = sample_images['gradient']  # Top-right: medium-high
    # large_image[400:800, 0:400] = sample_images['pattern']  # Bottom-left: medium
    # large_image[400:800, 400:800] = sample_images['noise']  # Bottom-right: low
    
    # # Save the large image
    # os.makedirs('results', exist_ok=True)
    # save_image(large_image, 'results/large_test_image.jpg')
    
    # print("Created large test image (800x800)")
    print("Performing sliding window scoring...")
###############################
    print("Loading image from disk...")
    image_path = "image.jpg" 
    large_image = cv2.imread(image_path)
##############################
    # Get top crops
    top_crops = scorer.get_top_crops(large_image, cropper, top_k=10)
    
    print(f"\nTop 10 crops found:")
    for i, (score, coords, crop) in enumerate(top_crops):
        x, y, w, h = coords
        print(f"  {i+1}. Score: {score:.2f}, Coords: ({x}, {y}, {w}, {h})")
        
        # Save individual crops
        os.makedirs('results/top_crops', exist_ok=True)
        save_image(crop, f'results/top_crops/crop_{i+1}_score_{score:.2f}.jpg')
    
    # Visualize results
    visualize_crops(large_image, top_crops, top_k=10, 
                   save_path='results/crop_visualization.png')
    
    # Create crop grid
    crop_grid = create_crop_grid(top_crops, top_k=9)
    if crop_grid.size > 0:
        save_image(crop_grid, 'results/top_crops_grid.jpg')
        print("Saved crop grid visualization")
    
    return top_crops


def test_multi_scale_scoring():
    """
    Test multi-scale cropping and scoring
    """
    print("\n=== Testing Multi-Scale Scoring ===")
    
    # Initialize scorer and cropper
    scorer = AestheticScorerPipeline()
    cropper = ImageCropper(crop_size=(224, 224), stride=112)
    
    # Create a test image
    test_image = create_sample_images()['artistic']
    
    # Test multi-scale cropping
    multi_scale_crops = cropper.create_multi_scale_crops(
        test_image, scales=[0.5, 0.75, 1.0, 1.25]
    )
    
    print(f"Created {len(multi_scale_crops)} multi-scale crops")
    
    # Score all crops
    scored_crops = []
    for crop, coords in multi_scale_crops:
        score = scorer.score_image(crop)
        scored_crops.append((score, coords, crop))
    
    # Sort by score
    scored_crops.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\nTop 5 multi-scale crops:")
    for i, (score, coords, crop) in enumerate(scored_crops[:5]):
        x, y, w, h = coords
        print(f"  {i+1}. Score: {score:.2f}, Coords: ({x}, {y}, {w}, {h})")
    
    return scored_crops



def analyze_scoring_performance():
    """
    Analyze the performance of the scoring system
    """
    print("\n=== Analyzing Scoring Performance ===")
    
    scorer = AestheticScorerPipeline()
    sample_images = create_sample_images()
    
    # Expected order (from high to low aesthetic)
    expected_order = ['artistic', 'gradient', 'pattern', 'noise']
    
    # Score all images
    scores = {}
    for name, image in sample_images.items():
        score = scorer.score_image(image)
        scores[name] = score
    
    print("\nActual scores:")
    for name in expected_order:
        print(f"  {name}: {scores[name]:.2f}")
    
    # Check if scores follow expected order
    actual_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    actual_names = [name for name, score in actual_order]
    
    print(f"\nActual ranking: {actual_names}")
    print(f"Expected ranking: {expected_order}")
    
    # Calculate ranking accuracy
    correct_positions = sum(1 for i, name in enumerate(actual_names) 
                          if name == expected_order[i])
    accuracy = correct_positions / len(expected_order)
    
    print(f"Ranking accuracy: {accuracy:.2%}")


def main():
    """
    Main test function
    """
    print("Step 1 - Aesthetic Scorer Test")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Test individual image scoring
        # test_single_image_scoring()
        
        # Test sliding window scoring
        top_crops = test_sliding_window_scoring()
        
        # Test multi-scale scoring
        # multi_scale_crops = test_multi_scale_scoring()
        
        # Analyze performance
        # analyze_scoring_performance()
        
        print("\n" + "=" * 50)
        print("Step 1 Testing Completed Successfully!")
        print("\nResults saved in 'results/' directory:")
        print("  - sample_images/: Individual test images")
        print("  - large_test_image.jpg: Large composite test image")
        print("  - top_crops/: Individual high-scoring crops")
        print("  - crop_visualization.png: Visualization of crop locations")
        print("  - top_crops_grid.jpg: Grid of top crops")
        
        print(f"\nFound {len(top_crops)} high-quality crops from sliding window")
        # print(f"Found {len(multi_scale_crops)} crops from multi-scale analysis")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
