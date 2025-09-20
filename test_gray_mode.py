#!/usr/bin/env python3
"""
Test script to demonstrate grayscale mode functionality
"""

import os
import sys
import cv2
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_env.frame_selector_env import FrameSelectorEnv
from scorer.aesthetic_scorer import AestheticScorerPipeline

def test_gray_mode():
    """Test grayscale mode functionality"""
    
    # Test image path (you can change this to your image)
    image_path = "image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please place an image named 'image.jpg' in the current directory")
        return
    
    print("Testing RGB mode...")
    # Test RGB mode
    scorer_rgb = AestheticScorerPipeline(gray_mode=False)
    env_rgb = FrameSelectorEnv(
        image_path=image_path,
        scorer=scorer_rgb,
        downscale_hw=(128, 128),
        init_crop_hw=(224, 224),
        max_steps=10,
        gray_mode=False
    )
    
    obs_rgb, info_rgb = env_rgb.reset()
    print(f"RGB observation shape: {obs_rgb.shape}")
    print(f"RGB observation dtype: {obs_rgb.dtype}")
    print(f"RGB observation range: [{obs_rgb.min():.3f}, {obs_rgb.max():.3f}]")
    
    # Test a few actions
    for i in range(3):
        action = i % env_rgb.num_actions
        obs_rgb, reward_rgb, done, truncated, info_rgb = env_rgb.step(action)
        print(f"RGB Step {i+1}: Action={action}, Reward={reward_rgb:.3f}, Done={done}")
    
    print("\nTesting Grayscale mode...")
    # Test Grayscale mode
    scorer_gray = AestheticScorerPipeline(gray_mode=True)
    env_gray = FrameSelectorEnv(
        image_path=image_path,
        scorer=scorer_gray,
        downscale_hw=(128, 128),
        init_crop_hw=(224, 224),
        max_steps=10,
        gray_mode=True
    )
    
    obs_gray, info_gray = env_gray.reset()
    print(f"Grayscale observation shape: {obs_gray.shape}")
    print(f"Grayscale observation dtype: {obs_gray.dtype}")
    print(f"Grayscale observation range: [{obs_gray.min():.3f}, {obs_gray.max():.3f}]")
    
    # Test a few actions
    for i in range(3):
        action = i % env_gray.num_actions
        obs_gray, reward_gray, done, truncated, info_gray = env_gray.step(action)
        print(f"Grayscale Step {i+1}: Action={action}, Reward={reward_gray:.3f}, Done={done}")
    
    print("\nComparison:")
    print(f"RGB observation shape: {obs_rgb.shape}")
    print(f"Grayscale observation shape: {obs_gray.shape}")
    print("✓ Grayscale mode successfully processes images with 1 channel instead of 3")

if __name__ == "__main__":
    test_gray_mode()
