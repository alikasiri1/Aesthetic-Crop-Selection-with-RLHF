#!/usr/bin/env python3
"""
Test script to demonstrate grayscale mode functionality in step6_incremental_improvement.py
"""

import os
import sys
import subprocess

def test_step6_gray_mode():
    """Test step6 incremental improvement with grayscale mode"""
    
    # Test image path (you can change this to your image)
    image_path = "image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please place an image named 'image.jpg' in the current directory")
        return
    
    print("Testing Step 6 Incremental Improvement with Grayscale Mode")
    print("=" * 60)
    
    # Test Stage 1: Bootstrap with grayscale mode
    print("\n1. Testing Stage 1 (Bootstrap) with Grayscale Mode...")
    cmd_stage1 = [
        sys.executable, "step6_incremental_improvement.py",
        "--stage", "1",
        "--image", image_path,
        "--timesteps", "1000",  # Reduced for testing
        "--gray_mode"
    ]
    
    try:
        result = subprocess.run(cmd_stage1, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ Stage 1 (Bootstrap) with grayscale mode completed successfully")
            print("Output:", result.stdout[-200:])  # Last 200 chars
        else:
            print("✗ Stage 1 failed")
            print("Error:", result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Stage 1 timed out (this is expected for a quick test)")
    except Exception as e:
        print(f"✗ Stage 1 failed with error: {e}")
    
    # Test Stage 2: Generate preference pairs with grayscale mode
    print("\n2. Testing Stage 2 (Generate Preferences) with Grayscale Mode...")
    cmd_stage2_gen = [
        sys.executable, "step6_incremental_improvement.py",
        "--stage", "2",
        "--image", image_path,
        "--num_pairs", "5",  # Small number for testing
        "--gray_mode"
    ]
    
    try:
        result = subprocess.run(cmd_stage2_gen, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ Stage 2 (Generate Preferences) with grayscale mode completed successfully")
            print("Output:", result.stdout[-200:])
        else:
            print("✗ Stage 2 (Generate Preferences) failed")
            print("Error:", result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Stage 2 (Generate Preferences) timed out")
    except Exception as e:
        print(f"✗ Stage 2 (Generate Preferences) failed with error: {e}")
    
    print("\n" + "=" * 60)
    print("Grayscale mode test completed!")
    print("\nTo run the full pipeline with grayscale mode:")
    print("1. python step6_incremental_improvement.py --stage 1 --image image.jpg --gray_mode")
    print("2. python step6_incremental_improvement.py --stage 2 --image image.jpg --gray_mode")
    print("3. # Annotate preferences manually")
    print("4. python step6_incremental_improvement.py --stage 2 --preference_data results/step6/preference_pairs.json --gray_mode")
    print("5. python step6_incremental_improvement.py --stage 3 --image image.jpg --human_reward_model results/step6/human_reward_model.pth --gray_mode")

def test_step6_rgb_mode():
    """Test step6 incremental improvement with RGB mode (default)"""
    
    # Test image path (you can change this to your image)
    image_path = "image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please place an image named 'image.jpg' in the current directory")
        return
    
    print("Testing Step 6 Incremental Improvement with RGB Mode (Default)")
    print("=" * 60)
    
    # Test Stage 1: Bootstrap with RGB mode
    print("\n1. Testing Stage 1 (Bootstrap) with RGB Mode...")
    cmd_stage1 = [
        sys.executable, "step6_incremental_improvement.py",
        "--stage", "1",
        "--image", image_path,
        "--timesteps", "1000",  # Reduced for testing
        # No --gray_mode flag means RGB mode
    ]
    
    try:
        result = subprocess.run(cmd_stage1, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ Stage 1 (Bootstrap) with RGB mode completed successfully")
            print("Output:", result.stdout[-200:])  # Last 200 chars
        else:
            print("✗ Stage 1 failed")
            print("Error:", result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Stage 1 timed out (this is expected for a quick test)")
    except Exception as e:
        print(f"✗ Stage 1 failed with error: {e}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. RGB mode (default)")
    print("2. Grayscale mode")
    print("3. Both modes")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        test_step6_rgb_mode()
    elif choice == "2":
        test_step6_gray_mode()
    elif choice == "3":
        test_step6_rgb_mode()
        print("\n" + "=" * 60)
        test_step6_gray_mode()
    else:
        print("Invalid choice. Please run again and choose 1, 2, or 3.")
