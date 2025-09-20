#!/usr/bin/env python3
"""
Test script to demonstrate grayscale mode functionality in step5_combine_eval.py
"""

import os
import sys
import subprocess

def test_step5_gray_mode():
    """Test step5 combine and evaluate with grayscale mode"""
    
    # Test image path (you can change this to your image)
    image_path = "image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please place an image named 'image.jpg' in the current directory")
        return
    
    # Check if actor model exists
    actor_model = "Aesthetic-Crop-Selection-with-RLHF/actor/checkpoints/ppo_frame_selector.zip"
    if not os.path.exists(actor_model):
        print(f"Actor model not found: {actor_model}")
        print("Please train the actor first using: python actor/train_ppo.py")
        return
    
    print("Testing Step 5 Combine and Evaluate with Grayscale Mode")
    print("=" * 60)
    
    # Test RGB mode (default)
    print("\n1. Testing RGB Mode (Default)...")
    cmd_rgb = [
        sys.executable, "step5_combine_eval.py",
        "--image", image_path,
        "--actor_model", actor_model,
        "--output_dir", "results/step5_rgb",
        "--num_episodes", "3",  # Reduced for testing
        "--max_steps", "50"     # Reduced for testing
        # No --gray_mode flag means RGB mode
    ]
    
    try:
        result = subprocess.run(cmd_rgb, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✓ RGB mode completed successfully")
            print("Output:", result.stdout[-200:])  # Last 200 chars
        else:
            print("✗ RGB mode failed")
            print("Error:", result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ RGB mode timed out")
    except Exception as e:
        print(f"✗ RGB mode failed with error: {e}")
    
    # Test Grayscale mode
    print("\n2. Testing Grayscale Mode...")
    cmd_gray = [
        sys.executable, "step5_combine_eval.py",
        "--image", image_path,
        "--actor_model", actor_model,
        "--output_dir", "results/step5_gray",
        "--num_episodes", "3",  # Reduced for testing
        "--max_steps", "50",    # Reduced for testing
        "--gray_mode"
    ]
    
    try:
        result = subprocess.run(cmd_gray, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✓ Grayscale mode completed successfully")
            print("Output:", result.stdout[-200:])  # Last 200 chars
        else:
            print("✗ Grayscale mode failed")
            print("Error:", result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Grayscale mode timed out")
    except Exception as e:
        print(f"✗ Grayscale mode failed with error: {e}")
    
    print("\n" + "=" * 60)
    print("Grayscale mode test completed!")
    print("\nTo run step5 with grayscale mode:")
    print("python step5_combine_eval.py --image image.jpg --gray_mode")
    print("\nTo run step5 with RGB mode (default):")
    print("python step5_combine_eval.py --image image.jpg")

def test_step5_with_trained_gray_model():
    """Test step5 with a model trained in grayscale mode"""
    
    # Test image path
    image_path = "image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Check for grayscale-trained model
    gray_actor_model = "Aesthetic-Crop-Selection-with-RLHF/actor/checkpoints_gray/ppo_frame_selector.zip"
    if not os.path.exists(gray_actor_model):
        print(f"Grayscale-trained model not found: {gray_actor_model}")
        print("Please train a model in grayscale mode first:")
        print("python actor/train_ppo.py --image image.jpg --gray_mode --save_dir checkpoints_gray")
        return
    
    print("Testing Step 5 with Grayscale-Trained Model")
    print("=" * 50)
    
    cmd = [
        sys.executable, "step5_combine_eval.py",
        "--image", image_path,
        "--actor_model", gray_actor_model,
        "--output_dir", "results/step5_gray_trained",
        "--num_episodes", "3",
        "--max_steps", "50",
        "--gray_mode"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✓ Grayscale-trained model test completed successfully")
            print("Output:", result.stdout[-200:])
        else:
            print("✗ Grayscale-trained model test failed")
            print("Error:", result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Grayscale-trained model test timed out")
    except Exception as e:
        print(f"✗ Grayscale-trained model test failed with error: {e}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test both RGB and Grayscale modes")
    print("2. Test with grayscale-trained model")
    print("3. Both tests")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        test_step5_gray_mode()
    elif choice == "2":
        test_step5_with_trained_gray_model()
    elif choice == "3":
        test_step5_gray_mode()
        print("\n" + "=" * 60)
        test_step5_with_trained_gray_model()
    else:
        print("Invalid choice. Please run again and choose 1, 2, or 3.")
