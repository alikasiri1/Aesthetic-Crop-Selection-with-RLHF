#!/usr/bin/env python3
"""
Example script showing how to run training with grayscale mode
"""

import subprocess
import sys
import os

def run_training_with_gray_mode():
    """Run PPO training with grayscale mode enabled"""
    
    # Change to the actor directory
    actor_dir = os.path.join(os.path.dirname(__file__), "actor")
    
    # Command to run training with grayscale mode
    cmd = [
        sys.executable, "train_ppo.py",
        "--image", "../image.jpg",  # Adjust path as needed
        "--timesteps", "1000",      # Reduced for testing
        "--save_dir", "../checkpoints_gray",
        "--downscale", "128", "128",
        "--init_crop", "224", "224",
        "--gray_mode"  # This enables grayscale processing
    ]
    
    print("Running PPO training with grayscale mode...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {actor_dir}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, cwd=actor_dir, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("Error output:", e.stderr)
    except FileNotFoundError:
        print("Error: Could not find the training script. Make sure you're in the correct directory.")

def run_training_with_rgb_mode():
    """Run PPO training with RGB mode (default)"""
    
    # Change to the actor directory
    actor_dir = os.path.join(os.path.dirname(__file__), "actor")
    
    # Command to run training with RGB mode (default)
    cmd = [
        sys.executable, "train_ppo.py",
        "--image", "../image.jpg",  # Adjust path as needed
        "--timesteps", "1000",      # Reduced for testing
        "--save_dir", "../checkpoints_rgb",
        "--downscale", "128", "128",
        "--init_crop", "224", "224"
        # No --gray_mode flag means RGB mode
    ]
    
    print("Running PPO training with RGB mode...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {actor_dir}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, cwd=actor_dir, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("Error output:", e.stderr)
    except FileNotFoundError:
        print("Error: Could not find the training script. Make sure you're in the correct directory.")

if __name__ == "__main__":
    print("Choose training mode:")
    print("1. RGB mode (default)")
    print("2. Grayscale mode")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_training_with_rgb_mode()
    elif choice == "2":
        run_training_with_gray_mode()
    else:
        print("Invalid choice. Please run again and choose 1 or 2.")
