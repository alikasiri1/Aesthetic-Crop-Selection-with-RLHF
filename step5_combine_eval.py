"""
Step 5 – Combine and Evaluate

Loads the trained Scorer and Actor models, runs them on a new image,
gets suggested Actor frames, reports Scorer scores, and plots results
with suggested rectangles on the large image.

Usage:
  # RGB mode (default):
  python step5_combine_eval.py \
    --image path/to/image.jpg \
    --actor_model actor/checkpoints/ppo_frame_selector.zip \
    --output_dir results/step5 \
    --num_episodes 5

  # Grayscale mode:
  python step5_combine_eval.py \
    --image path/to/image.jpg \
    --actor_model actor/checkpoints/ppo_frame_selector.zip \
    --output_dir results/step5 \
    --num_episodes 5 \
    --gray_mode
"""

import argparse
import os
import json
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO

from scorer.aesthetic_scorer import AestheticScorerPipeline
from actor.gym_wrapper import FrameSelectorGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Combine and evaluate Scorer + Actor")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--actor_model", type=str, 
                       default="Aesthetic-Crop-Selection-with-RLHF/actor/checkpoints/ppo_frame_selector.zip",
                       help="Path to trained PPO model")
    parser.add_argument("--output_dir", type=str, default="results/step5", help="Output directory")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--downscale", type=int, nargs=2, default=[128, 128], help="Downscale resolution")
    parser.add_argument("--init_crop", type=int, nargs=2, default=[448, 448], help="Initial crop size")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--gray_mode", action="store_true", help="Process images in grayscale instead of RGB")
    return parser.parse_args()


def run_actor_episode(env: FrameSelectorGymEnv, model: PPO, max_steps: int = 50) -> Dict[str, Any]:
    """Run a single episode with the trained actor"""
    obs, info = env.reset()
    episode_data = {
        'actions': [],
        'rewards': [],
        'crop_boxes': [],
        'scores': [],
        'final_score': 0.0,
        'steps_taken': 0,
        'terminated': False
    }
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_data['actions'].append(int(action))
        episode_data['rewards'].append(float(reward))
        episode_data['crop_boxes'].append(list(info['crop_box']))
        episode_data['scores'].append(env.core_env.get_current_score())
        episode_data['steps_taken'] = step + 1
        
        if terminated or truncated:
            episode_data['terminated'] = terminated
            episode_data['final_score'] = env.core_env.get_current_score()
            break
    
    return episode_data


def visualize_episodes(image: np.ndarray, episodes: List[Dict[str, Any]], 
                      output_path: str, top_k: int = 3, gray_mode: bool = False):
    """Visualize the best episodes with crop rectangles"""
    # Sort episodes by final score
    sorted_episodes = sorted(episodes, key=lambda x: x['final_score'], reverse=True)
    best_episodes = sorted_episodes[:top_k]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image with all episode paths
    if gray_mode:
        # For grayscale, the image is already in BGR format but converted to grayscale
        # We need to convert it back to RGB for display
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Already grayscale
            display_image = image
        axes[0, 0].imshow(display_image, cmap='gray' if len(display_image.shape) == 2 else None)
    else:
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('All Episode Paths')
    axes[0, 0].axis('off')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, episode in enumerate(episodes):
        color = colors[i % len(colors)]
        crop_boxes = episode['crop_boxes']
        
        # Draw path
        for j, (x, y, w, h) in enumerate(crop_boxes):
            alpha = 0.3 + 0.7 * (j / len(crop_boxes))  # Fade in
            rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                   edgecolor=color, facecolor='none', alpha=alpha)
            axes[0, 0].add_patch(rect)
        
        # Mark final position
        if crop_boxes:
            final_x, final_y, final_w, final_h = crop_boxes[-1]
            rect = patches.Rectangle((final_x, final_y), final_w, final_h, 
                                   linewidth=3, edgecolor=color, facecolor='none')
            axes[0, 0].add_patch(rect)
            axes[0, 0].text(final_x, final_y-5, f'Ep{i+1}: {episode["final_score"]:.2f}', 
                          color=color, fontsize=8, weight='bold')
    
    # Show best episodes individually
    for i, episode in enumerate(best_episodes):
        if i >= 3:
            break
            
        row, col = (0, 1) if i == 0 else (1, 0) if i == 1 else (1, 1)
        
        # Show final crop
        if episode['crop_boxes']:
            final_x, final_y, final_w, final_h = episode['crop_boxes'][-1]
            crop = image[final_y:final_y+final_h, final_x:final_x+final_w]
            if gray_mode:
                if len(crop.shape) == 3 and crop.shape[2] == 3:
                    # Convert BGR to RGB
                    crop_display = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                else:
                    crop_display = crop
                axes[row, col].imshow(crop_display, cmap='gray' if len(crop_display.shape) == 2 else None)
            else:
                axes[row, col].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'Best Episode {i+1}: Score {episode["final_score"]:.2f}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_episode_results(episodes: List[Dict[str, Any]], output_path: str):
    """Save episode results to JSON"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_episodes = []
    for episode in episodes:
        serializable_episode = {
            'actions': episode['actions'],
            'rewards': episode['rewards'],
            'crop_boxes': episode['crop_boxes'],
            'scores': episode['scores'],
            'final_score': episode['final_score'],
            'steps_taken': episode['steps_taken'],
            'terminated': episode['terminated']
        }
        serializable_episodes.append(serializable_episode)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_episodes, f, indent=2)


def compare_with_sliding_window(image: np.ndarray, scorer: AestheticScorerPipeline, 
                               actor_crops: List[Tuple[int, int, int, int]], 
                               output_path: str, gray_mode: bool = False):
    """Compare actor results with sliding window baseline"""
    from scorer.aesthetic_scorer import ImageCropper
    
    # Get sliding window results
    cropper = ImageCropper(crop_size=(224, 224), stride=112)
    sliding_crops = scorer.get_top_crops(image, cropper, top_k=10)
    
    # Score actor crops
    actor_scores = []
    for x, y, w, h in actor_crops:
        crop = image[y:y+h, x:x+w]
        score = scorer.score_image(crop)
        actor_scores.append(score)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sliding window results
    if gray_mode:
        if len(image.shape) == 3 and image.shape[2] == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
        ax1.imshow(display_image, cmap='gray' if len(display_image.shape) == 2 else None)
    else:
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Sliding Window Top 10')
    ax1.axis('off')
    
    for i, (score, (x, y, w, h), _) in enumerate(sliding_crops):
        color = plt.cm.viridis(i / len(sliding_crops))
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x, y-5, f'{score:.2f}', color=color, fontsize=8)
    
    # Actor results
    if gray_mode:
        if len(image.shape) == 3 and image.shape[2] == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
        ax2.imshow(display_image, cmap='gray' if len(display_image.shape) == 2 else None)
    else:
        ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax2.set_title('Actor Selected Crops')
    ax2.axis('off')
    
    for i, ((x, y, w, h), score) in enumerate(zip(actor_crops, actor_scores)):
        color = plt.cm.plasma(i / len(actor_crops))
        rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                               edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x, y-5, f'{score:.2f}', color=color, fontsize=8, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return sliding_crops, actor_scores


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    
    print(f"Loaded image: {image.shape}")
    
    # Initialize scorer
    scorer = AestheticScorerPipeline(gray_mode=args.gray_mode)
    print(f"Initialized aesthetic scorer (gray_mode={args.gray_mode})")
    
    # Load trained actor
    if not os.path.exists(args.actor_model):
        print(f"Warning: Actor model not found at {args.actor_model}")
        print("Please train the actor first using: python actor/train_ppo.py")
        return
    
    model = PPO.load(args.actor_model)
    print(f"Loaded trained actor from {args.actor_model}")
    
    # Create environment
    env = FrameSelectorGymEnv(
        image_path=args.image,
        scorer=scorer,
        downscale_hw=(args.downscale[0], args.downscale[1]),
        init_crop_hw=(args.init_crop[0], args.init_crop[1]),
        max_steps=args.max_steps,
        gray_mode=args.gray_mode
    )
    
    # Run episodes
    print(f"Running {args.num_episodes} episodes...")
    episodes = []
    for i in range(args.num_episodes):
        print(f"Episode {i+1}/{args.num_episodes}")
        episode_data = run_actor_episode(env, model, args.max_steps)
        episodes.append(episode_data)
        print(f"  Final score: {episode_data['final_score']:.2f}, Steps: {episode_data['steps_taken']}")
    
    # Analyze results
    final_scores = [ep['final_score'] for ep in episodes]
    avg_score = np.mean(final_scores)
    max_score = np.max(final_scores)
    min_score = np.min(final_scores)
    
    print(f"\nResults Summary:")
    print(f"  Average final score: {avg_score:.2f}")
    print(f"  Best score: {max_score:.2f}")
    print(f"  Worst score: {min_score:.2f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "episode_results.json")
    save_episode_results(episodes, results_path)
    print(f"Saved episode results to {results_path}")
    
    # Create visualizations
    vis_path = os.path.join(args.output_dir, "episode_visualization.png")
    visualize_episodes(image, episodes, vis_path, gray_mode=args.gray_mode)
    print(f"Saved episode visualization to {vis_path}")
    
    # Compare with sliding window
    actor_crops = [ep['crop_boxes'][-1] for ep in episodes if ep['crop_boxes']]
    if actor_crops:
        comparison_path = os.path.join(args.output_dir, "actor_vs_sliding_window.png")
        sliding_crops, actor_scores = compare_with_sliding_window(
            image, scorer, actor_crops, comparison_path, gray_mode=args.gray_mode)
        print(f"Saved comparison visualization to {comparison_path}")
        
        # Print comparison summary
        sliding_scores = [score for score, _, _ in sliding_crops]
        print(f"\nComparison with Sliding Window:")
        print(f"  Sliding window best: {max(sliding_scores):.2f}")
        print(f"  Actor best: {max(actor_scores):.2f}")
        print(f"  Actor average: {np.mean(actor_scores):.2f}")
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
