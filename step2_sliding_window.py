"""
Step 2 – Sliding Window + Scorer Test

Usage:
  python step2_sliding_window.py \
    --image image.jpg \
    --output_dir results/step2 \
    --crop_size 224 224 \
    --stride 112 \
    --top_k 10 \
    --multi_scale 0.75 1.0 1.25
"""

import argparse
import os
import csv
from typing import List, Tuple

import cv2
import numpy as np

from scorer.aesthetic_scorer import AestheticScorerPipeline, ImageCropper
from scorer.policy_scorer import PolicyScorerPipeline
from utils.image_utils import visualize_crops, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window scoring and top-K crop extraction")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="results/step2", help="Directory to save outputs")
    parser.add_argument("--crop_size", type=int, nargs=2, default=[224, 224], help="Crop size h w")
    parser.add_argument("--stride", type=int, default=112, help="Sliding window stride (in pixels)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top crops to save")
    parser.add_argument("--multi_scale", type=float, nargs="*", default=None, help="Optional multi-scale factors, e.g. 0.75 1.0 1.25")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Backbone model name")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_topk_metadata(top_crops: List[Tuple[float, Tuple[int, int, int, int], np.ndarray]], csv_path: str) -> None:
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "x", "y", "w", "h", "filename"])
        for i, (score, (x, y, w, h), _) in enumerate(top_crops, start=1):
            writer.writerow([i, f"{score:.4f}", x, y, w, h, f"crop_{i}_score_{score:.2f}.jpg"])

from stable_baselines3 import PPO

class RLScorer:
    def __init__(self, policy_path):
        self.policy = PPO.load(policy_path)
    def score_image(self, crop):
        obs = self._preprocess_crop(crop)
        action, _ = self.policy.predict(obs, deterministic=True)
        # or use self.policy.value_net(obs) as reward proxy
        return float(action)  # or float(value)



def main() -> None:
    args = parse_args()

    ensure_dir(args.output_dir)
    crops_dir = os.path.join(args.output_dir, "top_crops")
    ensure_dir(crops_dir)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # scorer = AestheticScorerPipeline(backbone_name=args.backbone)
    scorer = PolicyScorerPipeline("Aesthetic-Crop-Selection-with-RLHF/results/step6/final_policy.zip", gray_mode=True) 
    cropper = ImageCropper(crop_size=(args.crop_size[0], args.crop_size[1]), stride=args.stride)

    if args.multi_scale and len(args.multi_scale) > 0:
        # Multi-scale mode
        multi_scale_crops = cropper.create_multi_scale_crops(image, scales=args.multi_scale)
        scored_crops = []
        for crop, coords in multi_scale_crops:
            score = scorer.score_image(crop)
            scored_crops.append((score, coords, crop))
        scored_crops.sort(key=lambda x: x[0], reverse=True)
        top_crops = scored_crops[: args.top_k]
    else:
        # Single-scale sliding window
        top_crops = scorer.get_top_crops(image, cropper, top_k=args.top_k)

    # Save crops and metadata
    for i, (score, coords, crop) in enumerate(top_crops, start=1):
        out_path = os.path.join(crops_dir, f"crop_{i}_score_{score:.2f}.jpg")
        save_image(crop, out_path)

    csv_path = os.path.join(args.output_dir, "top_crops.csv")
    save_topk_metadata(top_crops, csv_path)

    # Save a visualization overlay
    vis_path = os.path.join(args.output_dir, "visualization.png")
    visualize_crops(image, top_crops, top_k=len(top_crops), save_path=vis_path)

    print(f"Saved top {len(top_crops)} crops to: {crops_dir}")
    print(f"Saved metadata CSV to: {csv_path}")
    print(f"Saved visualization to: {vis_path}")


if __name__ == "__main__":
    main()
