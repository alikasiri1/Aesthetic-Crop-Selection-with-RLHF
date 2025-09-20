"""
Step 4 – Train Actor with PPO using Stable-Baselines3

Usage:
  # RGB mode (default):
  python actor/train_ppo.py \
    --image Aesthetic-Crop-Selection-with-RLHF/image.jpg \
    --timesteps 20000 \
    --save_dir Aesthetic-Crop-Selection-with-RLHF/actor/checkpoints \
    --downscale 128 128 --init_crop 224 224

  # Grayscale mode:
  python actor/train_ppo.py \
    --image Aesthetic-Crop-Selection-with-RLHF/image.jpg \
    --timesteps 20000 \
    --save_dir Aesthetic-Crop-Selection-with-RLHF/actor/checkpoints \
    --downscale 128 128 --init_crop 224 224 \
    --gray_mode
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .gym_wrapper import FrameSelectorGymEnv
from scorer.aesthetic_scorer import AestheticScorerPipeline


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for both RGB (3-channel) and grayscale (1-channel) input
    """
    def __init__(self, observation_space, features_dim: int = 256, gray_mode: bool = False):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        self.gray_mode = gray_mode
        input_channels = 1 if gray_mode else 3
        
        # Input: (H, W, C) -> (C, H, W)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, observation_space.shape[0], observation_space.shape[1])
            dummy_output = self.cnn(dummy_input)
            n_flatten = dummy_output.shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert from (batch, H, W, C) to (batch, C, H, W)
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--timesteps", type=int, default=20000)
    p.add_argument("--save_dir", type=str, default="Aesthetic-Crop-Selection-with-RLHF/actor/checkpoints")
    p.add_argument("--downscale", type=int, nargs=2, default=[128, 128])
    p.add_argument("--init_crop", type=int, nargs=2, default=[224, 224])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--gray_mode", action="store_true", help="Process images in grayscale instead of RGB")
    return p.parse_args()


def make_env(image_path: str, downscale, init_crop, seed: int, max_steps: int, gray_mode: bool = False):
    def _thunk():
        scorer = AestheticScorerPipeline()
        env = FrameSelectorGymEnv(
            image_path=image_path,
            scorer=scorer,
            downscale_hw=(downscale[0], downscale[1]),
            init_crop_hw=(init_crop[0], init_crop[1]),
            max_steps=max_steps,
            seed=seed,
            gray_mode=gray_mode,
        )
        return env
    return _thunk


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = DummyVecEnv([make_env(args.image, args.downscale, args.init_crop, args.seed, args.max_steps, args.gray_mode)])

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.05, # .01 is better for smaller models
        seed=args.seed,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256, gray_mode=args.gray_mode),
        ),
    )

    model.learn(total_timesteps=args.timesteps)
    save_path = os.path.join(args.save_dir, "ppo_frame_selector")
    model.save(save_path)
    print(f"Saved PPO model to: {save_path}")


if __name__ == "__main__":
    main()
