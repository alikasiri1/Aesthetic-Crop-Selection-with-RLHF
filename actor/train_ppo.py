"""
Step 4 – Train Actor with PPO using Stable-Baselines3

Usage:
  python actor/train_ppo.py \
    --image Aesthetic-Crop-Selection-with-RLHF/image.jpg \
    --timesteps 20000 \
    --save_dir Aesthetic-Crop-Selection-with-RLHF/actor/checkpoints \
    --downscale 128 128 --init_crop 224 224
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
from ..scorer.aesthetic_scorer import AestheticScorerPipeline


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for 4-channel input (RGB + mask)
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # Input: (H, W, 4) -> (4, H, W)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, observation_space.shape[0], observation_space.shape[1])
            dummy_output = self.cnn(dummy_input)
            n_flatten = dummy_output.shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert from (batch, H, W, 4) to (batch, 4, H, W)
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
    return p.parse_args()


def make_env(image_path: str, downscale, init_crop, seed: int):
    def _thunk():
        scorer = AestheticScorerPipeline()
        env = FrameSelectorGymEnv(
            image_path=image_path,
            scorer=scorer,
            downscale_hw=(downscale[0], downscale[1]),
            init_crop_hw=(init_crop[0], init_crop[1]),
            max_steps=50,
            seed=seed,
        )
        return env
    return _thunk


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = DummyVecEnv([make_env(args.image, args.downscale, args.init_crop, args.seed)])

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
        ent_coef=0.01,
        seed=args.seed,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        ),
    )

    model.learn(total_timesteps=args.timesteps)
    save_path = os.path.join(args.save_dir, "ppo_frame_selector")
    model.save(save_path)
    print(f"Saved PPO model to: {save_path}")


if __name__ == "__main__":
    main()
