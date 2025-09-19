"""
Step 6 – Incremental Improvement Strategy (RLHF-inspired)

This implements a practical 3-stage incremental improvement approach:

Stage 1 (Bootstrap): Use pre-trained aesthetic scorer to train initial policy
Stage 2 (Personalize): Collect human preferences and train reward model  
Stage 3 (Fine-tune): Combine NIMA + human reward for final policy

Usage:
  python step6_incremental_improvement.py --stage 1 --image image.jpg
  python step6_incremental_improvement.py --stage 2 --preference_data preferences.json
  python step6_incremental_improvement.py --stage 3 --image image.jpg --human_reward_model human_reward.pth
"""

import argparse
import json
import os
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scorer.aesthetic_scorer import AestheticScorerPipeline
from actor.gym_wrapper import FrameSelectorGymEnv
from stable_baselines3 import PPO


@dataclass
class PreferencePair:
    """Represents a human preference between two crops"""
    image_path: str
    crop_a: Tuple[int, int, int, int]  # (x, y, w, h)
    crop_b: Tuple[int, int, int, int]
    preference: int  # 0 for A, 1 for B, -1 for no preference
    confidence: float  # 0.0 to 1.0


class HumanRewardModel(nn.Module):
    """
    Small neural network to predict human preferences
    Trained on preference pairs to learn human aesthetic taste
    """
    
    def __init__(self, backbone_name: str = 'efficientnet_b0'):
        super(HumanRewardModel, self).__init__()
        
        # Use same backbone as aesthetic scorer for consistency
        if backbone_name.startswith('efficientnet'):
            import timm
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
            feature_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Preference prediction head
        self.preference_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # *2 because we concatenate features from both crops
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, crop_a, crop_b):
        # Extract features from both crops
        features_a = self.backbone(crop_a)
        features_b = self.backbone(crop_b)
        
        if len(features_a.shape) > 2:
            features_a = features_a.view(features_a.size(0), -1)
        if len(features_b.shape) > 2:
            features_b = features_b.view(features_b.size(0), -1)
            
        # Concatenate features
        combined_features = torch.cat([features_a, features_b], dim=1)
        
        # Predict preference (logit for crop_a > crop_b)
        preference_logit = self.preference_head(combined_features)
        
        return preference_logit


class PreferenceDataset(Dataset):
    """Dataset for human preference pairs"""
    
    def __init__(self, preferences: List[PreferencePair], transform=None):
        self.preferences = preferences
        self.transform = transform
        
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        pref = self.preferences[idx]
        
        # Load image
        image = cv2.imread(pref.image_path)
        if image is None:
            raise ValueError(f"Could not load image: {pref.image_path}")
        
        # Extract crops
        x_a, y_a, w_a, h_a = pref.crop_a
        x_b, y_b, w_b, h_b = pref.crop_b
        
        crop_a = image[y_a:y_a+h_a, x_a:x_a+w_a]
        crop_b = image[y_b:y_b+h_b, x_b:x_b+w_b]
        
        # Resize to standard size
        crop_a = cv2.resize(crop_a, (224, 224))
        crop_b = cv2.resize(crop_b, (224, 224))
        
        # Convert BGR to RGB
        crop_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2RGB)
        crop_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            crop_a = self.transform(crop_a)
            crop_b = self.transform(crop_b)
        
        # Convert to tensor
        crop_a = torch.from_numpy(crop_a).permute(2, 0, 1).float() / 255.0
        crop_b = torch.from_numpy(crop_b).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        crop_a = (crop_a - mean) / std
        crop_b = (crop_b - mean) / std
        
        return crop_a, crop_b, torch.tensor(pref.preference, dtype=torch.float32)


class IncrementalImprovementPipeline:
    """
    Main pipeline for incremental improvement using RLHF-inspired approach
    """
    
    def __init__(self, base_dir: str = "Aesthetic-Crop-Selection-with-RLHF/results/step6"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize components
        self.aesthetic_scorer = AestheticScorerPipeline()
        self.human_reward_model = None
        
    def stage1_bootstrap(self, image_path: str, timesteps: int = 20000):
        """
        Stage 1: Bootstrap with pre-trained aesthetic scorer
        Train initial policy using NIMA-style aesthetic scoring
        """
        print("=== Stage 1: Bootstrap with Pre-trained Aesthetic Scorer ===")
        
        # Create environment with aesthetic scorer
        env = FrameSelectorGymEnv(
            image_path=image_path,
            scorer=self.aesthetic_scorer,
            downscale_hw=(128, 128),
            init_crop_hw=(448, 448),
            max_steps=1000
        )
        
        # Train PPO policy
        from stable_baselines3.common.vec_env import DummyVecEnv
        from actor.train_ppo import make_env, CustomCNN
        
        vec_env = DummyVecEnv([make_env(image_path, (128, 128), (448, 448), 0)])
        
        model = PPO(
            policy="CnnPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            seed=0,
            verbose=1,
            policy_kwargs=dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=256),
            ),
        )
        
        print(f"Training bootstrap policy for {timesteps} timesteps...")
        model.learn(total_timesteps=timesteps)
        
        # Save bootstrap model
        bootstrap_path = os.path.join(self.base_dir, "bootstrap_policy.zip")
        model.save(bootstrap_path)
        print(f"Saved bootstrap policy to: {bootstrap_path}")
        
        return model, bootstrap_path
    
    def stage2_collect_preferences(self, image_path: str, num_pairs: int = 100):
        """
        Stage 2: Collect human preferences
        Generate crop pairs and save for human annotation
        """
        print("=== Stage 2: Collect Human Preferences ===")
        
        # Load bootstrap policy
        bootstrap_path = os.path.join(self.base_dir, "bootstrap_policy.zip")
        if not os.path.exists(bootstrap_path):
            raise FileNotFoundError("Bootstrap policy not found. Run Stage 1 first.")
        
        model = PPO.load(bootstrap_path)
        
        # Create environment
        env = FrameSelectorGymEnv(
            image_path=image_path,
            scorer=self.aesthetic_scorer,
            downscale_hw=(128, 128),
            init_crop_hw=(448, 448),
            max_steps=1000
        )
        
        # Generate diverse crop pairs
        preferences = []
        print(f"Generating {num_pairs} crop pairs for human annotation...")
        
        for i in tqdm(range(num_pairs)):
            # Run episode to get crop
            obs, info = env.reset()
            for _ in range(random.randint(10, 50)):  # Random episode length
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            crop_a = info['crop_box']
            
            # Generate second crop (random or from different episode)
            obs, info = env.reset()
            for _ in range(random.randint(10, 50)):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            crop_b = info['crop_box']
            
            # Create preference pair (preference = -1 means needs human annotation)
            pref = PreferencePair(
                image_path=image_path,
                crop_a=crop_a,
                crop_b=crop_b,
                preference=-1,  # To be filled by human
                confidence=1.0
            )
            preferences.append(pref)
        
        # Save preference pairs for human annotation
        preferences_path = os.path.join(self.base_dir, "preference_pairs.json")
        self._save_preferences(preferences, preferences_path)
        print(f"Saved {len(preferences)} preference pairs to: {preferences_path}")
        print("Please annotate these pairs with human preferences (0=A better, 1=B better, -1=no preference)")
        
        return preferences_path
    
    def stage2_train_reward_model(self, preferences_path: str, epochs: int = 50):
        """
        Stage 2: Train human reward model on annotated preferences
        """
        print("=== Stage 2: Train Human Reward Model ===")
        
        # Load annotated preferences
        preferences = self._load_preferences(preferences_path)
        
        # Filter out unannotated pairs
        annotated_preferences = [p for p in preferences if p.preference != -1]
        print(f"Training on {len(annotated_preferences)} annotated preference pairs")
        
        if len(annotated_preferences) < 10:
            print("Warning: Very few annotated preferences. Consider collecting more data.")
        
        # Create dataset and dataloader
        dataset = PreferenceDataset(annotated_preferences)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Initialize human reward model
        self.human_reward_model = HumanRewardModel()
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.human_reward_model.parameters(), lr=1e-4)
        
        # Training loop
        self.human_reward_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for crop_a, crop_b, preference in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.human_reward_model(crop_a, crop_b)
                loss = criterion(logits.squeeze(), preference)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Save human reward model
        reward_model_path = os.path.join(self.base_dir, "human_reward_model.pth")
        torch.save(self.human_reward_model.state_dict(), reward_model_path)
        print(f"Saved human reward model to: {reward_model_path}")
        
        return reward_model_path
    
    def stage3_fine_tune(self, image_path: str, human_reward_model_path: str, timesteps: int = 10000):
        """
        Stage 3: Fine-tune policy with combined NIMA + human reward
        """
        print("=== Stage 3: Fine-tune with Combined Reward ===")
        
        # Load human reward model
        self.human_reward_model = HumanRewardModel()
        self.human_reward_model.load_state_dict(torch.load(human_reward_model_path))
        self.human_reward_model.eval()
        
        # Create custom environment with combined reward
        class CombinedRewardEnv(FrameSelectorGymEnv):
            def __init__(self, image_path, scorer, downscale_hw, init_crop_hw, max_steps, 
                         human_reward_model=None, aesthetic_scorer=None, **kwargs):
                super().__init__(image_path, scorer, downscale_hw, init_crop_hw, max_steps, **kwargs)
                self.human_reward_model = human_reward_model
                self.aesthetic_scorer = aesthetic_scorer
                
            def step(self, action):
                obs, reward, terminated, truncated, info = self.core_env.step(int(action))
                
                # Get aesthetic score
                aesthetic_score = self.core_env.get_current_score()
                
                # Get human preference score (if available)
                if self.human_reward_model is not None:
                    x, y, w, h = info['crop_box']
                    crop = self.core_env.image_bgr[y:y+h, x:x+w]
                    crop = cv2.resize(crop, (224, 224))
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                    
                    # Normalize
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    crop_tensor = (crop_tensor - mean) / std
                    crop_tensor = crop_tensor.unsqueeze(0)
                    
                    # Get human preference score (compare with average crop)
                    with torch.no_grad():
                        # Create a dummy "average" crop for comparison
                        avg_crop = torch.zeros_like(crop_tensor)
                        human_score = torch.sigmoid(self.human_reward_model(crop_tensor, avg_crop)).item()
                else:
                    human_score = 0.5
                
                # Combine rewards (weighted average)
                combined_reward = 0.7 * (aesthetic_score - 1) / 9 + 0.3 * human_score
                combined_reward -= 0.005  # Small step penalty
                
                return obs.astype(np.float32), float(combined_reward), bool(terminated), bool(truncated), info
        
        # Create environment with combined reward
        env = CombinedRewardEnv(
            image_path=image_path,
            scorer=self.aesthetic_scorer,
            downscale_hw=(128, 128),
            init_crop_hw=(448, 448),
            max_steps=1000,
            human_reward_model=self.human_reward_model,
            aesthetic_scorer=self.aesthetic_scorer
        )
        
        # Load bootstrap policy as starting point
        bootstrap_path = os.path.join(self.base_dir, "bootstrap_policy.zip")
        model = PPO.load(bootstrap_path)
        
        # Fine-tune with combined reward
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([lambda: env])
        
        model.set_env(vec_env)
        print(f"Fine-tuning policy for {timesteps} timesteps with combined reward...")
        model.learn(total_timesteps=timesteps)
        
        # Save fine-tuned model
        final_path = os.path.join(self.base_dir, "final_policy.zip")
        model.save(final_path)
        print(f"Saved final fine-tuned policy to: {final_path}")
        
        return model, final_path
    
    def _save_preferences(self, preferences: List[PreferencePair], path: str):
        """Save preference pairs to JSON"""
        data = []
        for pref in preferences:
            data.append({
                'image_path': pref.image_path,
                'crop_a': pref.crop_a,
                'crop_b': pref.crop_b,
                'preference': pref.preference,
                'confidence': pref.confidence
            })
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_preferences(self, path: str) -> List[PreferencePair]:
        """Load preference pairs from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        preferences = []
        for item in data:
            pref = PreferencePair(
                image_path=item['image_path'],
                crop_a=tuple(item['crop_a']),
                crop_b=tuple(item['crop_b']),
                preference=item['preference'],
                confidence=item['confidence']
            )
            preferences.append(pref)
        
        return preferences


def main():
    parser = argparse.ArgumentParser(description="Incremental Improvement Pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True, 
                       help="Stage to run: 1=bootstrap, 2=collect/train, 3=fine-tune")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--timesteps", type=int, default=20000, help="Training timesteps")
    parser.add_argument("--preference_data", type=str, help="Path to preference data JSON")
    parser.add_argument("--human_reward_model", type=str, help="Path to human reward model")
    parser.add_argument("--num_pairs", type=int, default=100, help="Number of preference pairs to generate")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for reward model training")
    
    args = parser.parse_args()
    
    pipeline = IncrementalImprovementPipeline()
    
    if args.stage == 1:
        if not args.image:
            raise ValueError("--image required for stage 1")
        pipeline.stage1_bootstrap(args.image, args.timesteps)
        
    elif args.stage == 2:
        if not args.preference_data:
            # Generate preference pairs
            if not args.image:
                raise ValueError("--image required for generating preference pairs")
            pipeline.stage2_collect_preferences(args.image, args.num_pairs)
        else:
            # Train reward model
            pipeline.stage2_train_reward_model(args.preference_data, args.epochs)
            
    elif args.stage == 3:
        if not args.image or not args.human_reward_model:
            raise ValueError("--image and --human_reward_model required for stage 3")
        pipeline.stage3_fine_tune(args.image, args.human_reward_model, args.timesteps)


if __name__ == "__main__":
    main()
