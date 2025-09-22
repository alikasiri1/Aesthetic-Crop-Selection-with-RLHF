"""
Step 6 – Incremental Improvement Strategy (RLHF-inspired)

This implements a practical 3-stage incremental improvement approach:

Stage 1 (Bootstrap): Use pre-trained aesthetic scorer to train initial policy
Stage 2 (Personalize): Collect human preferences and train reward model  
Stage 3 (Fine-tune): Combine NIMA + human reward for final policy

Usage:
  # RGB mode (default):
  python step6_incremental_improvement.py --stage 1 --image image.jpg
  python step6_incremental_improvement.py --stage 2 --preference_data preferences.json
  python step6_incremental_improvement.py --stage 3 --image image.jpg --human_reward_model human_reward.pth
  
  # Grayscale mode:
  python step6_incremental_improvement.py --stage 1 --image image.jpg --gray_mode
  python step6_incremental_improvement.py --stage 2 --preference_data preferences.json --gray_mode
  python step6_incremental_improvement.py --stage 3 --image image.jpg --human_reward_model human_reward.pth --gray_mode
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
    
    def __init__(self, backbone_name: str = 'efficientnet_b0', gray_mode: bool = False):
        super(HumanRewardModel, self).__init__()
        
        self.gray_mode = gray_mode
        
        # Use same backbone as aesthetic scorer for consistency
        if backbone_name.startswith('efficientnet'):
            import timm
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
            
            # Modify first conv layer for grayscale if needed
            if gray_mode:
                original_conv = self.backbone.conv_stem
                self.backbone.conv_stem = nn.Conv2d(1, original_conv.out_channels, 
                                                   kernel_size=original_conv.kernel_size,
                                                   stride=original_conv.stride,
                                                   padding=original_conv.padding,
                                                   bias=original_conv.bias is not None)
                # Initialize with average of RGB weights
                with torch.no_grad():
                    self.backbone.conv_stem.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            
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
    
    def __init__(self, preferences: List[PreferencePair], transform=None, gray_mode: bool = False):
        self.preferences = preferences
        self.transform = transform
        self.gray_mode = gray_mode
        
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        pref = self.preferences[idx]
        
        # Load image
        image = cv2.imread(pref.image_path)
        if image is None:
            raise ValueError(f"Could not load image: {pref.image_path}")
        
        # Extract crops (now using x_min, y_min, x_max, y_max format)
        x_min_a, y_min_a, x_max_a, y_max_a = pref.crop_a
        x_min_b, y_min_b, x_max_b, y_max_b = pref.crop_b
        
        crop_a = image[y_min_a:y_max_a, x_min_a:x_max_a]
        crop_b = image[y_min_b:y_max_b, x_min_b:x_max_b]
        
        # Resize to standard size
        crop_a = cv2.resize(crop_a, (224, 224))
        crop_b = cv2.resize(crop_b, (224, 224))
        
        if self.gray_mode:
            # Convert BGR to grayscale
            crop_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY)
            crop_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2GRAY)
            
            if self.transform:
                crop_a = self.transform(crop_a)
                crop_b = self.transform(crop_b)
            
            # Convert to tensor
            crop_a = torch.from_numpy(crop_a).unsqueeze(0).float() / 255.0  # Add channel dimension
            crop_b = torch.from_numpy(crop_b).unsqueeze(0).float() / 255.0  # Add channel dimension
            
            # Normalize grayscale
            mean = torch.tensor([0.5]).view(1, 1, 1)
            std = torch.tensor([0.5]).view(1, 1, 1)
            crop_a = (crop_a - mean) / std
            crop_b = (crop_b - mean) / std
        else:
            # Convert BGR to RGB
            crop_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2RGB)
            crop_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                crop_a = self.transform(crop_a)
                crop_b = self.transform(crop_b)
            
            # Convert to tensor
            crop_a = torch.from_numpy(crop_a).permute(2, 0, 1).float() / 255.0
            crop_b = torch.from_numpy(crop_b).permute(2, 0, 1).float() / 255.0
            
            # Normalize RGB
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            crop_a = (crop_a - mean) / std
            crop_b = (crop_b - mean) / std
        
        return crop_a, crop_b, torch.tensor(pref.preference, dtype=torch.float32)


class IncrementalImprovementPipeline:
    """
    Main pipeline for incremental improvement using RLHF-inspired approach
    """
    
    def __init__(self, base_dir: str = "Aesthetic-Crop-Selection-with-RLHF/results/step6", gray_mode: bool = False):
        self.base_dir = base_dir
        self.gray_mode = gray_mode
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize components
        self.aesthetic_scorer = AestheticScorerPipeline(gray_mode=gray_mode)
        self.human_reward_model = None
        
    def stage1_bootstrap(self, image_path: str, timesteps: int = 20000, downscale: Tuple[int, int] = (128, 128), init_crop: Tuple[int, int] = (448, 448), max_steps: int = 1000):
        """
        Stage 1: Bootstrap with pre-trained aesthetic scorer
        Train initial policy using NIMA-style aesthetic scoring
        """
        print("=== Stage 1: Bootstrap with Pre-trained Aesthetic Scorer ===")
        
        # Create environment with aesthetic scorer
        env = FrameSelectorGymEnv(
            image_path=image_path,
            scorer=self.aesthetic_scorer,
            downscale_hw=downscale,
            init_crop_hw=init_crop,
            max_steps=max_steps,
            gray_mode=self.gray_mode
        )
        
        # Train PPO policy
        from stable_baselines3.common.vec_env import DummyVecEnv
        from actor.train_ppo import make_env, CustomCNN
        
        vec_env = DummyVecEnv([make_env(image_path, downscale, init_crop, 0, max_steps, self.gray_mode)])
        # vec_env = DummyVecEnv([env])
        
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
                features_extractor_kwargs=dict(features_dim=256, gray_mode=self.gray_mode),
            ),
        )
        
        print(f"Training bootstrap policy for {timesteps} timesteps...")
        model.learn(total_timesteps=timesteps)
        
        # Save bootstrap model
        bootstrap_path = os.path.join(self.base_dir, "bootstrap_policy.zip")
        model.save(bootstrap_path)
        print(f"Saved bootstrap policy to: {bootstrap_path}")
        
        return model, bootstrap_path
    
    def stage2_collect_preferences(self, image_path: str, num_pairs: int = 100, downscale: Tuple[int, int] = (128, 128), init_crop: Tuple[int, int] = (448, 448), max_steps: int = 1000):
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
            downscale_hw=downscale,
            init_crop_hw=init_crop,
            max_steps=max_steps,
            gray_mode=self.gray_mode
        )
        
        # Generate diverse crop pairs
        preferences = []
        print(f"Generating {num_pairs} crop pairs for human annotation...")
        
        for i in tqdm(range(num_pairs)):
            # Run episode to get crop
            obs, info = env.reset(random_init=True)
            for _ in range(random.randint(10, 50)):  # Random episode length
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            crop_a = info['crop_box']
            
            # Generate second crop (random or from different episode)
            obs, info = env.reset(random_init=True)
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
    
    def stage2_train_reward_model(self, preferences_path: str, epochs: int = 50, downscale: Tuple[int, int] = (128, 128), init_crop: Tuple[int, int] = (448, 448), max_steps: int = 1000):
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
        dataset = PreferenceDataset(annotated_preferences, gray_mode=self.gray_mode)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Initialize human reward model
        self.human_reward_model = HumanRewardModel(backbone_name='efficientnet_b0', gray_mode=self.gray_mode)
        
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
                # Ensure both tensors have the same shape for BCE loss
                if logits.shape != preference.shape:
                    if logits.dim() > preference.dim():
                        logits = logits.squeeze()
                    elif preference.dim() > logits.dim():
                        preference = preference.squeeze()
                    # If still different shapes, reshape to match
                    if logits.shape != preference.shape:
                        logits = logits.view(-1)
                        preference = preference.view(-1)
                loss = criterion(logits, preference)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            # if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Save human reward model
        reward_model_path = os.path.join(self.base_dir, "human_reward_model.pth")
        torch.save(self.human_reward_model.state_dict(), reward_model_path)
        print(f"Saved human reward model to: {reward_model_path}")
        
        return reward_model_path
    
    def stage3_fine_tune(self, image_path: str, human_reward_model_path: str, timesteps: int = 10000, downscale: Tuple[int, int] = (128, 128), init_crop: Tuple[int, int] = (448, 448), max_steps: int = 1000):
        """
        Stage 3: Fine-tune policy with combined NIMA + human reward
        """
        print("=== Stage 3: Fine-tune with Combined Reward ===")
        
        # Load human reward model
        self.human_reward_model = HumanRewardModel(backbone_name='efficientnet_b0', gray_mode=self.gray_mode)
        self.human_reward_model.load_state_dict(torch.load(human_reward_model_path))
        self.human_reward_model.eval()
        
        # Create custom environment with combined reward
        class CombinedRewardEnv(FrameSelectorGymEnv):
            def __init__(self, image_path, scorer, downscale_hw, init_crop_hw, max_steps, 
                         human_reward_model=None, aesthetic_scorer=None, gray_mode=False, **kwargs):
                super().__init__(image_path, scorer, downscale_hw, init_crop_hw, max_steps, gray_mode=gray_mode, **kwargs)
                self.human_reward_model = human_reward_model
                self.aesthetic_scorer = aesthetic_scorer
                self.init_crop_hw = init_crop_hw
                self.gray_mode = gray_mode
                self._baseline_crops = None
                self._init_baseline_crops()
            
            def _init_baseline_crops(self):
                """Initialize baseline crops for human scoring comparison"""
                if self.human_reward_model is None:
                    return
                
                h, w = self.core_env.image_bgr.shape[:2]
                crop_size = 224
                
                # Create multiple baseline crops for more robust comparison
                self._baseline_crops = []
                
                # Center crop
                center_x, center_y = w // 2, h // 2
                x1 = max(0, center_x - crop_size // 2)
                y1 = max(0, center_y - crop_size // 2)
                x2 = min(w, x1 + crop_size)
                y2 = min(h, y1 + crop_size)
                center_crop = self.core_env.image_bgr[y1:y2, x1:x2]
                self._baseline_crops.append(center_crop)
                
                # Corner crops
                corners = [(0, 0), (w-crop_size, 0), (0, h-crop_size), (w-crop_size, h-crop_size)]
                for cx, cy in corners:
                    if cx >= 0 and cy >= 0 and cx + crop_size <= w and cy + crop_size <= h:
                        corner_crop = self.core_env.image_bgr[cy:cy+crop_size, cx:cx+crop_size]
                        self._baseline_crops.append(corner_crop)
                
                # Random crops
                for _ in range(3):
                    rx = np.random.randint(0, max(1, w - crop_size))
                    ry = np.random.randint(0, max(1, h - crop_size))
                    random_crop = self.core_env.image_bgr[ry:ry+crop_size, rx:rx+crop_size]
                    self._baseline_crops.append(random_crop)
                
                print(f"Initialized {len(self._baseline_crops)} baseline crops for human scoring")
            
            def _get_human_score(self, crop_tensor):
                """Get human score by comparing with multiple baseline crops"""
                if self.human_reward_model is None or self._baseline_crops is None:
                    return 0.5
                
                scores = []
                for baseline_crop in self._baseline_crops:
                    # Preprocess baseline crop
                    baseline_crop = cv2.resize(baseline_crop, (224, 224))
                    
                    if self.gray_mode:
                        baseline_crop = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY)
                        baseline_tensor = torch.from_numpy(baseline_crop).unsqueeze(0).float() / 255.0
                        baseline_tensor = (baseline_tensor - 0.5) / 0.5
                    else:
                        baseline_crop = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2RGB)
                        baseline_tensor = torch.from_numpy(baseline_crop).permute(2, 0, 1).float() / 255.0
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        baseline_tensor = (baseline_tensor - mean) / std
                    
                    baseline_tensor = baseline_tensor.unsqueeze(0)
                    
                    # Get preference score
                    with torch.no_grad():
                        preference_logit = self.human_reward_model(crop_tensor, baseline_tensor)
                        score = torch.sigmoid(preference_logit).item()
                        scores.append(score)
                
                # Return average score across all baselines
                return np.mean(scores)
            def step(self, action):
                obs, reward, terminated, truncated, info = self.core_env.step(int(action))

                # فقط human score
                if self.human_reward_model is not None:
                    x_min, y_min, x_max, y_max = info['crop_box']
                    crop = self.core_env.image_bgr[y_min:y_max, x_min:x_max]
                    crop = cv2.resize(crop, (224, 224))

                    if self.gray_mode:
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        crop_tensor = torch.from_numpy(crop).unsqueeze(0).float() / 255.0
                        mean = torch.tensor([0.5]).view(1, 1, 1)
                        std = torch.tensor([0.5]).view(1, 1, 1)
                        crop_tensor = (crop_tensor - mean) / std
                    else:
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        crop_tensor = (crop_tensor - mean) / std

                    crop_tensor = crop_tensor.unsqueeze(0)
                    human_score = self._get_human_score(crop_tensor)
                else:
                    human_score = 0.5

                # نسبت طول و عرض کراپ فعلی
                crop_width = x_max - x_min
                crop_height = y_max - y_min
                aspect_ratio = crop_width / crop_height if crop_height > 0 else 1.0

                # 1) پنالتی عکس‌های خیلی عریض (هرچه aspect_ratio بزرگتر از 1 منفی‌تر)
                penalty_wide = -0.3 * min(0, aspect_ratio - 1.0)

                # 2) ریوارد نزدیکی به نسبت طلایی (۱.۶۱۸)
                golden_ratio = 1.618
                # اختلاف از نسبت طلایی
                aspect_ratio = max(crop_width, crop_height) / min(crop_width, crop_height)
                diff_from_golden = abs(aspect_ratio - golden_ratio)
                # هرچه کمتر، ریوارد بیشتر
                reward_golden = 0.3 * (1.0 - min(diff_from_golden / golden_ratio, 1.0))

                # ریوارد نهایی = human_score + penalty_wide + reward_golden
                print(f"HumanScore={human_score:.3f}, PenaltyWide={penalty_wide:.3f}, RewardGolden={reward_golden:.3f}")
                final_reward = human_score + penalty_wide + reward_golden
                
                # برای جلوگیری از مقادیر عجیب
                final_reward = float(np.clip(final_reward, 0.0, 1.5))

                if hasattr(self, '_step_count'):
                    self._step_count += 1
                else:
                    self._step_count = 1

                
                print(f"Step {self._step_count}: Human={human_score:.3f}, Aspect={aspect_ratio:.3f}, "
                    f"PenaltyWide={penalty_wide:.3f}, RewardGolden={reward_golden:.3f}, "
                    f"FinalReward={final_reward:.3f}")

                return obs.astype(np.float32), final_reward, bool(terminated), bool(truncated), info

        
        # Create environment with combined reward
        env = CombinedRewardEnv(
            image_path=image_path,
            scorer=self.aesthetic_scorer,
            downscale_hw=downscale,
            init_crop_hw=init_crop,
            max_steps=max_steps,
            human_reward_model=self.human_reward_model,
            aesthetic_scorer=self.aesthetic_scorer,
            gray_mode=self.gray_mode
        )
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([lambda: env])
        # Load bootstrap policy as starting point
        # bootstrap_path = os.path.join(self.base_dir, "bootstrap_policy.zip")
        # model = PPO.load(bootstrap_path)
        from actor.train_ppo import make_env, CustomCNN
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
                features_extractor_kwargs=dict(features_dim=256, gray_mode=self.gray_mode),
            ),
        )
        
        
        
        # model.set_env(vec_env)
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
    parser.add_argument("--downscale", type=int, nargs=2, default=[128, 128], help="Downscale dimensions")
    parser.add_argument("--init_crop", type=int, nargs=2, default=[448, 448], help="Initial crop dimensions")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--gray_mode", action="store_true", help="Process images in grayscale instead of RGB")
    args = parser.parse_args()
    
    pipeline = IncrementalImprovementPipeline(gray_mode=args.gray_mode)
    
    if args.stage == 1:
        if not args.image:
            raise ValueError("--image required for stage 1")
        pipeline.stage1_bootstrap(args.image, args.timesteps, args.downscale, args.init_crop, args.max_steps)
        
    elif args.stage == 2:
        if not args.preference_data:
            # Generate preference pairs
            if not args.image:
                raise ValueError("--image required for generating preference pairs")
            pipeline.stage2_collect_preferences(args.image, args.num_pairs, args.downscale, args.init_crop, args.max_steps)
        else:
            # Train reward model
            pipeline.stage2_train_reward_model(args.preference_data, args.epochs, args.downscale, args.init_crop, args.max_steps)
            
    elif args.stage == 3:
        if not args.image or not args.human_reward_model:
            raise ValueError("--image and --human_reward_model required for stage 3")
        pipeline.stage3_fine_tune(args.image, args.human_reward_model, args.timesteps, args.downscale, args.init_crop, args.max_steps)


if __name__ == "__main__":
    main()
