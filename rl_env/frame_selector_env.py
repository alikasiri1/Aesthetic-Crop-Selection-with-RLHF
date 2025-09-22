"""
Step 3 – RL Environment (Frame Selector Agent)

Environment definition (Gym-like):
- state: concatenation of a downscaled RGB image and a crop mask/box encoding
- action: discrete set {move up/down/left/right, expand/contract w/h, jump proposals, select}
- reward: aesthetic score from the Scorer for the current crop (normalized), with optional shaping
- done: after max_steps or when the agent selects

Usage example:
    env = FrameSelectorEnv(image_path, scorer)
    obs, info = env.reset()
    obs, reward, done, trunc, info = env.step(action)
"""

from typing import Tuple, Dict, Any, Optional, List
import os
import math

import cv2
import numpy as np

from scorer.aesthetic_scorer import AestheticScorerPipeline, ImageCropper


class FrameSelectorEnv:
    def __init__(
        self,
        image_path: str,
        scorer: Optional[AestheticScorerPipeline] = None,
        downscale_hw: Tuple[int, int] = (128, 128),
        init_crop_hw: Tuple[int, int] = (224, 224),
        step_pixels: int = 32, # 32 is better for smaller models
        size_step_pixels: int = 32,
        max_steps: int = 50,
        proposals: Optional[List[Tuple[int, int, int, int]]] = None,
        backbone: str = "efficientnet_b0",
        seed: int = 0,
        allow_resize: bool = False,
        gray_mode: bool = False,
    ) -> None:
        assert os.path.exists(image_path), f"Image not found: {image_path}"
        self.rng = np.random.RandomState(seed)
        self.gray_mode = gray_mode

        self.image_bgr = cv2.imread(image_path)
        if self.image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        
        # Convert to grayscale if gray_mode is enabled
        if self.gray_mode:
            self.image_bgr = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
            # Convert grayscale to 3-channel for compatibility with existing code
            self.image_bgr = cv2.cvtColor(self.image_bgr, cv2.COLOR_GRAY2BGR)
            print("Processing image in grayscale mode")
        
        self.H, self.W = self.image_bgr.shape[:2]
        print(f"Image size: {self.H}x{self.W}")
        
        self.scorer = scorer or AestheticScorerPipeline(backbone_name=backbone, gray_mode=gray_mode)
        self.downscale_hw = downscale_hw
        self.init_crop_hw = init_crop_hw
        self.step_pixels = step_pixels
        self.size_step_pixels = size_step_pixels
        self.max_steps = max_steps
        self.step_count = 0

        # Action space definition - 8 edge controls + 1 finish action
        # 0: Decrease bottom width   -> move bottom edge up (reduce height)
        # 1: Increase bottom width   -> move bottom edge down (increase height)
        # 2: Decrease top width      -> move top edge down (reduce height)
        # 3: Increase top width      -> move top edge up (increase height)
        # 4: Decrease left height    -> move left edge right (reduce width)
        # 5: Increase left height    -> move left edge left (increase width)
        # 6: Decrease right height   -> move right edge left (reduce width)
        # 7: Increase right height   -> move right edge right (increase width)
        # 8: Finish/Save             -> end episode (agent decides to stop)
        self.num_actions = 9
        print(f"Number of actions: {self.num_actions}")
        print(f"Max steps: {self.max_steps}")

        # Initialize crop box (x_min, y_min, x_max, y_max) - starts as full image
        self.crop_box = self._init_full_image_crop()

        # Proposals (optional): list of (x_min, y_min, x_max, y_max)
        self.proposals = proposals if proposals is not None else self._generate_proposals()

        # Cached downscaled image for observation
        self.obs_image = cv2.resize(self.image_bgr, (self.downscale_hw[1], self.downscale_hw[0]))

    # ------------------------ Core API ------------------------
    def reset(self, random_init: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        if random_init:
            self.crop_box = self._init_random_crop()
        else:
            self.crop_box = self._init_full_image_crop()
        
        obs = self._build_observation()
        return obs, {"crop_box": self.crop_box}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        # Handle finish/save action
        if action == 8:
            # Do not change crop box; compute final reward and terminate
            reward = self._compute_reward()
            done = True
            truncated = False
            obs = self._build_observation()
            info = {"crop_box": self.crop_box, "step": self.step_count, "finished": True}
            return obs, reward, done, truncated, info

        self._apply_action(action)
        self._clamp_crop_box()

        reward = self._compute_reward()
        done = False
        truncated = self.step_count >= self.max_steps

        obs = self._build_observation()
        info = {"crop_box": self.crop_box, "step": self.step_count}
        return obs, reward, done, truncated, info

    # ------------------------ Helpers ------------------------
    def _init_full_image_crop(self) -> Tuple[int, int, int, int]:
        """Initialize crop as full image (x_min, y_min, x_max, y_max)"""
        return (0, 0, self.W, self.H)

    def _init_random_crop(self) -> Tuple[int, int, int, int]:
        """Random initial crop (x_min, y_min, x_max, y_max)"""
        w, h = self.init_crop_hw[1], self.init_crop_hw[0]
        w = min(w, self.W)
        h = min(h, self.H)
        x_min = self.rng.randint(0, max(1, self.W - w))
        y_min = self.rng.randint(0, max(1, self.H - h))
        x_max = x_min + w
        y_max = y_min + h
        return (x_min, y_min, x_max, y_max)

    def _generate_proposals(self, k: int = 20) -> List[Tuple[int, int, int, int]]:
        # Use a simple grid of proposals
        cropper = ImageCropper(crop_size=self.init_crop_hw, stride=max(self.init_crop_hw[0] // 2, 32))
        crops = cropper.create_crops(self.image_bgr)
        # Score quickly and pick top-k as proposals
        scored = []
        for crop, box in crops:
            s = self.scorer.score_image(crop)
            # Convert (x, y, w, h) to (x_min, y_min, x_max, y_max)
            x, y, w, h = box
            converted_box = (x, y, x + w, y + h)
            scored.append((s, converted_box))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [box for _, box in scored[:k]]

    def _apply_action(self, action: int) -> None:
        x_min, y_min, x_max, y_max = self.crop_box

        # Edge manipulation actions remapped per spec
        if action == 0:  # Decrease bottom width -> move bottom edge up
            y_max -= self.step_pixels
        elif action == 1:  # Increase bottom width -> move bottom edge down
            y_max += self.step_pixels
        elif action == 2:  # Decrease top width -> move top edge down
            y_min += self.step_pixels
        elif action == 3:  # Increase top width -> move top edge up
            y_min -= self.step_pixels
        elif action == 4:  # Decrease left height -> move left edge right
            x_min += self.step_pixels
        elif action == 5:  # Increase left height -> move left edge left
            x_min -= self.step_pixels
        elif action == 6:  # Decrease right height -> move right edge left
            x_max -= self.step_pixels
        elif action == 7:  # Increase right height -> move right edge right
            x_max += self.step_pixels
        else:
            raise ValueError(f"Invalid action: {action}")

        self.crop_box = (x_min, y_min, x_max, y_max)

    def _clamp_crop_box(self) -> None:
        x_min, y_min, x_max, y_max = self.crop_box
        
        # Ensure crop stays within image boundaries
        x_min = max(0, min(x_min, self.W))
        y_min = max(0, min(y_min, self.H))
        x_max = max(0, min(x_max, self.W))
        y_max = max(0, min(y_max, self.H))
        
        # Ensure crop doesn't invert (x_min < x_max, y_min < y_max)
        # and has minimum size
        min_size = 32
        if x_max - x_min < min_size:
            if x_min == 0:
                x_max = min_size
            else:
                x_min = x_max - min_size
        if y_max - y_min < min_size:
            if y_min == 0:
                y_max = min_size
            else:
                y_min = y_max - min_size
                
        self.crop_box = (x_min, y_min, x_max, y_max)

    def _compute_reward(self) -> float:
        x_min, y_min, x_max, y_max = self.crop_box
        crop = self.image_bgr[y_min:y_max, x_min:x_max]
        score_1_to_10 = self.scorer.score_image(crop)
        # Normalize to [0, 1] and optionally add small step penalty to encourage quicker selection
        reward = (score_1_to_10 - 1.0) / 9.0
        reward -= 0.005  # small per-step penalty
        return float(reward)

    def _build_observation(self) -> np.ndarray:
        # Get current crop and resize to observation size
        x_min, y_min, x_max, y_max = self.crop_box
        crop = self.image_bgr[y_min:y_max, x_min:x_max]
        
        # Resize crop to observation size
        obs_crop = cv2.resize(crop, (self.downscale_hw[1], self.downscale_hw[0]))
        
        if self.gray_mode:
            # Convert to grayscale and add channel dimension
            obs = cv2.cvtColor(obs_crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            obs = np.expand_dims(obs, axis=2)  # Add channel dimension (H, W, 1)
        else:
            # Convert to RGB and normalize
            obs = cv2.cvtColor(obs_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        return obs

    # Convenience helpers
    def render(self, overlay_box: bool = True, scale: float = 0.5) -> np.ndarray:
        vis = self.image_bgr.copy()
        if overlay_box:
            x_min, y_min, x_max, y_max = self.crop_box
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if scale != 1.0:
            vis = cv2.resize(vis, (int(self.W * scale), int(self.H * scale)))
        return vis

    def get_current_score(self) -> float:
        x_min, y_min, x_max, y_max = self.crop_box
        crop = self.image_bgr[y_min:y_max, x_min:x_max]
        return float(self.scorer.score_image(crop))
