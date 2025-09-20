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
        step_pixels: int = 64, # 32 is better for smaller models
        size_step_pixels: int = 32,
        max_steps: int = 50,
        proposals: Optional[List[Tuple[int, int, int, int]]] = None,
        backbone: str = "efficientnet_b0",
        seed: int = 0,
        allow_resize: bool = False,
    ) -> None:
        assert os.path.exists(image_path), f"Image not found: {image_path}"
        self.rng = np.random.RandomState(seed)

        self.image_bgr = cv2.imread(image_path)
        if self.image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        self.H, self.W = self.image_bgr.shape[:2]

        self.scorer = scorer or AestheticScorerPipeline(backbone_name=backbone)
        self.downscale_hw = downscale_hw
        self.init_crop_hw = init_crop_hw
        self.step_pixels = step_pixels
        self.size_step_pixels = size_step_pixels
        self.max_steps = max_steps
        self.step_count = 0

        # Action space definition
        # 0: up, 1: down, 2: left, 3: right,
        # 4: wider, 5: narrower, 6: taller, 7: shorter,
        # 8: jump_to_best_proposal, 9: select
        self.num_actions = 10 if allow_resize else 6
        print(f"Number of actions: {self.num_actions}")

        # Initialize crop box (x, y, w, h)
        self.crop_box = self._init_centered_crop()

        # Proposals (optional): list of (x, y, w, h)
        self.proposals = proposals if proposals is not None else self._generate_proposals()

        # Cached downscaled image for observation
        self.obs_image = cv2.resize(self.image_bgr, (self.downscale_hw[1], self.downscale_hw[0]))

    # ------------------------ Core API ------------------------
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        self.crop_box = self._init_centered_crop()
        obs = self._build_observation()
        return obs, {"crop_box": self.crop_box}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        self._apply_action(action)
        self._clamp_crop_box()

        reward = self._compute_reward()
        done = action == 9  # select
        truncated = self.step_count >= self.max_steps

        obs = self._build_observation()
        info = {"crop_box": self.crop_box, "step": self.step_count}
        return obs, reward, done, truncated, info

    # ------------------------ Helpers ------------------------
    def _init_centered_crop(self) -> Tuple[int, int, int, int]:
        w, h = self.init_crop_hw[1], self.init_crop_hw[0]
        w = min(w, self.W)
        h = min(h, self.H)
        x = (self.W - w) // 2
        y = (self.H - h) // 2
        return (x, y, w, h)

    def _generate_proposals(self, k: int = 20) -> List[Tuple[int, int, int, int]]:
        # Use a simple grid of proposals
        cropper = ImageCropper(crop_size=self.init_crop_hw, stride=max(self.init_crop_hw[0] // 2, 32))
        crops = cropper.create_crops(self.image_bgr)
        # Score quickly and pick top-k as proposals
        scored = []
        for crop, box in crops:
            s = self.scorer.score_image(crop)
            scored.append((s, box))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [box for _, box in scored[:k]]

    def _apply_action(self, action: int) -> None:
        x, y, w, h = self.crop_box

        # Movement
        if action == 0:  # up
            y -= self.step_pixels
        elif action == 1:  # down
            y += self.step_pixels
        elif action == 2:  # left
            x -= self.step_pixels
        elif action == 3:  # right
            x += self.step_pixels

        # jump to best proposal
        elif action == 4:  # jump to best proposal
            if self.proposals:
                self.crop_box = self.proposals[0]
                return

        # select
        elif action == 5:  # select
            pass

        # if allow_resize is True:
        elif self.allow_resize:
            if action == 6:  # wider
                w += self.size_step_pixels
            elif action == 7:  # narrower
                w -= self.size_step_pixels
            elif action == 8:  # taller
                h += self.size_step_pixels
            elif action == 9:  # shorter
                h -= self.size_step_pixels
            else:
                raise ValueError(f"Invalid action: {action}")
        else:
            # if allow_resize=False and action is greater than 5
            if action > 5:
                raise ValueError(f"Invalid action: {action}")

        self.crop_box = (x, y, w, h)

    def _clamp_crop_box(self) -> None:
        x, y, w, h = self.crop_box
        w = max(32, min(w, self.W))
        h = max(32, min(h, self.H))
        x = max(0, min(x, self.W - w))
        y = max(0, min(y, self.H - h))
        self.crop_box = (x, y, w, h)

    def _compute_reward(self) -> float:
        x, y, w, h = self.crop_box
        crop = self.image_bgr[y : y + h, x : x + w]
        score_1_to_10 = self.scorer.score_image(crop)
        # Normalize to [0, 1] and optionally add small step penalty to encourage quicker selection
        reward = (score_1_to_10 - 1.0) / 9.0
        reward -= 0.005  # small per-step penalty
        return float(reward)

    def _build_observation(self) -> np.ndarray:
        # Downscaled RGB image in [0,1]
        img = cv2.cvtColor(self.obs_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        Hs, Ws = img.shape[:2]
        # Encode crop box as a mask channel
        mask = np.zeros((Hs, Ws, 1), dtype=np.float32)
        # Map crop box from original coords to downscaled grid
        scale_y = Hs / self.H
        scale_x = Ws / self.W
        x, y, w, h = self.crop_box
        xs, ys = int(x * scale_x), int(y * scale_y)
        ws, hs = max(1, int(w * scale_x)), max(1, int(h * scale_y))
        mask[ys : ys + hs, xs : xs + ws, 0] = 1.0
        obs = np.concatenate([img, mask], axis=2)  # shape (H, W, 4)
        return obs

    # Convenience helpers
    def render(self, overlay_box: bool = True, scale: float = 0.5) -> np.ndarray:
        vis = self.image_bgr.copy()
        if overlay_box:
            x, y, w, h = self.crop_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if scale != 1.0:
            vis = cv2.resize(vis, (int(self.W * scale), int(self.H * scale)))
        return vis

    def get_current_score(self) -> float:
        x, y, w, h = self.crop_box
        crop = self.image_bgr[y : y + h, x : x + w]
        return float(self.scorer.score_image(crop))
