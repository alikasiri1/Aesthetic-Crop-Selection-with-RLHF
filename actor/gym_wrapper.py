"""
Gymnasium wrapper for FrameSelectorEnv so it can be used with Stable-Baselines3 PPO.
"""
from typing import Tuple, Dict, Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl_env.frame_selector_env import FrameSelectorEnv
from scorer.aesthetic_scorer import AestheticScorerPipeline

class FrameSelectorGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        image_path: str,
        scorer: Optional[AestheticScorerPipeline] = None,
        downscale_hw: Tuple[int, int] = (128, 128),
        init_crop_hw: Tuple[int, int] = (224, 224),
        max_steps: int = 50,
        seed: int = 0,
        gray_mode: bool = False,
    ) -> None:
        super().__init__()
        self.gray_mode = gray_mode
        self.core_env = FrameSelectorEnv(
            image_path=image_path,
            scorer=scorer,
            downscale_hw=downscale_hw,
            init_crop_hw=init_crop_hw,
            max_steps=max_steps,
            seed=seed,
            gray_mode=gray_mode,
        )

        H, W = downscale_hw
        # Observation: (H, W, C) float32 in [0,1] - RGB or grayscale crop image
        channels = 1 if gray_mode else 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(H, W, channels), dtype=np.float32)
        # Action: discrete choices from core env
        self.action_space = spaces.Discrete(self.core_env.num_actions)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, random_init: bool = False):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.core_env.reset(random_init=random_init)
        return obs.astype(np.float32), info

    def step(self, action: int):
        obs, reward, done, truncated, info = self.core_env.step(int(action))
        # SB3 expects terminated/truncated for Gymnasium API
        terminated = bool(done)
        return obs.astype(np.float32), float(reward), terminated, bool(truncated), info

    def render(self):
        vis = self.core_env.render(overlay_box=True, scale=1.0)
        # Convert BGR to RGB
        return vis[:, :, ::-1]

    def close(self):
        return
